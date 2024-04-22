from typing import Dict, Iterable, Callable
from common.config import patch_size

import sys

sys.path.append('..')
import torch
from torch import nn, Tensor

from matplotlib import pyplot as plt
from common.config import img_size
from detr.config import cid_to_name
from torchvision import transforms
import random


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict(self.model.named_modules())[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = self.model(x)
        return out


def attention(q, k):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = torch.nn.functional.softmax(q @ k / d ** 0.5, dim=-1)
    return attn


def examine_attn(img, extractor, n_head, device,
                 q1_module_name, k1_module_name,
                 q2_module_name, k2_module_name,
                 anchors=None):
    img_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img_ = img_.to(device)
    with torch.no_grad():
        if anchors is not None:
            boxes_pred_xyxy, cls_logits_pred, _, _  = extractor.model(img_, anchors)
            anchors_expanded = anchors.view(anchors.shape[0], 1, -1).repeat(1, 6, 1).reshape(anchors.shape[0] * 6, -1)
        else:
            boxes_pred_xyxy, cls_logits_pred, _, _, _, _, _, _ = extractor.model(img_)
            # boxes_pred_xyxy, cls_logits_pred, _, _ = extractor.model(img_)
    anchors_ = []
    anchors_new = []
    boxes = []
    attns1 = []
    attns2 = []
    names = []

    if boxes_pred_xyxy is None:
        return img[0], anchors_, anchors_new, boxes, names, attns1, attns2, n_head, boxes_pred_xyxy, None

    boxes_pred_xyxy = boxes_pred_xyxy * max(img_size)
    cls_pred_sm = cls_logits_pred.softmax(-1)
    cls_pred = cls_pred_sm.argmax(-1)
    # cls_pred = cls_pred_sm[..., 1:].argmax(-1)
    # cls_pred = (cls_pred + 1) * (torch.max(cls_pred_sm[..., 1:], dim=-1)[0] > .35)

    img = torch.permute(img, [0, 2, 3, 1])
    H, W = img.shape[1], img.shape[2]
    grid_size_y = H // patch_size
    grid_size_x = W // patch_size

    n_obj = min((cls_pred[0] > 0).sum().item(), 20)

    if n_obj == 0:
        return img[0], anchors_, anchors_new, boxes, names, attns1, attns2, n_head, boxes_pred_xyxy, cls_pred

    pos_indices = torch.where(cls_pred[0] > 0)[0].tolist()
    pos_indices = random.sample(pos_indices, n_obj)

    for i in range(n_obj):
        obj_idx = pos_indices[i]
        cls = cls_pred[0][obj_idx].item()
        name = cid_to_name.get(cls)
        names.append(name)

        x1, y1, x2, y2 = boxes_pred_xyxy[0, obj_idx]
        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
        boxes.append([x1, y1, x2, y2])

        q1 = extractor._features[q1_module_name]
        k1 = extractor._features[k1_module_name]

        lq = q1.shape[1]
        lk = k1.shape[1]

        q1 = q1.view(lq, n_head, -1).transpose(0, 1)
        k1 = k1.view(lk, n_head, -1).transpose(0, 1)

        q2 = extractor._features[q2_module_name]
        k2 = extractor._features[k2_module_name]
        q2 = q2.view(lq, n_head, -1).transpose(0, 1)
        k2 = k2.view(lk, n_head, -1).transpose(0, 1)
        # q = q[:, lx:].view(q.shape[1] - lx, n_head, -1).transpose(0, 1)
        # k = k[:, :lx].view(lx, n_head, -1).transpose(0, 1)

        attn1 = attention(q1, k1)
        attn2 = attention(q2, k2)

        attns1.append(attn1[:, obj_idx].view(n_head, grid_size_y, grid_size_x))
        attns2.append(attn2[:, obj_idx].view(n_head, grid_size_y, grid_size_x))

        # print(anchor_shift)
        if anchors is not None:
            # print('============')
            # print(anchor_shift)
            # print(anchors[obj_idx])
            # print(anchor_shift + anchors[obj_idx])
            # print((anchor_shift + anchors[obj_idx]) * max(img_size))
            x1_anchor, y1_anchor, x2_anchor, y2_anchor = anchors_expanded[obj_idx]
            x1_anchor = (x1_anchor * max(img_size)).item()
            y1_anchor = (y1_anchor * max(img_size)).item()
            x2_anchor = (x2_anchor * max(img_size)).item()
            y2_anchor = (y2_anchor * max(img_size)).item()
            anchors_.append([x1_anchor, y1_anchor, x2_anchor, y2_anchor])

            anchor_shift = torch.tanh(extractor._features['decoder.anchor_shift_reg'])[0, obj_idx] / 3
            x1_new = (anchor_shift[0] * max(img_size) + x1_anchor).item()
            y1_new = (anchor_shift[1] * max(img_size) + y1_anchor).item()
            x2_new = (anchor_shift[2] * max(img_size) + x2_anchor).item()
            y2_new = (anchor_shift[3] * max(img_size) + y2_anchor).item()
            anchors_new.append([x1_new, y1_new, x2_new, y2_new])

    return img[0], anchors_, anchors_new, boxes, names, attns1, attns2, n_head, boxes_pred_xyxy, cls_pred


def draw_attn(img, anchors, anchors_new, boxes, names, attns, n_head):
    n_obj = len(boxes)

    if n_obj == 0:
        return

    fig, axes = plt.subplots(n_obj, n_head + 2, figsize=((n_head + 2) * 5, n_obj * 5))

    H, W = img.shape[0], img.shape[1]
    grid_size_y = H // patch_size
    grid_size_x = W // patch_size

    for i in range(n_obj):

        if n_obj == 1:
            axes_ = axes
        else:
            axes_ = axes[i]

        axes_[0].imshow(img.cpu().numpy())
        axes_[0].axis('off')

        if len(anchors) > 0:
            x1_anchor, y1_anchor, x2_anchor, y2_anchor = anchors[i]
            axes_[0].add_patch(
                plt.Rectangle((x1_anchor, y1_anchor), x2_anchor - x1_anchor, y2_anchor - y1_anchor, fill=False,
                              edgecolor='green', lw=4))

        if len(anchors_new) > 0:
            x1_anchor_new, y1_anchor_new, x2_anchor_new, y2_anchor_new = anchors_new[i]
            # x1_anchor = x1_anchor.item()
            # y1_anchor = y1_anchor.item()
            # x2_anchor = x2_anchor.item()
            # y2_anchor = y2_anchor.item()
            # print(anchors_new[i])
            axes_[0].add_patch(
                plt.Rectangle((x1_anchor_new, y1_anchor_new), x2_anchor_new - x1_anchor_new, y2_anchor_new - y1_anchor_new, fill=False,
                              edgecolor='yellow', lw=2))

        name = names[i]
        x1, y1, x2, y2 = boxes[i]
        axes_[0].add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1))
        axes_[0].text(x1 + 2, y1 + 16, name, color='red')

        attn = attns[i]
        for h in range(1, n_head + 1):
            axes_[h].axis('off')
            axes_[h].imshow(attn[h - 1].cpu().detach().numpy())

        attn_= attn.mean(dim=0).view(grid_size_y, grid_size_x, 1, 1).repeat(1, 1, patch_size, patch_size).contiguous()
        attn_ = attn_.permute(0, 2, 1, 3).reshape(grid_size_y * patch_size, grid_size_x * patch_size, 1)
        axes_[n_head + 1].axis('off')
        axes_[n_head + 1].imshow((attn_ / attn_.max() * img).cpu().detach().numpy())
        axes_[n_head + 1].add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1))
    # plt.pause(0)


def pred_and_show(img, model, device, n_anchor, thresh=0.6):
    img_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img_ = img_.to(device)
    with torch.no_grad():
        boxes_pred_xyxy, cls_logits_pred, _, _ = model(img_)
    boxes_pred_xyxy = boxes_pred_xyxy * max(img_size)
    cls_pred_sm = cls_logits_pred.softmax(-1)
    cls_pred = cls_pred_sm.argmax(-1)

    img = torch.permute(img, [0, 2, 3, 1])
    bsz = img.shape[0]

    fig, axes = plt.subplots(1, bsz, figsize=(40, 40))
    # fig, axes = plt.subplots(1, bsz)

    for b in range(bsz):
        axes[b].imshow(img[b])
        axes[b].axis('off')

        for i in range(n_anchor):
            cls = cls_pred[b, i].item()
            if cls == 0:
                continue
            if cls_pred_sm[b, i, cls] < thresh:
                continue

            name = cid_to_name.get(cls)

            x1, y1, x2, y2 = boxes_pred_xyxy[b, i]
            x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()

            axes[b].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1))
            axes[b].text(x1 + 2, y1 + 16, name)

    return boxes_pred_xyxy, cls_pred
