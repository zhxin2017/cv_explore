from typing import Dict, Iterable, Callable

import sys
sys.path.append('..')
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from od import detr_dataset as ds
from common.config import val_annotation_file, val_img_od_dict_file, val_img_dir, train_annotation_file, train_img_od_dict_file, train_img_dir, img_size
from od.config import cid_to_name
from torchvision import transforms
from od import detr_model, anno
import random
import math


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


def examine_attn(img, extractor, n_head, device):
    img_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img_ = img_.to(device)
    with torch.no_grad():
        boxes_pred_xyxy, cls_logits_pred, _, _ = extractor.model(img_)
    boxes_pred_xyxy = boxes_pred_xyxy * max(img_size)
    cls_pred_sm = cls_logits_pred.softmax(-1)
    cls_pred = cls_pred_sm.argmax(-1)

    img = torch.permute(img, [0, 2, 3, 1])

    n_obj = min((cls_pred[0] > 0).sum().item(), 10)

    anchors = []
    boxes = []
    attns = []
    names = []

    if n_obj == 0:
        return img[0], anchors, boxes, names, attns, boxes_pred_xyxy, cls_pred

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

        x1_anchor, y1_anchor, x2_anchor, y2_anchor = extractor.model.decoder.anchors[obj_idx]
        x1_anchor = x1_anchor * max(img_size)
        y1_anchor = y1_anchor * max(img_size)
        x2_anchor = x2_anchor * max(img_size)
        y2_anchor = y2_anchor * max(img_size)
        anchors.append([x1_anchor, y1_anchor, x2_anchor, y2_anchor])

        q = extractor._features['decoder.decoder_layers.0.cross_attn.q_proj']
        k = extractor._features['decoder.decoder_layers.0.cross_attn.k_proj']

        lq = q.shape[1]
        lk = k.shape[1]

        q = q.view(lq, n_head, -1).transpose(0, 1)
        k = k.view(lk, n_head, -1).transpose(0, 1)

        attn = attention(q, k)
        attn = attn.view(n_head, 742, 32, 32)[:, obj_idx]

        attns.append(attn)

    return img[0], anchors, boxes, names, attns, n_head, boxes_pred_xyxy, cls_pred


def draw_attn(img, anchors, boxes, names, attns, n_head, show_row_per_obj = 1):
    n_obj = len(anchors)

    if n_obj == 0:
        return

    col = math.ceil((n_head + 1) // show_row_per_obj)

    fig, axes = plt.subplots(n_obj * show_row_per_obj, col, figsize=(n_obj * show_row_per_obj * 4, col * 4))

    for i in range(n_obj):

        axes[i * show_row_per_obj, 0].imshow(img)
        axes[i * show_row_per_obj, 0].axis('off')

        name = names[i]

        x1, y1, x2, y2 = boxes[i]
        x1_anchor, y1_anchor, x2_anchor, y2_anchor = anchors[i]
        attn = attns[i]

        axes[i * show_row_per_obj, 0].add_patch(
            plt.Rectangle((x1_anchor, y1_anchor), x2_anchor - x1_anchor, y2_anchor - y1_anchor, fill=False,
                          edgecolor='green', lw=2))
        axes[i * show_row_per_obj, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1))
        axes[i * show_row_per_obj, 0].text(x1 + 2, y1 + 16, name, color='red')

        for h in range(1, n_head + 1):
            axes[i * show_row_per_obj + h // col, h % col].axis('off')
            axes[i * show_row_per_obj + h // col, h % col].imshow(attn[h - 1].cpu().detach().numpy())
    # plt.pause(0)


def pred_and_show(img, model, device, thresh=0.6):
    img_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img_ = img_.to(device)
    with torch.no_grad():
        boxes_pred_xyxy, cls_logits_pred, _, _ = model(img_)
    boxes_pred_xyxy = boxes_pred_xyxy * max(img_size)
    cls_pred_sm = cls_logits_pred.softmax(-1)
    cls_pred = cls_pred_sm.argmax(-1)
    print(cls_pred_sm.shape)

    img = torch.permute(img, [0, 2, 3, 1])
    bsz = img.shape[0]

    fig, axes = plt.subplots(1, bsz, figsize=(40, 40))
    # fig, axes = plt.subplots(1, bsz)

    for b in range(bsz):
        axes[b].imshow(img[b])
        axes[b].axis('off')

        for i in range(model.decoder.n_anchor):
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


