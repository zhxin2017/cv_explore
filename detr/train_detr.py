import os
import torch
from torch import optim
from detr import detr_model, match, eval
from config import img_root_dir, xml_root_dir, filelist_files, img_h, img_w, \
    batch_size, num_enc_layer, num_dec_layer, dmodel, dhead, num_query, epoch
import focalloss
from data.voc import VocDataset, collate_fn
import time
from data.classes import voc_classes
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

h = img_h // 16
w = img_w // 16
num_classes = len(voc_classes)

detr = detr_model.DETR(dmodel, dhead, h, w, num_enc_layer, num_dec_layer, num_query, num_classes)

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

optimizer = optim.Adam(detr.parameters(), lr=1e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('mps')

detr.to(device)
dataset = VocDataset(img_root_dir, xml_root_dir, filelist_files, [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

box_resize_factor = torch.tensor([img_w, img_h, img_w, img_h], device=device)
box_resize_factor = box_resize_factor.view([1, 1, 4])
def train_one_epoch(e):
    for j, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        n = len(targets)
        cids_gt = []
        boxes_gt = []
        for k in range(n):
            cids_per_img = targets[k]['cids'].to(torch.long).to(device)
            pad_num = num_query - len(cids_per_img)
            cids_padding_per_img = torch.zeros([pad_num], dtype=torch.long, device=device)
            cids_per_img = torch.concat([cids_per_img, cids_padding_per_img], dim=0)
            cids_gt.append(cids_per_img)
            boxes_per_img = targets[k]['boxes'].to(torch.float).to(device)
            boxes_padding_per_img = torch.zeros([pad_num, 4], dtype=torch.float, device=device)
            boxes_per_img = torch.concat([boxes_per_img, boxes_padding_per_img], dim=0)
            boxes_gt.append(boxes_per_img)

        cids_gt = torch.stack(cids_gt, dim=0)
        print(cids_gt.shape)
        boxes_gt = torch.stack(boxes_gt, dim=0)
        print(boxes_gt.shape)
        boxes_gt = boxes_gt / box_resize_factor

        boxes_pred, cls_logits = detr(imgs)
        gt_pos_mask = (cids_gt > 0)
        gt_pos_mask = gt_pos_mask.view(n, 1, num_query) * 1

        rows, cols = match.assign_query(boxes_gt, boxes_pred, cids_gt, cls_logits, gt_pos_mask)
        cols = torch.tensor(np.stack(cols), device=device)

        num_pos_cls = torch.sum(gt_pos_mask, dim=-1)

        gt_matched_indices_batch = torch.arange(n, device=device).view(n, 1). \
            expand(n, num_query).contiguous().view(n * num_query)

        gt_matched_indices_query = cols.view(n * num_query)

        query_pos_mask = (cols < num_pos_cls).view(n * num_query) * 1
        n_pos = query_pos_mask.sum()

        # cls loss
        cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        cls_logits = cls_logits.view(n * num_query, -1)
        cls_loss = loss_fn(cls_logits, cids_gt) * query_pos_mask
        cls_loss = cls_loss.sum() / (n_pos + 1e-5)

        cls_pred = cls_logits.argmax(dim=-1)
        accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt, query_pos_mask)

        # box loss
        boxes_gt = boxes_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        box_loss = distance_box_iou_loss(boxes_pred.view(n * num_query, -1), boxes_gt.view(n * num_query, -1))
        box_loss = box_loss * query_pos_mask
        box_loss = box_loss.sum() / (n_pos + 1e-5)

        loss = cls_loss * 5 + box_loss
        optimizer.zero_grad()
        t = time.time()
        loss.backward()
        t_bp = time.time() - t
        # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
        optimizer.step()

        print(f'|epoch {e + 1}/{epoch}|batch {j}|'
                f'cl {cls_loss.detach().item() * 1000:.3f}|'
                f'bl {box_loss.detach().item():.3f}|'
                f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|')


if __name__ == '__main__':
    for i in range(epoch):
        train_one_epoch(i)
