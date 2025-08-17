import os
import torch
from torch import optim
from detr import detr_seg_model, match_seg, eval
from config import img_root_dir, xml_root_dir, filelist_files, img_h, img_w, downsample_seg, \
    batch_size, num_enc_layer, num_dec_layer, dmodel, dhead, num_query, epoch
from data.voc import VocDataset, collate_fn
import time
from data.classes import voc_classes
import numpy as np
from torch.utils.data import DataLoader

num_classes = len(voc_classes)

detr = detr_seg_model.DETR(dmodel, dhead, num_enc_layer, num_dec_layer, num_query, num_classes)

ckpt = 'detr_seg_epoch_2_batch_1000.pt'
detr.load_state_dict(torch.load(ckpt, map_location='cpu'))

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

optimizer = optim.Adam(detr.parameters(), lr=1e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

detr.to(device)
dataset = VocDataset(img_root_dir, xml_root_dir, filelist_files, [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

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
        boxes_gt = torch.stack(boxes_gt, dim=0)

        gt_pos_mask = (cids_gt > 0)
        gt_pos_mask = gt_pos_mask.view(n, 1, num_query) * 1
        num_pos_cls = torch.sum(gt_pos_mask, dim=-1)

        seg_logits = detr(imgs)
        seg_pred_pos_mask = (torch.argmax(seg_logits, dim=-1, keepdim=True) > 0) * 1

        seg_gt_pos_mask = torch.zeros_like(seg_pred_pos_mask, device=device)
        for k in range(n):
            for p in range(num_pos_cls[k]):
                x1, y1, x2, y2 = boxes_gt[k, p] / downsample_seg
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                seg_gt_pos_mask[k, p, y1:y2, x1:x2, 0] = 1
        
        cids_gt = cids_gt.view(n, num_query, 1, 1, 1) * seg_gt_pos_mask

        rows, cols = match_seg.assign_query(cids_gt, seg_logits, gt_pos_mask)
        cols = torch.tensor(np.stack(cols), device=device)

        gt_matched_indices_batch = torch.arange(n, device=device).view(n, 1). \
            expand(n, num_query).contiguous().view(n * num_query)

        gt_matched_indices_query = cols.view(n * num_query)

        # query_pos_mask = (cols < num_pos_cls).view(n * num_query) * 1
        # n_pos = query_pos_mask.sum()

        seg_prob = torch.softmax(seg_logits, dim=-1)
        # overlapping loss
        overlapping_mask = (torch.sum(seg_gt_pos_mask, dim=1, keepdim=True) > 1) * 1
        overlapping_loss = (torch.sum(seg_prob * overlapping_mask, dim=1) - 1)**2
        overlapping_loss = torch.sum(overlapping_loss) / (torch.sum(overlapping_mask) + 1e-5) / num_query
        
        # cls loss
        seg_gt_pos_mask = seg_gt_pos_mask[(gt_matched_indices_batch, gt_matched_indices_query)]
        cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        flat_dim = n * num_query * img_h // downsample_seg * img_w // downsample_seg
        cids_gt = cids_gt.view(flat_dim)
        seg_logits = seg_logits.view(flat_dim, -1)
        cls_loss = loss_fn(seg_logits, cids_gt)

        seg_pred_pos_mask = seg_pred_pos_mask.view(n * num_query, img_h // downsample_seg, img_w // downsample_seg, 1)
        cls_loss_mask = 1 - (1 - seg_pred_pos_mask) * seg_gt_pos_mask
        cls_loss_mask = cls_loss_mask.view(flat_dim)
        cls_loss = torch.sum(cls_loss * cls_loss_mask) / (torch.sum(cls_loss_mask) + 1e-5)

        # positive sum loss
        seg_gt_pos_mask = seg_gt_pos_mask.view(flat_dim, 1)
        seg_prob = seg_prob.view(flat_dim, -1)
        pos_sum = torch.sum(seg_prob[..., 1:] * seg_gt_pos_mask) / (torch.sum(seg_gt_pos_mask) + 1e-5)
        pos_sum_loss = 1 - pos_sum

        accu, recall = eval.eval_pred2(seg_prob.argmax(dim=-1), cids_gt)
        loss = cls_loss + pos_sum_loss + overlapping_loss

        optimizer.zero_grad()
        t = time.time()
        loss.backward()
        t_bp = time.time() - t
        # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
        optimizer.step()

        print(f'|epoch {e + 1}/{epoch}|batch {j + 1}|'
                f'cl {cls_loss.detach().item():.3f}|'
                f'sl {pos_sum_loss.detach().item():.3f}|'
                f'ol {overlapping_loss.detach().item():.3f}|'
                f'ac {accu:.3f}|rc {recall:.3f}|'
                )
        if (j + 1) % 500 == 0:
            torch.save(detr.state_dict(), f'detr_seg_epoch_{e + 1}_batch_{j + 1}.pt')
            print(f'saved detr_seg_epoch_{e + 1}_batch_{j + 1}.pt')


if __name__ == '__main__':
    for i in range(epoch):
        train_one_epoch(i)
