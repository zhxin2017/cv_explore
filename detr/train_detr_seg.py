import os
import torch
from torch import optim
from detr import detr_seg_model, match_seg, eval
from config import img_root_dir, xml_root_dir, filelist_files, img_h, img_w, ds1, ds2, \
    batch_size, num_enc_layer, num_dec_layer, dmodel, dhead, num_query, epoch
from data.voc import VocDataset, collate_fn
import time
from data.classes import voc_classes
import numpy as np
from torch.utils.data import DataLoader

num_classes = len(voc_classes)
ds = ds1 * ds2
fm_h, fm_w = img_h // ds, img_w // ds

detr = detr_seg_model.DETR(dmodel, dhead, num_enc_layer, num_dec_layer, num_query, num_classes)

ckpt = 'detr_seg_epoch_1_batch_19000.pt'
# detr.load_state_dict(torch.load(ckpt, map_location='cpu'))

mean_celoss = float(np.log(num_classes))

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

optimizer = optim.Adam(detr.parameters(), lr=1e-3 / batch_size)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

detr.to(device)
# dataset = VocDataset(img_root_dir, xml_root_dir, filelist_files, [img_h, img_w])
dataset = VocDataset(img_root_dir, xml_root_dir, ['minibatch.txt'], [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)

cross_entropy = torch.nn.CrossEntropyLoss(reduction='none').to(device)
bce = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)

def train_one_epoch(e):
    for j, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = len(targets)
        cids_gt = []
        boxes_gt = []
        for m in range(batch_size):
            cids_per_img = targets[m]['cids'].to(torch.long).to(device)
            pad_num = num_query - len(cids_per_img)
            cids_padding_per_img = torch.zeros([pad_num], dtype=torch.long, device=device)
            cids_per_img = torch.concat([cids_per_img, cids_padding_per_img], dim=0)
            cids_gt.append(cids_per_img)
            boxes_per_img = targets[m]['boxes'].to(torch.float).to(device)
            boxes_padding_per_img = torch.zeros([pad_num, 4], dtype=torch.float, device=device)
            boxes_per_img = torch.concat([boxes_per_img, boxes_padding_per_img], dim=0)
            boxes_gt.append(boxes_per_img)

        cids_gt = torch.stack(cids_gt, dim=0)
        boxes_gt = torch.stack(boxes_gt, dim=0)

        gt_pos_mask = (cids_gt > 0)
        gt_pos_mask = gt_pos_mask.view(batch_size, 1, num_query) * 1
        num_pos_cls = torch.sum(gt_pos_mask, dim=-1)

        seg_logits, cls_logits = detr(imgs)

        seg_pred = (seg_logits > 0) * 1

        seg_gt = torch.zeros([batch_size, num_query, fm_h, fm_w], device=device)
        for m in range(batch_size):
            for q in range(num_pos_cls[m]):
                x1, y1, x2, y2 = boxes_gt[m, q] / ds
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                seg_gt[m, q, y1:y2, x1:x2] = 1
        seg_gt = seg_gt.view(batch_size, num_query, fm_h * fm_w)
        rows, cols = match_seg.assign_query(seg_gt, seg_pred, cids_gt, cls_logits, gt_pos_mask)
        cols = torch.tensor(np.stack(cols), device=device)

        gt_matched_indices_batch = torch.arange(batch_size, device=device).view(batch_size, 1). \
            expand(batch_size, num_query).contiguous().view(batch_size * num_query)

        gt_matched_indices_query = cols.view(batch_size * num_query)

        query_pos_mask = (cols < num_pos_cls).view(batch_size, num_query, 1) * 1

        seg_prob = torch.sigmoid(seg_logits) * query_pos_mask
        seg_pred = seg_pred * query_pos_mask

        # overlapping loss
        num_grid = fm_h * fm_w
        ovlp_mask = torch.sum(seg_pred, dim=1, keepdim=True) > 1 # shape(bsz, 1, fm_h * fm_w)
        ovlp_mask_sum = torch.sum(ovlp_mask, dim=(1, 2)).view(batch_size, 1)
        ovlp_sum = torch.sum(seg_prob * ovlp_mask, dim=1) # shape(bsz, fm_h * fm_w)
        ovlp_mask = ovlp_mask.view(batch_size, num_grid)
        ovlp_loss = torch.sum((ovlp_sum - 1) * ovlp_mask, dim=1, keepdim=True)
        ovlp_loss = ovlp_loss * (ovlp_mask_sum > 0)
        ovlp_loss = ovlp_loss / (ovlp_mask_sum * num_pos_cls + 1e-5)
        ovlp_loss = torch.mean(ovlp_loss)

        flat_dim = batch_size * num_query
        query_pos_mask = query_pos_mask.view(flat_dim, 1)

        # seg negative loss
        seg_gt = seg_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        seg_logits = seg_logits.view(flat_dim, num_grid)
        seg_neg_loss = bce(seg_logits, seg_gt)
        seg_neg_mask = (1 - seg_gt) * query_pos_mask
        seg_neg_loss = torch.sum(seg_neg_loss * seg_neg_mask) / (torch.sum(seg_neg_mask) + 1e-5)

        # seg positive loss
        seg_pos_ratio = torch.sum(seg_prob.view(flat_dim, num_grid) * seg_gt) / (torch.sum(seg_gt) + 1e-5)
        if seg_pos_ratio > 0.8:
            seg_pos_loss = torch.tensor(0.0, device=device)
        else:
            seg_pos_loss = -torch.log(seg_pos_ratio + 1e-5)

        # iou
        seg_pred = seg_pred.view(flat_dim, num_grid)
        seg_inter = torch.sum(seg_gt * seg_pred)
        seg_union = torch.sum(seg_gt * query_pos_mask + seg_pred) - seg_inter
        iou = seg_inter / (seg_union + 1e-5)

        # cls loss
        cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        cls_logits = cls_logits.view(flat_dim, num_classes)
        cls_loss = cross_entropy(cls_logits, cids_gt)
        query_pos_mask = query_pos_mask.view(-1)
        # print(cls_logits[query_pos_mask > 0])
        cls_loss = cls_loss * (0.01 + query_pos_mask)
        cls_loss = cls_loss.mean()
        cls_pred = torch.argmax(cls_logits, dim=-1)
        accu, recall, f1, tp = eval.eval_pred(cls_pred, cids_gt, query_pos_mask.view(-1))
        
        loss = cls_loss / mean_celoss + seg_neg_loss + seg_pos_loss + ovlp_loss

        optimizer.zero_grad()
        t = time.time()
        loss.backward()
        t_bp = time.time() - t
        # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
        optimizer.step()

        print(f'|epoch {e + 1}/{epoch}|batch {j + 1}|'
                f'cl {cls_loss.detach().item():.3f}|'
                f'pl {seg_pos_loss.detach().item():.3f}|'
                f'nl {seg_neg_loss.detach().item():.3f}|'
                f'ol {ovlp_loss.detach().item():.3f}|'
                f'iou {iou.item():.3f}|'
                f'ac {accu:.3f}|rc {recall:.3f}|f1 {f1:.3f}|tp {tp.item()}|'
                )
        if (j + 1) % 2000 == 0:
            torch.save(detr.state_dict(), f'detr_seg_epoch_{e + 1}_batch_{j + 1}.pt')
            print(f'saved detr_seg_epoch_{e + 1}_batch_{j + 1}.pt')


if __name__ == '__main__':
    for i in range(epoch):
        train_one_epoch(i)
