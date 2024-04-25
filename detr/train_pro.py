import math
import os
import random

import torch
from torch import optim
from scipy.optimize import linear_sum_assignment
import sys

sys.path.append('..')
from detr import anno, detr_pro_dataset, detr_pro
from common.config import train_annotation_file, train_img_od_dict_file, patch_size, max_grid_len, max_img_len, \
    train_pro_bsz, model_save_dir, model_save_stride, device_type, train_img_dir
import time
from datetime import datetime
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms


device = torch.device(device_type)


model = detr_pro.DETR(dmodel=160, dhead=32,n_enc_layer=16, n_dec_layer=6, exam_diff=True)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')

ce = torch.nn.CrossEntropyLoss()

@torch.no_grad()
def assign_query(boxes_gt, boxes_pred, cids_gt, cls_pred):
    n_obj = len(boxes_gt)
    n_query = boxes_pred.shape[0]
    boxes_pred = boxes_pred.view(n_query, 1, 4).expand(n_query, n_obj, 4)
    boxes_gt = boxes_gt.view(1, n_obj, 4).expand(n_query, n_obj, 4)
    iouloss = distance_box_iou_loss(boxes_pred, boxes_gt)
    d_logits = cls_pred.shape[-1]
    cls_pred = cls_pred.view(n_query, 1, d_logits).expand(n_query, n_obj, d_logits) \
        .contiguous().view(n_query * n_obj, d_logits)
    cids_gt = cids_gt.view(1, n_obj).expand(n_query, n_obj).contiguous().view(n_query * n_obj)
    cls_loss = torch.nn.functional.cross_entropy(cls_pred, cids_gt, reduction='none').view(n_query, n_obj)

    ratio = cls_loss.mean() / iouloss.mean()

    total_loss = iouloss + cls_loss / ratio
    rows, cols = linear_sum_assignment(total_loss.detach().cpu().numpy())
    cols = cols.tolist()
    rows = rows.tolist()
    return rows, cols


def train(epoch, batch_size, population, num_sample, random_shift=True):
    ds = detr_pro_dataset.DetrProDS(train_img_dir, dicts, sample_num=population, random_flip='random')
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=detr_pro_dataset.collate_fn, shuffle=False)
    for i in range(epoch):
        for j, (img_batch, masks_batch, boxes_gt_xyxy_batch, cids_gt_batch, img_id_batch) in enumerate(dl):
            bsz, c, H, W = img_batch.shape
            h, w = H // patch_size, W // patch_size
            if random_shift:
                x_shift = random.randint(0, max_grid_len - w)
                y_shift = random.randint(0, max_grid_len - h)
                box_x_shift = x_shift * patch_size / max_img_len
                box_y_shift = y_shift * patch_size / max_img_len
            else:
                x_shift = y_shift = box_x_shift = box_y_shift = 0

            boxes_gt_xyxy_batch = [boxes_gt + torch.tensor([[box_x_shift, box_y_shift, box_x_shift, box_y_shift]])
                                   for boxes_gt in boxes_gt_xyxy_batch]
            img_id_batch = [(6 - len(img_id)) * ' ' + img_id for img_id in img_id_batch]
            print(f'{num_sample}|{i + 1}/{epoch}|{j}|id {" ".join(img_id_batch)}'
                  f'|{H}*{W}|{datetime.strftime(datetime.now(), "%m.%d %H:%M:%S")}', end="")
            # forward
            img_batch = img_batch.to(device)
            if masks_batch is not None:
                masks_batch = masks_batch.to(device)
            img_batch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_batch)

            (boxes_pred_xyxy_batch, cls_logits_pred_batch, src_cls_pos_loss, src_cls_neg_loss,
             src_cls_recall, src_cls_accu, sampled_pos_cids, enc_diff, logits_diff) = (
                model(img_batch, x_shift, y_shift, masks=masks_batch,
                      cids_gt_batch=cids_gt_batch, boxes_gt_batch=boxes_gt_xyxy_batch))
            sampled_objs_indices = [[p for p in range(len(cids_gt_batch[b])) if cids_gt_batch[b][p] in sampled_pos_cids[b]] for b in range(bsz)]

            n_query = cls_logits_pred_batch.shape[1]
            print(f'|nq {" " * (3 - len(str(n_query)))}{n_query}', end="")
            cls_pred_batch = torch.argmax(cls_logits_pred_batch, dim=-1)

            # loss
            cls_pos_loss_batch = 0
            cls_neg_loss_batch = 0
            box_loss_batch = 0
            t_match = 0
            n_pos_batch = 0
            n_neg_batch = 0
            tp = 0
            tn = 0

            for b in range(bsz):
                cids_gt = torch.tensor(cids_gt_batch[b], dtype=torch.long, device=device)[sampled_objs_indices[b]]
                boxes_gt = boxes_gt_xyxy_batch[b][sampled_objs_indices[b]]
                boxes_gt = boxes_gt.to(device)

                t = time.time()
                rows, cols = assign_query(boxes_gt, boxes_pred_xyxy_batch[b], cids_gt, cls_logits_pred_batch[b])
                t_match += (time.time() - t)

                n_pos = len(cids_gt)
                n_pos_batch += n_pos
                n_neg = n_query - n_pos
                n_neg_batch += n_neg
                # cls_loss = focalloss.focal_loss(cls_logits_pred, cids_gt_batch, ce_alpha, gamma=gamma).mean()
                cls_pos_loss = ce(cls_logits_pred_batch[b, rows], cids_gt[cols])
                cls_pos_loss_batch += cls_pos_loss * n_pos
                tp += torch.sum(cls_pred_batch[b, rows] == cids_gt[cols])

                cls_query_neg_indices = [i for i in range(n_query) if i not in rows]
                cids_gt_neg = torch.zeros(n_neg, dtype=torch.long, device=device)
                cls_neg_loss = ce(cls_logits_pred_batch[b, cls_query_neg_indices], cids_gt_neg)
                cls_neg_loss_batch += (cls_neg_loss * n_neg)
                tn += torch.sum(cls_pred_batch[b, cls_query_neg_indices] == cids_gt_neg)

                box_loss = distance_box_iou_loss(boxes_pred_xyxy_batch[b, rows], boxes_gt, reduction='mean')
                box_loss_batch += box_loss * n_pos

            # accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt_batch, query_pos_mask)
            recall = tp / n_pos_batch
            accu = (tp + tn) / (n_pos_batch + n_neg_batch)

            cls_pos_loss_batch = cls_pos_loss_batch / n_pos_batch
            cls_neg_loss_batch = cls_neg_loss_batch / n_neg_batch
            box_loss_batch = box_loss_batch / n_pos_batch
            loss = cls_pos_loss_batch + cls_neg_loss_batch * 5 + box_loss_batch * 10 + src_cls_pos_loss + src_cls_neg_loss
            # print(f'loss size {sys.getsizeof(loss)}', end='|')
            t = time.time()
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
            t_bp = time.time() - t

            if isinstance(src_cls_neg_loss, torch.Tensor):
                src_cls_neg_loss = src_cls_neg_loss.detach().item()

            print(f'|qpl {cls_pos_loss_batch.detach().item() * 1000:.3f}'
                  f'|qnl {cls_neg_loss_batch.detach().item() * 1000:.3f}'
                  f'|bl {box_loss_batch.detach().item():.3f}'
                  f'|gpl {src_cls_pos_loss.detach().item() * 1000:.3f}'
                  f'|gnl {src_cls_neg_loss * 1000:.3f}'
                  f'|grc {src_cls_recall:.3f}'
                  f'|gac {src_cls_accu:.3f}'
                  f'|qac {accu:.3f}'
                  f'|qrc {recall:.3f}: {tp}/{n_pos_batch}'
                  # f'|dif {enc_diff:.3f} {logits_diff:.3f}'
                  f'|tmch {t_match:.3f}|tbp {t_bp:.3f}'
                  # f'|match {matched_pos_gt}|'
                  )


if __name__ == '__main__':
    model_files = os.listdir(model_save_dir)
    model_files = [f for f in model_files if f.endswith('.pt') and f.startswith('od_pro')]
    if len(model_files) == 0:
        model_path_old = None
        latest_version = 0
    else:
        versions = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_version = max(versions)
        model_path_old = f'{model_save_dir}/od_pro_{latest_version}.pt'
        saved_state = torch.load(model_path_old, map_location=device)
        model.load_state_dict(saved_state)
        # state = model.state_dict()
        # for k in state:
        #     if k in saved_state:
        #         state[k] = saved_state[k]
        # model.load_state_dict(state)
    for i in range(5000):
        n_smp = latest_version + 1 + i
        ts = time.time()
        train(1, batch_size=train_pro_bsz, population=1000, num_sample=n_smp, random_shift=True)
        te = time.time()
        print(f'----------------------used {te - ts:.3f} secs---------------------------')
        # train(400, batch_size=1, population=2, num_sample=i, weight_recover=0, gamma=4)
        # train(2, batch_size=2, population=2, num_sample=i, weight_recover=.5, gamma=2)
        if n_smp % model_save_stride == 0:
            model_path_new = f'{model_save_dir}/od_pro_{n_smp}.pt'
            torch.save(model.state_dict(), model_path_new)
            if model_path_old is not None:
                os.remove(model_path_old)
            model_path_old = model_path_new
        # break
