import math
import os
import torch
from torch import optim
from scipy.optimize import linear_sum_assignment
import sys

sys.path.append('..')
from detr import anno, detr_pro_dataset, detr_pro, match, eval
from common.config import train_annotation_file, train_img_od_dict_file, max_img_size, patch_size, \
    train_base_bsz, model_save_dir, model_save_stride, device_type, train_img_dir
from detr.config import loss_weights
import focalloss
import time
import match
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device(device_type)

model = detr_pro.DETR(d_cont=256, d_head=64, d_src_pos=64, d_tgt_pos=64, n_enc_layer=16, n_dec_layer=6, exam_diff=True)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')

ce = torch.nn.CrossEntropyLoss()


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
    # total_loss[total_loss == torch.nan] = 1e8
    rows, cols = linear_sum_assignment(total_loss.detach().cpu().numpy())
    cols = cols.tolist()
    rows = rows.tolist()
    return rows, cols

grid_area = patch_size**2

def train(epoch, batch_size, population, num_sample, weight_recover=0.5, gamma=4):
    ds = detr_pro_dataset.DetrProDS(train_img_dir, dicts, sample_num=population, random_flip='none')
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=detr_pro_dataset.collate_fn, shuffle=False)
    for i in range(epoch):
        for j, (img_batch, masks_batch, boxes_gt_xyxy_batch, cids_gt_batch, img_id_batch) in enumerate(dl):
            bsz, c, H, W = img_batch.shape
            # forward
            img_batch = img_batch.to(device)
            # masks_batch = masks_batch.to(device)
            img_batch = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_batch)
            cids_gt_reduced = [list(set(c)) for c in cids_gt_batch]
            cids_gt_sample_num_batch = []

            for b in range(bsz):
                cids_gt_box_area = {c: 0 for c in cids_gt_reduced[b]}
                for o in range(len(cids_gt_batch[b])):
                    cid = cids_gt_batch[b][o]
                    x1, y1, x2, y2 = boxes_gt_xyxy_batch[b][o]
                    obj_area = (x2 - x1) * (y2 - y1)
                    cids_gt_box_area[cid] = cids_gt_box_area[cid] + obj_area

                cids_gt_sample_num = [math.ceil(cids_gt_box_area[c] / grid_area / 8) for c in cids_gt_reduced[b]]
                cids_gt_sample_num.append(max(int((H * W - sum(cids_gt_box_area.values())) / grid_area), 1))  # neg num
                cids_gt_sample_num_batch.append(cids_gt_sample_num)

            boxes_pred_xyxy_batch, cls_logits_pred_batch, src_cls_loss, src_cls_recall, src_cls_accu, \
                enc_diff, logits_diff = model(img_batch, cids_gt_batch=cids_gt_reduced,
                                              cids_gt_sample_num_batch=cids_gt_sample_num_batch)

            n_query = cls_logits_pred_batch.shape[1]
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
                cids_gt = torch.tensor(cids_gt_batch[b], dtype=torch.long, device=device)
                boxes_gt_xyxy = boxes_gt_xyxy_batch[b].to(device) / max_img_size

                t = time.time()
                rows, cols = assign_query(boxes_gt_xyxy, boxes_pred_xyxy_batch[b], cids_gt, cls_logits_pred_batch[b])
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
                cls_neg_loss_batch += cls_neg_loss * n_neg
                tn += torch.sum(cls_pred_batch[b, cls_query_neg_indices] == cids_gt_neg)

                box_loss = distance_box_iou_loss(boxes_pred_xyxy_batch[b, rows], boxes_gt_xyxy, reduction='mean')
                box_loss_batch += box_loss * n_pos

            # accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt_batch, query_pos_mask)
            recall = tp / n_pos_batch
            accu = (tp + tn) / (n_pos_batch + n_neg_batch)

            cls_pos_loss_batch = cls_pos_loss_batch / n_pos_batch
            cls_neg_loss_batch = cls_neg_loss_batch / n_neg_batch
            box_loss_batch = box_loss_batch / n_pos_batch
            loss = cls_pos_loss_batch + cls_neg_loss_batch + box_loss_batch * 10 + src_cls_loss
            optimizer.zero_grad()
            t = time.time()
            loss.backward()
            t_bp = time.time() - t
            # nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()

            print(f'smp {num_sample}|epoch {i + 1}/{epoch}|batch {j}'
                  f'|pl {cls_pos_loss_batch.detach().item() * 1000:.3f}'
                  f'|nl {cls_neg_loss_batch.detach().item() * 1000:.3f}'
                  f'|sl {src_cls_loss.detach().item() * 1000:.3f}'
                  f'|bl {box_loss_batch.detach().item():.3f}'
                  f'|src {src_cls_recall:.3f}'
                  f'|sac {src_cls_accu:.3f}'
                  f'|qac {accu:.3f}|qrc {recall:.3f}: {tp}/{n_pos_batch}'
                  # f'|dif {enc_diff:.3f} {logits_diff:.3f}'
                  f'|tmch {t_match:.3f}|tbp {t_bp:.3f}'
                  # f'|match {matched_pos_gt}|'
                  f'|img {" ".join(img_id_batch)}'
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
        #     state[k] = saved_state[k]
        # model.load_state_dict(state)
    for i in range(500):
        n_smp = latest_version + 1 + i
        ts = time.time()
        train(1, batch_size=train_base_bsz, population=1000, num_sample=n_smp, weight_recover=0, gamma=4)
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