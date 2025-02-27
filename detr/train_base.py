import os
import torch
from torch import optim
import sys
sys.path.append('..')
from detr import anno, detr_dataset, detr_model_base, match, eval
from common.config import train_annotation_file, train_img_od_dict_file, img_size, \
    train_base_bsz, model_save_dir, model_save_stride, device_type
from detr.config import loss_weights, n_query
import focalloss
import time
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device(device_type)

model = detr_model_base.DETR(d_cont=384, d_enc_coord_emb=64, d_head=64, n_enc_layer=20, n_dec_layer=8, n_query=n_query)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='detr')


def train(epoch, batch_size, population, num_sample, weight_recover=0.5, gamma=4):
    ds = detr_dataset.OdDataset(dicts, n_query, train=True, sample_num=population, random_shift=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for i in range(epoch):
        for j, (img, boxes_gt_xyxy, cids_gt, _, img_id) in enumerate(dl):
            img = img.to(device)
            cids_gt = cids_gt.to(torch.long)
            cids_gt = cids_gt.to(device)
            boxes_gt_xyxy = boxes_gt_xyxy.to(device) / max(img_size)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            B = img.shape[0]

            boxes_pred_xyxy, cls_logits_pred, enc_diff, logits_diff = model(img)
            gt_pos_mask = (cids_gt > 0)
            gt_pos_mask = gt_pos_mask.view(B, 1, n_query) * 1

            t = time.time()
            rows, cols = match.assign_query(boxes_gt_xyxy, boxes_pred_xyxy, cids_gt, cls_logits_pred, gt_pos_mask, anchors=None)
            t_match = time.time() - t
            cols = torch.tensor(np.stack(cols), device=device)

            cls_pos_num = torch.sum(gt_pos_mask, dim=-1)

            gt_matched_indices_batch = torch.arange(B, device=device).view(B, 1). \
                expand(B, n_query).contiguous().view(B * n_query)

            gt_matched_indices_query = cols.view(B * n_query)

            query_pos_mask = (cols < cls_pos_num).view(B * n_query) * 1
            n_pos = query_pos_mask.sum()

            # cls loss
            cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
            cls_logits_pred = cls_logits_pred.view(B * n_query, -1)

            cls_pred = cls_logits_pred.argmax(dim=-1)

            accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt, query_pos_mask)

            ce_alpha = torch.tensor(loss_weights, device=device) ** weight_recover
            cls_loss = focalloss.focal_loss(cls_logits_pred, cids_gt, ce_alpha, gamma=gamma).mean()

            # box loss
            boxes_gt_xyxy = boxes_gt_xyxy[(gt_matched_indices_batch, gt_matched_indices_query)]
            box_loss = distance_box_iou_loss(boxes_pred_xyxy.view(B * n_query, -1), boxes_gt_xyxy.view(B * n_query, -1))
            box_loss = box_loss * query_pos_mask
            box_loss = box_loss.sum() / (n_pos + 1e-5)

            loss = cls_loss * 5 + box_loss
            optimizer.zero_grad()
            t = time.time()
            loss.backward()
            t_bp = time.time() - t
            # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
            optimizer.step()

            print(f'smp {num_sample}|epoch {i + 1}/{epoch}|batch {j}|'
                  f'cl {cls_loss.detach().item() * 1000:.3f}|'
                  f'bl {box_loss.detach().item():.3f}|'
                  f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|'
                  f'dif {enc_diff.item():.3f} {logits_diff.item():.3f}|'
                  f'tmch {t_match:.3f}|tbp {t_bp:.3f}|'
                  # f'match {matched_pos_gt}|'
                  f'img {" ".join(img_id)}')


if __name__ == '__main__':
    model_files = os.listdir(model_save_dir)
    model_files = [f for f in model_files if f.endswith('.pt') and f.startswith('od_base')]
    if len(model_files) == 0:
        model_path_old = None
        latest_version = 0
    else:
        versions = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_version = max(versions)
        model_path_old = f'{model_save_dir}/od_base_{latest_version}.pt'
        saved_state = torch.load(model_path_old, map_location=device)
        model.load_state_dict(saved_state)
        # state = tsfm.state_dict()
        # for k in state:
        #     state[k] = saved_state[k]
        # tsfm.load_state_dict(state)
    for i in range(500):
        n_smp = latest_version + 1 + i
        ts = time.time()
        train(1, batch_size=train_base_bsz, population=1000, num_sample=n_smp, weight_recover=0, gamma=4)
        te = time.time()
        print(f'----------------------used {te-ts:.3f} secs---------------------------')
        # train(400, batch_size=2, population=2, num_sample=i, weight_recover=0, gamma=4)
        # train(2, batch_size=2, population=2, num_sample=i, weight_recover=.5, gamma=2)
        if n_smp % model_save_stride == 0:
            model_path_new = f'{model_save_dir}/od_base_{n_smp}.pt'
            torch.save(model.state_dict(), model_path_new)
            if model_path_old is not None:
                os.remove(model_path_old)
            model_path_old = model_path_new
        # break
