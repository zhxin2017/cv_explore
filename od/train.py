import os
import torch.nn.functional as F
import torch
from torch import nn, optim
from od import anno, detr_dataset, detr_model, match, eval
from common.config import train_annotation_file, train_img_od_dict_file, img_size
from od.config import loss_weights
import focalloss
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("mps")
# device = torch.device("cpu")

cls_loss_fun = nn.CrossEntropyLoss(reduction='none')
model = detr_model.DETR(d_enc=384, d_extremity=128, d_coord_emb=128,
                        n_enc_head=6, n_enc_layer=18, exam_diff=True)
model.to(device)
n_query = model.decoder.n_anchor

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# alpha = torch.tensor(loss_weights, device=device)**0.8

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')


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

            boxes_pred_xyxy, cls_logits_pred1, cls_logits_pred2, enc_diff, logits_diff, center_features, extremity_features, extremity_emb, _, _ = model(img)

            gt_pos_mask = (cids_gt > 0).view(B, 1, n_query) * 1

            _, cols = match.assign_query(boxes_gt_xyxy, boxes_pred_xyxy, cids_gt, cls_logits_pred2, gt_pos_mask)
            cols = torch.tensor(np.stack(cols), device=device)

            cls_pos_num = torch.sum(gt_pos_mask, dim=-1)

            matched_pos_gt = torch.where(cols < cls_pos_num)
            matched_pos_gt = [(matched_pos_gt[0][i].item(), matched_pos_gt[1][i].item()) for i in
                              range(len(matched_pos_gt[0]))]

            gt_matched_indices_batch = torch.arange(B, device=device).view(B, 1). \
                expand(B, n_query).contiguous().view(B * n_query)

            gt_matched_indices_query = cols.view(B * n_query)

            cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
            cids_gt = cids_gt.view(B * n_query)

            cls_logits_pred1 = cls_logits_pred1.view(B * n_query, -1)
            cls_logits_pred2 = cls_logits_pred2.view(B * n_query, -1)

            # cids_gt_onehot = torch.zeros(B * n_query, category_num, device=device).scatter_(1, cids_gt.unsqueeze(1), 1)
            # cids_num = cids_gt_onehot.sum(dim=0)
            # alpha = focalloss.cal_weights(cids_num, recover=weight_recover)
            alpha = torch.tensor(loss_weights, device=device) ** weight_recover
            cls_loss1 = focalloss.focal_loss(cls_logits_pred1, cids_gt, alpha, gamma=gamma).mean()
            cls_loss2 = focalloss.focal_loss(cls_logits_pred2, cids_gt, alpha, gamma=gamma).mean()
            # cls_loss = cls_loss_fun(cls_logits_pred, cids_gt)

            # cls_loss = -torch.log_softmax(cls_logits_pred, dim=-1) * cids_gt_onehot
            # cls_loss = cls_loss * (alpha.view(1, category_num))**weight_recover
            # cls_loss = cls_loss.sum(dim=-1).mean() * 10

            boxes_gt_xyxy = boxes_gt_xyxy[(gt_matched_indices_batch, gt_matched_indices_query)]
            box_loss = distance_box_iou_loss(boxes_pred_xyxy.view(B * n_query, -1), boxes_gt_xyxy.view(B * n_query, -1))

            query_pos_mask = (cols < cls_pos_num).view(B * n_query) * 1
            box_loss = box_loss * query_pos_mask

            accu, recall, f1, n_pos, n_tp = eval.eval_pred(cls_logits_pred2, cids_gt, query_pos_mask)
            box_loss = box_loss.sum() / (n_pos + 1e-5)

            extremity_emb_ = extremity_emb[:, 0]
            extremity_loss_center = F.mse_loss(extremity_emb_, center_features[..., model.decoder.d_cont:], reduction='none')
            extremity_loss_center = 2 ** -extremity_loss_center
            extremity_loss_center = extremity_loss_center.reshape(B * n_query, -1) * (query_pos_mask.unsqueeze(dim=1))
            extremity_loss_center = extremity_loss_center.sum() / n_pos

            extremity_emb = extremity_emb.transpose(1, 2).reshape(B * n_query, -1)
            extremity_feat = extremity_features[..., model.decoder.d_cont:].reshape(B * n_query, -1)
            extremity_loss = F.mse_loss(extremity_emb, extremity_feat, reduction='none')
            extremity_loss = extremity_loss * (query_pos_mask.unsqueeze(dim=1))
            extremity_loss = extremity_loss.sum() / n_pos

            loss = cls_loss1 * 10 + cls_loss2 * 10 + box_loss + extremity_loss * .005 + extremity_loss_center * .005
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()

            print(f'smp {num_sample}|epoch {i + 1}/{epoch}|batch {j}|'
                  f'cl1 {cls_loss1.detach().item() * 1000:.3f}|'
                  f'cl2 {cls_loss2.detach().item() * 1000:.3f}|'
                  f'bl {box_loss.detach().item():.3f}|'
                  f'el {extremity_loss.detach().item():.3f} {extremity_loss_center.detach().item():.3f}|'
                  f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|f1 {f1:.3f}|'
                  f'dif {enc_diff.item():.3f} {logits_diff.item():.3f}|'
                  # f'match {matched_pos_gt}|'
                  f'img {" ".join(img_id)}')


if __name__ == '__main__':
    model_dir = '/Users/zx/Documents/ml/restart/resources'
    model_files = os.listdir(model_dir)
    model_files = [f for f in model_files if f.endswith('.pt')]
    if len(model_files) == 0:
        model_path_old = None
        latest_version = -1
    else:
        versions = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_version = max(versions)
        model_path_old = f'{model_dir}/od_detr_{latest_version}.pt'
        saved_state = torch.load(model_path_old)
        state = model.state_dict()
        for k, v in saved_state.items():
            if k in state:
                state[k] = v
        model.load_state_dict(state)
        # model.load_state_dict(saved_state)
    for i in range(500):
        batch = latest_version + 1 + i
        # train(1, batch_size=2, population=1000, num_sample=batch, weight_recover=.5, gamma=8)
        train(500, batch_size=2, population=2, num_sample=i, weight_recover=.6, gamma=8)
        model_path_new = f'{model_dir}/od_detr_{batch}.pt'
        torch.save(model.state_dict(), model_path_new)
        if model_path_old is not None:
            os.remove(model_path_old)
        model_path_old = model_path_new
        # break
