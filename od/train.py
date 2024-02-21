import torch
from torch import nn, optim
from od import anno, detr_dataset, detr_model, match, box
from common.config import train_annotation_file, train_img_od_dict_file, img_size
from od.config import category_num, n_query
import focalloss
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("mps")
# device = torch.device("cpu")


def eval_pred(cls_logits_pred, cids_gt, query_pos_mask):
    cls_pred = torch.argmax(cls_logits_pred, dim=-1)
    tp_mask = (cls_pred == cids_gt) * query_pos_mask
    query_neg_mask = 1 - query_pos_mask
    tp = tp_mask.sum()
    n_pos = query_pos_mask.sum()
    n_neg = query_neg_mask.sum()
    n_cls = n_pos + n_neg
    tn = ((cls_pred == cids_gt) * query_neg_mask).sum()
    fn = n_pos - tp
    accu = (tp + tn) / (n_cls + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    return accu, recall, n_pos, tp


cls_loss_fun = nn.CrossEntropyLoss(reduction='none')

model = detr_model.DETR(d_cont=256, device=device)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# alpha = torch.tensor(loss_weights, device=device)**0.8

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')


def train(epoch, batch_size, population, num_sample, weight_recover=0.8):
    ds = detr_dataset.OdDataset(dicts, train=True, sample_num=population, random_shift=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for i in range(epoch):
        for j, (img, boxes_gt_xyxy, cids_gt, img_id) in enumerate(dl):
            img = img.to(device)
            cids_gt = cids_gt.to(torch.long)
            cids_gt = cids_gt.to(device)
            boxes_gt_xyxy = boxes_gt_xyxy.to(device) / max(img_size)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            B = img.shape[0]

            boxes_pred_cxcywh, cls_logits_pred = model(img)
            boxes_pred_xyxy = box.cxcywh_to_xyxy(boxes_pred_cxcywh)

            gt_pos_mask = (cids_gt > 0).view(B, 1, n_query) * 1

            _, cols = match.assign_query(boxes_gt_xyxy, boxes_pred_xyxy, cids_gt, cls_logits_pred, gt_pos_mask)
            cols = torch.tensor(np.stack(cols), device=device)

            cls_pos_num = torch.sum(gt_pos_mask, dim=-1)

            gt_matched_indices_batch = torch.arange(B, device=device).view(B, 1). \
                expand(B, n_query).contiguous().view(B * n_query)

            gt_matched_indices_query = cols.view(B * n_query)

            cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
            cids_gt = cids_gt.view(B * n_query)

            cls_logits_pred = cls_logits_pred.view(B * n_query, -1)

            cids_gt_onehot = torch.zeros(B * n_query, category_num, device=device).scatter_(1, cids_gt.unsqueeze(1), 1)
            cids_num = cids_gt_onehot.sum(dim=0)
            alpha = focalloss.cal_weights(cids_num, recover=weight_recover)
            cls_loss = focalloss.focal_loss(cls_logits_pred, cids_gt, alpha).mean() * 100
            # cls_loss = cls_loss_fun(cls_logits_pred, cids_gt)

            boxes_gt_xyxy = boxes_gt_xyxy[(gt_matched_indices_batch, gt_matched_indices_query)]
            box_loss = distance_box_iou_loss(boxes_pred_xyxy.view(B * n_query, -1), boxes_gt_xyxy.view(B * n_query, -1))

            query_pos_mask = (cols < cls_pos_num).view(B * n_query) * 1
            box_loss = box_loss * query_pos_mask

            accu, recall, n_pos, n_tp = eval_pred(cls_logits_pred, cids_gt, query_pos_mask)
            box_loss = box_loss.sum() / (n_pos + 1e-5)

            loss = cls_loss + box_loss * 10

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()

            print(f'smp {num_sample}, epoch {i + 1}/{epoch}, batch {j}|'
                  f'cl {cls_loss.detach().item():.3f}|b'
                  f'l {box_loss.detach().item():.3f}|'
                  f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|'
                  f'img {" ".join(img_id)}')


if __name__ == '__main__':
    model_file = f'/Users/zx/Documents/ml/restart/resources/od_detr.pt'
    # model.load_state_dict(torch.load(model_file))
    for i in range(500):
        # train(1, batch_size=2, population=1000, num_sample=i, weight_recover=.5)
        train(1000, batch_size=2, population=2, num_sample=i, weight_recover=1)
        # torch.save(model.state_dict(), model_file)
        break
