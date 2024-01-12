import torch
from torch import nn, optim
from od import anno, detr_dataset, detr_model, match, box
from od.config_file import val_annotation_file, val_img_od_dict_file
import focalloss
import numpy as np
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device("mps")


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


cls_loss_fun = nn.CrossEntropyLoss(reduction='mean')

model = detr_model.DETR(d_cont=256)
model.to(device)

sub_epoch = 10000
cnt = 0
optimizer = optim.Adam(model.parameters(), lr=1e-5)
n_query = 300

while True:
    cnt += 1
    img_sz = 512

    dicts = anno.build_img_dict(val_annotation_file, val_img_od_dict_file, task='od')
    ds = detr_dataset.OdDataset(dicts, train=False, sample_num=10)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    bcnt = 0

    for img, boxes_gt_xyxy, cids_gt, img_id in dl:
        bcnt += 1
        print(f'=== sub epoch #{cnt}, iter #{bcnt} out of {sub_epoch}==================================')
        img = img.to(device)
        cids_gt = cids_gt.to(device)

        boxes_gt_xyxy = boxes_gt_xyxy.to(device) / img_sz

        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        B = img.shape[0]

        boxes_pred_cxcywh, cls_logits_pred = model(img)

        diff_sum = (boxes_pred_cxcywh[1] - boxes_pred_cxcywh[0]).abs().sum() if B > 1 else 0

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

        # cls_loss = cls_loss_fun(cls_logits_pred, cids_gt)
        cls_loss = focalloss.focal_loss(cls_logits_pred, cids_gt).mean()

        boxes_gt_xyxy = boxes_gt_xyxy[(gt_matched_indices_batch, gt_matched_indices_query)]

        box_loss = distance_box_iou_loss(boxes_pred_xyxy.view(B * n_query, -1), boxes_gt_xyxy.view(B * n_query, -1))

        query_pos_mask = (cols < cls_pos_num).view(B * n_query) * 1
        box_loss = box_loss * query_pos_mask

        accu, recall, n_pos, n_tp = eval_pred(cls_logits_pred, cids_gt, query_pos_mask)

        box_loss = box_loss.sum() / (n_pos + 1e-5)

        cls_loss_ = cls_loss.detach().item()
        box_loss_ = box_loss.detach().item()

        loss = box_loss_ / cls_loss_ * cls_loss + box_loss

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), 0.05)
        optimizer.step()

        print(f'|#{cnt}-{bcnt}|cl {cls_loss_:.3f}|bl {box_loss_:.3f}|'
              f'ac {accu:.3f}/rc:{recall:.3f}, {n_tp} out of {n_pos}|dif {diff_sum:.3f}')

        if bcnt == sub_epoch:
            break
    # if cnt % 4 == 0:
    #     torch.save(model.state_dict(), f'resources/od_detr.pt')
