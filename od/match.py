import torch
from torch import nn
import scipy
import torchvision.ops
from od.config import category_num


def assign_query(boxes_gt, boxes_pred, cids_gt, cls_pred, gt_pos_mask):
    B, N, C = boxes_pred.shape
    boxes_pred = boxes_pred.view(B, N, 1, C).expand(B, N, N, C)
    boxes_gt = boxes_gt.view(B, 1, N, C).expand(B, N, N, C)
    iouloss = torchvision.ops.distance_box_iou_loss(boxes_pred, boxes_gt) * gt_pos_mask

    cls_pred = cls_pred.view(B, N, 1, category_num).expand(B, N, N, category_num).contiguous().view(-1, category_num)
    cids_gt = cids_gt.view(B, 1, N).expand(B, N, N).contiguous().view(-1)
    cls_loss = nn.CrossEntropyLoss(reduction='none')(cls_pred, cids_gt).view(B, N, N) * gt_pos_mask

    total_loss = iouloss * 10 + cls_loss * 0
    total_loss[total_loss == torch.nan] = 1e8

    rows = []
    cols = []
    for i in range(B):
        row, col = scipy.optimize.linear_sum_assignment(total_loss[i].cpu().detach().numpy())
        rows.append(row)
        cols.append(col)

    return rows, cols
