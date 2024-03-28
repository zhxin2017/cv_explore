import torch
from torch import nn
import scipy
import torchvision.ops
from od.config import n_cls


def assign_query(boxes_gt, boxes_pred, cids_gt, cls_pred, gt_pos_mask, anchors=None):
    B, N, C = boxes_gt.shape
    if len(boxes_pred.shape) == 2:
        boxes_pred = boxes_pred.unsqueeze(0).repeat(B, 1, 1)
    n_pos = gt_pos_mask.sum(dim=-1).view(B)


    rows = []
    cols = []
    for i in range(B):

        boxes_pred_ = boxes_pred[i].view(N, 1, C).expand(N, n_pos[i], C)
        boxes_gt_ = boxes_gt[i, :n_pos[i]].view(1, n_pos[i], C).expand(N, n_pos[i], C)
        iouloss = torchvision.ops.distance_box_iou_loss(boxes_pred_, boxes_gt_)

        cls_pred_ = cls_pred[i].view(N, 1, -1).repeat(1, n_pos[i], 1).contiguous().view(N * n_pos[i], -1)
        cids_gt_ = cids_gt[i, :n_pos[i]].view(1, n_pos[i]).expand(N, n_pos[i]).contiguous().view(-1)
        cls_loss = nn.CrossEntropyLoss(reduction='none')(cls_pred_, cids_gt_).view(N, n_pos[i])

        if anchors is not None:
            anchors_ = anchors.view(N, 1, C).expand(N, n_pos[i], C)
            iouloss_anchor = torchvision.ops.distance_box_iou_loss(anchors_, boxes_gt_)
        else:
            iouloss_anchor = 0

        total_loss = iouloss * .5 + iouloss_anchor * .5 + cls_loss * .05
        # total_loss[total_loss == torch.nan] = 1e8
        row_, col_ = scipy.optimize.linear_sum_assignment(total_loss.detach().cpu().numpy())
        row = list(range(N))
        col = list(range(N))
        col[:n_pos[i]] = row_
        for j in range(len(row_)):
            col[row_[j]] = col_[j]

        rows.append(row)
        cols.append(col)

    return rows, cols
