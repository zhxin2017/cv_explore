import torch
from torch import nn
import scipy
import torchvision.ops


def assign_query(seg_gt_pos_mask, seg_pred_pos_mask, cids_gt, cls_pred, gt_pos_mask):
    B, N, H, W, C = seg_gt_pos_mask.shape
    n_pos = gt_pos_mask.sum(dim=-1).view(B)

    rows = []
    cols = []
    for i in range(B):
        with torch.no_grad():
            seg_gt_pos_mask_ = seg_gt_pos_mask[i].view(N, 1, H, W, C).expand(N, n_pos[i], H, W, C)
            seg_pred_pos_mask_ = seg_pred_pos_mask[i].view(1, n_pos[i], H, W, C).expand(N, n_pos[i], H, W, C)
            inter = torch.sum(seg_gt_pos_mask_ * seg_pred_pos_mask_, dim=(2, 3, 4))
            union = torch.sum(seg_gt_pos_mask_, dim=(2, 3, 4)) + torch.sum(seg_pred_pos_mask_, dim=(2, 3, 4)) - inter
            iou = inter / (union + 1e-5)
            iou_loss = 1 - iou

            cls_pred_ = cls_pred[i].view(N, 1, -1).repeat(1, n_pos[i], 1).contiguous().view(N * n_pos[i], -1)
            cids_gt_ = cids_gt[i, :n_pos[i]].view(1, n_pos[i]).expand(N, n_pos[i]).contiguous().view(-1)
            cls_loss = nn.CrossEntropyLoss(reduction='none')(cls_pred_, cids_gt_).view(N, n_pos[i])

            total_loss = iou_loss + cls_loss
        # total_loss[total_loss == torch.nan] = 1e8
        row_, col_ = scipy.optimize.linear_sum_assignment(total_loss.detach().cpu().numpy())
        col_ = col_.tolist()
        row_ = row_.tolist()
        unmatched_col = set(range(N)) - set(col_)
        row = list(range(N))
        col = []
        for j in range(N):
            if len(row_) == 0 or j != row_[0]:
                col.append(unmatched_col.pop())
            else:
                col.append(col_.pop(0))
                row_.pop(0)

        rows.append(row)
        cols.append(col)

    return rows, cols
