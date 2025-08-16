import torch
from torch import nn
import scipy
import torchvision.ops


def assign_query(cids_gt, seg_logits, gt_pos_mask):
    B, N, H, W, C = seg_logits.shape
    n_pos = gt_pos_mask.sum(dim=-1).view(B)

    rows = []
    cols = []
    for i in range(B):
        with torch.no_grad():
            cids_gt_ = cids_gt[i, :n_pos[i]].view(1, n_pos[i], H, W).expand(N, n_pos[i], H, W)
            cids_gt_ = cids_gt_.contiguous().view(-1)
            cls_logits_ = seg_logits[i].view(N, 1, H, W, C).expand(N, n_pos[i], H, W, C)
            cls_logits_ = cls_logits_.contiguous().view(-1, C)
            cls_loss = nn.functional.cross_entropy(cls_logits_, cids_gt_, reduction='none')
            cls_loss = torch.mean(cls_loss.view(N, n_pos[i], H, W), dim=(2, 3))

        # total_loss[total_loss == torch.nan] = 1e8
        row_, col_ = scipy.optimize.linear_sum_assignment(cls_loss.detach().cpu().numpy())
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
