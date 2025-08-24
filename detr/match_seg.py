import torch
from torch import nn
import scipy
import torchvision.ops


def assign_query(seg_gt, seg_pred, cids_gt, cls_logits, gt_pos_mask):
    b, q, c = seg_gt.shape
    n_pos = gt_pos_mask.sum(dim=-1).view(b)

    rows = []
    cols = []
    for m in range(b):
        if n_pos[m] == 0:
            rows.append(list(range(q)))
            cols.append(list(range(q)))
            continue
        with torch.no_grad():
            cids_gt_ = cids_gt[m, :n_pos[m]].view(1, n_pos[m]).expand(q, n_pos[m])
            cids_gt_ = cids_gt_.contiguous().view(-1)
            cls_logits_ = cls_logits[m].view(q, 1, -1).expand(q, n_pos[m], -1)
            cls_logits_ = cls_logits_.contiguous().view(q * n_pos[m], -1)
            cls_loss = nn.functional.cross_entropy(cls_logits_, cids_gt_, reduction='none')
            cls_loss = cls_loss.view(q, n_pos[m])

            seg_gt_ = seg_gt[m, :n_pos[m]].view(1, n_pos[m], c).expand(q, n_pos[m], c)
            seg_pred_ = seg_pred[m].view(q, 1, c).expand(q, n_pos[m], c)
            inter = torch.sum(seg_gt_ * seg_pred_, dim=-1)
            union = torch.sum(seg_gt_ + seg_pred_, dim=-1) - inter
            iou_loss = 1 - (inter / (union + 1e-5))
        
        cls_loss_mean = cls_loss.mean()
        iou_loss_mean = iou_loss.mean()
        total_loss = iou_loss / (iou_loss_mean + 1e-5) + cls_loss / (cls_loss_mean + 1e-5)


        row_, col_ = scipy.optimize.linear_sum_assignment(total_loss.cpu().numpy())
        col_ = col_.tolist()
        row_ = row_.tolist()
        unmatched_col = set(range(q)) - set(col_)
        row = list(range(q))
        col = []
        for j in range(q):
            if len(row_) == 0 or j != row_[0]:
                col.append(unmatched_col.pop())
            else:
                col.append(col_.pop(0))
                row_.pop(0)

        rows.append(row)
        cols.append(col)

    return rows, cols
