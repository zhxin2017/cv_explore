import random
import math
import scipy
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from detr import box
from detr.config import n_cls, n_pos_query
from common.config import max_grid_h, max_grid_w, patch_size, max_img_len

ce = nn.CrossEntropyLoss()
grid_size = patch_size / max_img_len
grid_area = grid_size**2

class DetrEncoder(nn.Module):
    def __init__(self, n_enc_layer, d_cont, d_head, d_pos, pretrain=False):
        super().__init__()

        self.cnn1 = nn.Conv2d(3, 1024, (4, 4), stride=(4, 4))
        self.cnn2 = nn.Conv2d(1024, d_cont, (4, 4), stride=(4, 4))
        self.cnn_ln = nn.LayerNorm(d_cont)
        self.d_pos = d_pos
        self.pos_ln = nn.LayerNorm(d_pos // 2)

        dq = d_cont + d_pos
        n_head = dq // d_head

        self.pos_y_emb_m = nn.Embedding(max_grid_h, d_pos // 2)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, d_pos // 2)

        self.n_enc_layer = n_enc_layer
        self.attn_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.attn_layers.append(tsfm.AttnLayer(dq, dq, d_cont, n_head))

        self.pretrain = pretrain
        if pretrain:
            self.next_token_emb_m = pe.Embedding1D(1, d_cont)

    def forward(self, x, x_shift=0, y_shift=0, mask=None):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        b, h, w, c = x.shape
        num_grid = h * w

            # x_shift = random.randint(0, max_grid_x - w)
            # y_shift = random.randint(0, max_grid_y - h)

        y_indices = torch.arange(h, device=x.device) + y_shift
        x_indices = torch.arange(w, device=x.device) + x_shift
        pos_y_emb = self.pos_ln(self.pos_y_emb_m(y_indices).view(1, h, 1, self.d_pos // 2).repeat(b, 1, w, 1))
        pos_x_emb = self.pos_ln(self.pos_x_emb_m(x_indices).view(1, 1, w, self.d_pos // 2).repeat(b, h, 1, 1))
        pos_emb = torch.concat([pos_y_emb, pos_x_emb], dim=-1).view(b, num_grid, -1)
        x = x.view(b, num_grid, -1)

        if self.pretrain:
            next_token_emb = self.next_token_emb_m(x).repeat(1, num_grid, 1)
            x = torch.concat([x, next_token_emb], dim=1)
            pos_emb = torch.concat([pos_emb, pos_emb], dim=1)
            mask_11 = torch.tril(torch.ones([num_grid, num_grid], device=x.device))
            mask_12 = torch.diag(torch.ones([num_grid - 1], device=x.device), diagonal=1)
            mask_21 = torch.tril(torch.ones([num_grid, num_grid], device=x.device), diagonal=-1)
            mask_22 = torch.diag(torch.ones([num_grid], device=x.device))
            mask_1 = torch.concat([mask_11, mask_12], dim=-1)
            mask_2 = torch.concat([mask_21, mask_22], dim=-1)
            mask = torch.concat([mask_1, mask_2], dim=0)  # 2 * num_grid, 2 * num_grid
        else:
            mask = mask

        for i in range(self.n_enc_layer):
            q = torch.concat([x, pos_emb], dim=-1)
            x = self.attn_layers[i](q, q, x, q, mask)
        return x, pos_emb


class DetrDecoder(nn.Module):
    def __init__(self, n_dec_layer, d_cont, d_head, cont_ln, cls_emb_m, d_src_pos, d_tgt_pos):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        self.n_pos_query = n_pos_query
        self.pos_query_emb_m = nn.Embedding(n_pos_query, d_tgt_pos)
        self.cls_emb_m = cls_emb_m
        self.cont_ln = cont_ln

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()

        dq = d_cont + d_tgt_pos
        n_head = dq // d_head

        dk = d_cont + d_src_pos
        for i in range(n_dec_layer):

            ca_layer = tsfm.AttnLayer(dq, dk, dk, n_head)
            self.ca_layers.append(ca_layer)

            if i < n_dec_layer:
                sa_layer = tsfm.AttnLayer(dq, dq, dq, n_head)
                self.sa_layers.append(sa_layer)

        self.cls_reg = nn.Linear(dq, n_cls, bias=False)
        self.box_reg = base.MLP(dq, dq * 2, 4, 2)
        self.out_ln = nn.LayerNorm(dq)

    def forward(self, src, src_pos_emb, cls_query, mask=None):
        B = src.shape[0]
        pos_query = self.pos_query_emb_m(torch.arange(self.n_pos_query, device=src.device))
        n_cls_query = cls_query.shape[1]
        pos_query = pos_query.view(1, 1, self.n_pos_query, -1).repeat(B, n_cls_query, 1, 1)
        cls_query = cls_query.view(B, n_cls_query, 1, -1).repeat(1, 1, self.n_pos_query, 1)
        q = torch.concat([cls_query, pos_query], dim=-1).reshape(B, n_cls_query * self.n_pos_query, -1)

        src_with_pos = torch.concat([self.cont_ln(src), src_pos_emb], dim=-1)

        for i in range(self.n_dec_layer):

            q = self.ca_layers[i](q, src_with_pos, src_with_pos, q, mask)

            if i < self.n_dec_layer - 1:
                q = self.sa_layers[i](q, q, q, q)

        boxes = F.sigmoid(self.box_reg(self.out_ln(q)))
        cls_logits = self.cls_reg(self.out_ln(q))
        return boxes, cls_logits


class DETR(nn.Module):

    def __init__(self, d_cont, d_head, d_src_pos, d_tgt_pos, n_enc_layer, n_dec_layer, exam_diff=True, train=True):
        super().__init__()
        self.encoder = DetrEncoder(n_enc_layer, d_cont, d_head, d_src_pos)
        self.cls_emb_m = nn.Embedding(n_cls, d_cont)
        self.cont_ln = nn.LayerNorm(d_cont)
        self.decoder = DetrDecoder(n_dec_layer, d_cont, d_head, self.cont_ln, self.cls_emb_m, d_src_pos, d_tgt_pos)
        self.exam_diff = exam_diff
        self.train = train
        self.cid_set = set(range(n_cls))

    def forward(self, x, x_shift=0, y_shift=0, masks=None, cids_gt_batch=None, boxes_gt_batch=None):
        h, w = x.shape[2] // patch_size, x.shape[3] // patch_size
        l_src = h * w
        if masks is not None:
            enc_masks = torch.permute(masks, [0, 2, 1]) @ masks
            diag = torch.diag(torch.ones(l_src, device=x.device)).view(1, l_src, l_src)
            enc_masks = enc_masks * (1 - diag) + diag
        else:
            enc_masks = None
        src, src_pos_emb = self.encoder(x, x_shift, y_shift, mask=enc_masks)
        bsz = src.shape[0]
        src_cls_logits = self.cont_ln(src) @ self.cls_emb_m.weight.T
        src_cls_prob = F.softmax(src_cls_logits, dim=-1)
        src_cls_prob_max, src_cls = torch.max(src_cls_prob, dim=-1)  # both bsz * seq_len
        src_cids = [list(set(p) - {0}) for p in src_cls.tolist()]

        src_cls_pos_loss_batch = 0
        src_cls_neg_loss_batch = 0

        n_grid = src.shape[1]

        if self.train:
            assert cids_gt_batch is not None
            tp_batch = sum(len(set(cids_gt_batch[b]).intersection(set(src_cids[b]))) for b in range(bsz))
            tn_batch = sum(len(self.cid_set - {0} - set(cids_gt_batch[b]) - set(src_cids[b])) for b in range(bsz))
            n_cid_gt = sum([len(set(cids)) for cids in cids_gt_batch])
            src_cls_recall = tp_batch / n_cid_gt
            src_cls_accu = (tp_batch + tn_batch) / (n_cls - 1) / bsz

            n_tgt_grid_pos_batch = 0
            n_tgt_grid_neg_batch = 0

            for b in range(bsz):
                boxes_gt = boxes_gt_batch[b]
                boxes_gt = boxes_gt.to(x.device)
                n_obj = len(cids_gt_batch[b])
                grid_x1 = (torch.arange(n_grid, device=x.device).view(n_grid, 1) % w + x_shift) * grid_size
                grid_y1 = (torch.arange(n_grid, device=x.device).view(n_grid, 1) // w + y_shift) * grid_size
                grid_center_x = grid_x1 + .5 * grid_size
                grid_center_y = grid_y1 + .5 * grid_size
                obj_x1, obj_y1, obj_x2, obj_y2 = boxes_gt.view(1, n_obj, 4).unbind(2)
                obj_center_x, obj_center_y = (obj_x1 + obj_x2) / 2, (obj_y1 + obj_y2) / 2

                distance_matrix = ((grid_center_x - obj_center_x)**2 + (grid_center_y - obj_center_y)**2)**.5

                # match positive samples
                cids_tgt = []
                tgt_indices = []
                for o in range(n_obj):
                    cid = cids_gt_batch[b][o]
                    x1, y1, x2, y2 = boxes_gt[o]
                    obj_area = (x2 - x1) * (y2 - y1)
                    n_grid_obj = math.ceil(obj_area / grid_area / 8)
                    cids_tgt.extend([cid] * n_grid_obj)
                    tgt_indices.extend([o] * n_grid_obj)
                cids_tgt = torch.tensor(cids_tgt, dtype=torch.long, device=x.device)
                cost_matrix = 1 - src_cls_prob[b, :, cids_tgt] + distance_matrix[:, tgt_indices]
                rows, cols = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                n_tgt_grid_pos = len(cids_tgt)
                src_cls_pos_loss = ce(src_cls_logits[b, rows], cids_tgt[cols]) * n_tgt_grid_pos
                src_cls_pos_loss_batch += src_cls_pos_loss
                n_tgt_grid_pos_batch += n_tgt_grid_pos

                # match negative samples
                unmatched_grid_indices = torch.tensor(list(set(range(n_grid)) - set(rows)), device=x.device)
                unmatched_cls = src_cls[b, unmatched_grid_indices]
                unmatched_fp_mask = unmatched_cls > 0
                if unmatched_fp_mask.sum() == 0:
                    continue

                unmatched_grid_indices = unmatched_grid_indices[unmatched_fp_mask]
                n_unmatched = len(unmatched_grid_indices)

                grid_box = torch.concat([grid_x1[unmatched_grid_indices],
                                         grid_y1[unmatched_grid_indices],
                                         grid_x1[unmatched_grid_indices] + grid_size,
                                         grid_y1[unmatched_grid_indices] + grid_size], dim=-1)
                grid_box = grid_box.view(n_unmatched, 1, 4).repeat(1, n_obj, 1)

                grid_box_intersections = box.inters(grid_box, boxes_gt.view(1, n_obj, 4).repeat(n_unmatched, 1, 1))
                grid_box_intersections = grid_box_intersections.sum(dim=-1)
                tgt_neg_grids = unmatched_grid_indices[grid_box_intersections == 0]
                n_grid_neg = len(tgt_neg_grids)
                print(f'|nng {n_grid_neg}', end="")
                if n_grid_neg == 0:
                    continue
                neg_cids_tgt = torch.zeros([n_grid_neg], dtype=torch.long, device=x.device)
                src_cls_neg_loss = ce(src_cls_logits[b, tgt_neg_grids], neg_cids_tgt) * n_grid_neg
                src_cls_neg_loss_batch += src_cls_neg_loss
                n_tgt_grid_neg_batch += n_grid_neg

            src_cls_neg_loss_batch = src_cls_neg_loss_batch / (n_tgt_grid_neg_batch + 1e-9)
            src_cls_pos_loss_batch = src_cls_pos_loss_batch / (n_tgt_grid_pos_batch + 1e-9)
            cids_as_query = deepcopy(cids_gt_batch)
        else:
            src_cls_recall = 0
            src_cls_accu = 0
            cids_as_query = src_cids

        n_cls_query = max([len(c) for c in cids_as_query])

        if n_cls_query == 0:
            return None, None, src_cls_pos_loss_batch, src_cls_neg_loss_batch, src_cls_recall, src_cls_accu, None,None

        for c in cids_as_query:
            if len(c) < n_cls_query + 2:
                c.extend(random.sample(list(self.cid_set - set(c)), n_cls_query + 2 - len(c)))
        cids_as_query = torch.tensor(cids_as_query, dtype=torch.long, device=x.device)
        cls_query = self.cls_emb_m(cids_as_query)

        if masks is not None:
            dec_masks = torch.ones([bsz, (n_cls_query + 2) * n_pos_query, 1], device=x.device) @ masks
        else:
            dec_masks = None

        boxes, cls_logits = self.decoder(src, src_pos_emb, cls_query, dec_masks)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (src[0] - src[1]).abs().mean().item()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean().item()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, src_cls_pos_loss_batch, src_cls_neg_loss_batch, src_cls_recall, src_cls_accu, enc_diff, logits_diff


if __name__ == '__main__':
    device = torch.device('cpu')
    B = 2
    imgs = torch.rand([B, 3, 512, 512], device=device)
    detr = DETR(d_cont=256, d_head=64, d_src_pos=64, d_tgt_pos=64, n_enc_layer=16, n_dec_layer=6)
    boxes, cls_logits, src_cls_loss, src_cls_recall, enc_diff, logits_diff = detr(imgs)
    print(boxes.shape)
    print(src_cls_recall)

