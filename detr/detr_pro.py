import random
import scipy
import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm, enc
from detr.config import n_cls, n_pos_query
from common.config import max_grid_y, max_grid_x

ce = nn.CrossEntropyLoss()

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

        self.pos_y_emb_m = nn.Embedding(max_grid_y, d_pos // 2)
        self.pos_x_emb_m = nn.Embedding(max_grid_x, d_pos // 2)

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

        self.cls_proj = nn.Linear(dq, d_cont, bias=False)
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
        cls_logits = self.cont_ln(self.cls_proj(q)) @ self.cls_emb_m.weight.T
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

    def forward(self, x, masks=None, cids_gt_batch=None):
        if masks is not None:
            enc_masks = torch.permute(masks, [0, 2, 1]) @ masks
        else:
            enc_masks = None
        src, src_pos_emb = self.encoder(x, mask=enc_masks)
        bsz = src.shape[0]
        src_cls_logits = self.cont_ln(src) @ self.cls_emb_m.weight.T
        src_cls_prob = F.softmax(src_cls_logits, dim=-1)
        src_cls_prob_max, src_cls = torch.max(src_cls_prob, dim=-1)  # both bsz * seq_len
        src_cids = [list(set(p) - {0}) for p in src_cls.tolist()]

        if self.train:
            assert cids_gt_batch is not None
            matched_cls_logits_batch = []
            matched_cids_batch = []
            n_obj_batch = 0
            tp_batch = 0
            tn_batch = 0
            for b in range(bsz):
                cids_pred = set(src_cids[b])
                cids_gt = cids_gt_batch[b]
                n_obj = len(cids_gt)
                tp_batch += len(set(cids_gt).intersection(cids_pred))
                n_obj_batch += n_obj
                cids_gt_neg = self.cid_set - set(cids_gt)
                tn_batch += len(cids_gt_neg.intersection(self.cid_set - cids_pred))

                cids_extended = cids_gt + list(cids_gt_neg)
                cids_tgt = [0] * n_cls
                cids_tgt[:n_obj] = cids_extended[:n_obj]
                cids_tgt = torch.tensor(cids_tgt, dtype=torch.long, device=x.device)
                prob_to_assign = 1 - src_cls_prob[b,:, cids_extended]  # subtract by 1 to fit hungarian method that minimize cost
                rows, cols = scipy.optimize.linear_sum_assignment(prob_to_assign.detach().cpu().numpy())
                matched_cls_logits_batch.append(src_cls_logits[b, rows])
                matched_cids_batch.append(cids_tgt[cols])
            matched_cls_logits_batch = torch.stack(matched_cls_logits_batch).view(bsz * n_cls, n_cls)
            matched_cids_batch = torch.stack(matched_cids_batch).view(bsz * n_cls)
            src_cls_loss = ce(matched_cls_logits_batch, matched_cids_batch)

            src_cls_recall = tp_batch / n_obj_batch
            src_cls_accu = (tp_batch + tn_batch) / (bsz * n_cls)

            cids_as_query = cids_gt_batch
        else:
            src_cls_loss = 0
            src_cls_recall = 0
            src_cls_accu = 0
            cids_as_query = src_cids

        n_cls_query = max([len(c) for c in cids_as_query])
        for c in cids_as_query:
            if len(c) < n_cls_query:
                c.extend(random.sample(list(self.cid_set - set(c)), n_cls_query - len(c)))
        cids_as_query = torch.tensor(cids_as_query, dtype=torch.long, device=x.device)
        cls_query = self.cls_emb_m(cids_as_query)

        if masks is not None:
            dec_masks = torch.ones([bsz, n_cls_query * n_pos_query, 1], device=x.device) @ masks
        else:
            dec_masks = None

        boxes, cls_logits = self.decoder(src, src_pos_emb, cls_query, dec_masks)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (src[0] - src[1]).abs().mean().item()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean().item()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, src_cls_loss, src_cls_recall, src_cls_accu, enc_diff, logits_diff


if __name__ == '__main__':
    device = torch.device('cpu')
    B = 2
    imgs = torch.rand([B, 3, 512, 512], device=device)
    detr = DETR(d_cont=256, d_head=64, d_src_pos=64, d_tgt_pos=64, n_enc_layer=16, n_dec_layer=6)
    boxes, cls_logits, src_cls_loss, src_cls_recall, enc_diff, logits_diff = detr(imgs)
    print(boxes.shape)
    print(src_cls_recall)

