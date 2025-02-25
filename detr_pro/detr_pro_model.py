import random
import math
import scipy
import torch
from torch import nn
import torch.nn.functional as F
import sys

sys.path.append('..')
from tsfm import base, pe, transformer
from detr_pro import assist
from detr.config import n_cls, n_pos_query
from common.config import max_grid_h, max_grid_w, patch_size, max_img_len

ce = nn.CrossEntropyLoss()
grid_size = patch_size / max_img_len
grid_area = grid_size ** 2


class DetrEncoder(nn.Module):
    def __init__(self, n_enc_layer, dmodel, dhead, pretrain=False):
        super().__init__()

        self.cnn = nn.Conv2d(3, dmodel, (patch_size, patch_size),
                             stride=(patch_size, patch_size))
        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        n_head = dmodel // dhead

        self.pos_y_emb_m = nn.Embedding(max_grid_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, dmodel)

        self.n_enc_layer = n_enc_layer
        self.enc_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.enc_layers.append(tsfm.Block(dmodel, dmodel, dmodel, n_head))
        self.pretrain = pretrain
        if pretrain:
            self.next_token_emb_m = pe.Embedding1D(1, dmodel)

    def forward(self, x, x_shift=0, y_shift=0, mask=None):
        x = F.relu(self.cnn(x))
        x = x.permute([0, 2, 3, 1])
        x = self.ln(x)

        b, h, w, c = x.shape
        num_grid = h * w
        y_indices = torch.arange(h, device=x.device) + y_shift
        x_indices = torch.arange(w, device=x.device) + x_shift
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, h, 1, self.dmodel).
                     repeat(b, 1, w, 1).view(b, num_grid, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, w, self.dmodel).
                     repeat(b, h, 1, 1).view(b, num_grid, self.dmodel))
        pos_emb = self.ln(pos_y_emb + pos_x_emb)
        x = x.view(b, num_grid, -1)

        if self.pretrain:
            next_token_emb = self.ln(self.next_token_emb_m(x).repeat(1, num_grid, 1))
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

        x = x + pos_emb

        for enc_layer in self.enc_layers:
            x = enc_layer(x, x, x, x, mask)
        return x, pos_emb


class DetrDecoder(nn.Module):
    def __init__(self, n_dec_layer, dmodel, d_head, cls_emb_m):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        self.pos_query_emb_m = nn.Embedding(n_pos_query, dmodel)
        self.cls_emb_m = cls_emb_m
        self.ln = nn.LayerNorm(dmodel)

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()

        n_head = dmodel // d_head

        for i in range(n_dec_layer):

            ca_layer = tsfm.Block(dmodel, dmodel, dmodel, n_head)
            self.ca_layers.append(ca_layer)

            if i < n_dec_layer:
                sa_layer = tsfm.Block(dmodel, dmodel, dmodel, n_head)
                self.sa_layers.append(sa_layer)

        self.cls_reg = nn.Linear(dmodel, n_cls, bias=False)
        self.box_reg = base.MLP(dmodel, dmodel * 2, 4, 2)

    def forward(self, src, cls_query, ca_mask=None):
        B = src.shape[0]
        pos_query = self.pos_query_emb_m(torch.arange(n_pos_query, device=src.device))
        n_cls_query = cls_query.shape[1]
        pos_query = pos_query.view(1, 1, n_pos_query, -1).repeat(B, n_cls_query, 1, 1)
        cls_query = cls_query.view(B, n_cls_query, 1, -1).repeat(1, 1, n_pos_query, 1)
        q = (cls_query + pos_query).reshape(B, n_cls_query * n_pos_query, -1)

        for i in range(self.n_dec_layer):

            q = self.ca_layers[i](q, src, src, q, ca_mask)

            if i < self.n_dec_layer - 1:
                n_query = n_cls_query * n_pos_query
                sa_mask = torch.zeros([n_query, n_query], device=src.device)
                for m in range(n_cls_query):
                    sa_mask[m * n_pos_query: (m + 1) * n_pos_query, m * n_pos_query: (m + 1) * n_pos_query] = 1
                q = self.sa_layers[i](q, q, q, q, sa_mask)

        boxes = F.sigmoid(self.box_reg(self.ln(q)))
        cls_logits = self.cls_reg(self.ln(q))
        return boxes, cls_logits


class DETR(nn.Module):

    def __init__(self, dmodel, dhead, n_enc_layer, n_dec_layer, exam_diff=True, train=True):
        super().__init__()
        self.encoder = DetrEncoder(n_enc_layer, dmodel, dhead)
        self.cls_emb_m = nn.Embedding(n_cls, dmodel)
        self.ln = nn.LayerNorm(dmodel)
        self.decoder = DetrDecoder(n_dec_layer, dmodel, dhead, self.cls_emb_m)
        self.exam_diff = exam_diff
        self.train = train
        self.cid_set = set(range(n_cls)) - {0}

    def forward(self, x, x_shift=0, y_shift=0, masks=None, cids_gt_batch=None, boxes_gt_batch=None):
        bsz = x.shape[0]
        h, w = x.shape[2] // patch_size, x.shape[3] // patch_size
        n_grid = h * w
        if masks is not None:
            enc_masks = torch.permute(masks, [0, 2, 1]) @ masks
            diag = torch.diag(torch.ones(n_grid, device=x.device)).view(1, n_grid, n_grid)
            enc_masks = enc_masks * (1 - diag) + diag
        else:
            enc_masks = None
        src, src_pos_emb = self.encoder(x, x_shift, y_shift, mask=enc_masks)
        src_with_pos= src + src_pos_emb

        src_cls_logits = self.ln(src) @ self.cls_emb_m.weight.T

        src_cls_prob = F.softmax(src_cls_logits, dim=-1)
        src_cls_prob_max, src_cls = torch.max(src_cls_prob, dim=-1)  # both bsz * seq_len
        src_cids = [list(set(p) - {0}) for p in src_cls.tolist()]

        src_cls_pos_loss_batch = 0
        src_cls_neg_loss_batch = 0

        grid_bgd_indices_batch = []
        grid_obj_indices_batch = []
        grid_obj_cids_batch = []

        if self.train:
            assert cids_gt_batch is not None
            tp_batch = sum(len(set(cids_gt_batch[b]).intersection(set(src_cids[b]))) for b in range(bsz))
            tn_batch = sum(len(self.cid_set - set(cids_gt_batch[b]) - set(src_cids[b])) for b in range(bsz))
            n_cid_gt = sum([len(set(cids)) for cids in cids_gt_batch])
            src_cls_recall = tp_batch / n_cid_gt
            src_cls_accu = (tp_batch + tn_batch) / (n_cls - 1) / bsz

            n_tgt_grid_pos_batch = 0
            n_tgt_grid_neg_batch = 0

            for b in range(bsz):
                boxes_gt = boxes_gt_batch[b]
                boxes_gt = boxes_gt.to(x.device)

                # match negative samples
                grid_bgd_indices, grid_obj_indices, grid_box_intersections, grid_indices,grid_x1, grid_y1, grid_x2, grid_y2 = (
                    assist.separate_bgd_grids(w, h, x_shift, y_shift, boxes_gt))
                grid_bgd_indices_batch.append(grid_bgd_indices)

                n_grid_bgd = len(grid_bgd_indices)
                print(f'|nng {" " * (3 - len(str(n_grid_bgd)))}{n_grid_bgd}', end="")
                if n_grid_bgd > 0:
                    neg_cids_tgt = torch.zeros([n_grid_bgd], dtype=torch.long, requires_grad=False, device=x.device)
                    src_cls_neg_loss = ce(src_cls_logits[b, grid_bgd_indices], neg_cids_tgt) * n_grid_bgd
                    src_cls_neg_loss_batch += src_cls_neg_loss
                    n_tgt_grid_neg_batch += n_grid_bgd

                # match positive samples
                n_obj = len(cids_gt_batch[b])

                cids_tgt = []
                grid_indices_input = []
                for o in range(n_obj):
                    cid = cids_gt_batch[b][o]
                    obj_intersection_mask = grid_box_intersections[:, o] > 0
                    obj_grid_indices = grid_indices[obj_intersection_mask]
                    n_obj_grid = math.ceil(len(obj_grid_indices) / 5)

                    obj_prob = src_cls_prob[b, obj_grid_indices, o]
                    obj_grid_indices_input = obj_grid_indices[obj_prob.argsort()[:n_obj_grid]]

                    cids_tgt.extend([cid] * n_obj_grid)
                    grid_indices_input.append(obj_grid_indices_input)

                n_tgt_grid_pos = len(cids_tgt)
                print(f'|npg {" " * (3 - len(str(n_tgt_grid_pos)))}{n_tgt_grid_pos}', end="")
                n_tgt_grid_pos_batch += n_tgt_grid_pos
                cids_tgt = torch.tensor(cids_tgt, dtype=torch.long, device=x.device)

                grid_indices_input = torch.concat(grid_indices_input, dim=-1)
                grid_obj_indices_batch.append(grid_indices_input)
                grid_obj_cids_batch.append(cids_tgt)

                src_cls_pos_loss = ce(src_cls_logits[b, grid_indices_input], cids_tgt) * n_tgt_grid_pos
                src_cls_pos_loss_batch += src_cls_pos_loss

            src_cls_neg_loss_batch = src_cls_neg_loss_batch / (n_tgt_grid_neg_batch + 1e-9)
            src_cls_pos_loss_batch = src_cls_pos_loss_batch / (n_tgt_grid_pos_batch + 1e-9)
        else:
            src_cls_recall = 0
            src_cls_accu = 0
        if cids_gt_batch is not None:
            cids_set = [list(set(cids)) for cids in cids_gt_batch]
        else:
            cids_set = src_cids

        max_n_cid = max([len(c) for c in cids_set])

        if max_n_cid == 0:
            return (None, None, src_cls_pos_loss_batch, src_cls_neg_loss_batch, src_cls_recall, src_cls_accu, None,
                    src_cls, grid_bgd_indices_batch, grid_obj_indices_batch, grid_obj_cids_batch, None, None)

        sampled_cids = []

        for b, c in enumerate(cids_set):
            # n_neg_query = 0
            neg_cids_fp = set(src_cids[b]) - set(c)
            n_src_cid_fp = len(neg_cids_fp)
            n_neg_query = max(random.randint(2, 8) + max_n_cid - len(c), n_src_cid_fp)
            neg_cids_random = random.sample(list(self.cid_set - set(c) - set(src_cids[b])), n_neg_query - n_src_cid_fp)
            neg_cids = list(neg_cids_fp) + neg_cids_random
            sampled_cids.append(c + neg_cids)

        sampled_cids = torch.tensor(sampled_cids, dtype=torch.long, device=x.device)
        cls_query = self.cls_emb_m(sampled_cids)

        if masks is not None:
            # todo it seems to be wrong here.
            dec_masks = torch.ones([bsz, (max_n_cid + 2) * n_pos_query, 1], device=x.device) @ masks
        else:
            dec_masks = None

        boxes, cls_logits = self.decoder(src_with_pos, cls_query, dec_masks)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (src[0] - src[1]).abs().mean().item()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean().item()
        else:
            enc_diff = 0
            logits_diff = 0

        return (boxes, cls_logits, src_cls_pos_loss_batch, src_cls_neg_loss_batch, src_cls_recall, src_cls_accu,
                src_cls, cids_set, grid_bgd_indices_batch, grid_obj_indices_batch, grid_obj_cids_batch, enc_diff, logits_diff)


if __name__ == '__main__':
    device = torch.device('cpu')
    B = 2
    imgs = torch.rand([B, 3, 512, 512], device=device)
    model = DETR(dmodel=160, dhead=32, n_enc_layer=16, n_dec_layer=6, exam_diff=True, train=False)
    boxes, cls_logits, _, _, _, _, _, _ = model(imgs, cids_gt_batch=[[1], [1]])

    print(boxes.shape)
    print(cls_logits.shape)
