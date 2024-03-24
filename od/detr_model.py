import random

import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor, box
from od.config import n_cls, anchor_stride, anchor_max_size
from common.config import patch_size, grid_size_y, grid_size_x


class DetrEncoder(nn.Module):
    def __init__(self, n_head, n_enc_layer, d_enc, d_coord_emb):
        super().__init__()

        self.pos_emb_m = pe.Sinusoidal(d_coord_emb)

        self.n_enc_layer = n_enc_layer
        self.attn_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.attn_layers.append(tsfm.AttnLayer(d_enc, d_enc, d_enc, n_head))

    def forward(self, x):
        bsz = x.shape[0]
        pos = pe.gen_pos_2d(x, pos='center').view(bsz, grid_size_y * grid_size_x, 2)

        pos_emb = self.pos_emb_m(pos)

        x = x.view(bsz, grid_size_y * grid_size_x, -1)
        x = torch.concat([x, pos_emb], dim=-1)

        for i in range(self.n_enc_layer):
            x = self.attn_layers[i](x, x, x)
        return x, pos


class DetrDecoder1(nn.Module):
    def __init__(self, n_head, n_dec_layer, d_src, d_coord_emb):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        d_anchor = d_coord_emb * 4
        d_pos_emb_src = d_coord_emb * 2
        self.pos_emb_m = pe.Sinusoidal(d_coord_emb)

        self.anchors = anchor.generate_anchors()
        self.n_anchor = len(self.anchors)

        self.attn1 = tsfm.MHA(d_anchor, d_pos_emb_src, d_src, n_head=4, project_v=False)
        dq = d_src + d_anchor
        dk = d_src + d_pos_emb_src
        self.decoder_ca_layers = nn.ModuleList()
        self.decoder_sa_layers = nn.ModuleList()
        for i in range(n_dec_layer):
            ca_layer = tsfm.AttnLayer(dq, dk, d_src, n_head)
            sa_layer = tsfm.AttnLayer(dq, dq, d_src, n_head)
            self.decoder_ca_layers.append(ca_layer)
            self.decoder_sa_layers.append(sa_layer)

        self.ln = nn.LayerNorm(d_src)
        self.cls_reg = nn.Linear(d_src, n_cls)
        self.anchor_shift_reg = base.MLP(d_src, d_src * 2, 4, 2)

    def forward(self, src, src_pos):
        B = src.shape[0]

        src_pos_emb = self.pos_emb_m(src_pos)
        src_with_pos = torch.concat([src, src_pos_emb], dim=-1)
        anchors = torch.tensor(self.anchors, device=src.device).unsqueeze(0).repeat(B, 1, 1)
        anchors_emb = self.pos_emb_m(anchors).view(B, self.n_anchor, -1)

        features = self.attn1(anchors_emb, src_pos_emb, src).mean(dim=-2)  # mean of different heads
        features = self.ln(features)

        for i in range(self.n_dec_layer):
            q_ca = torch.concat([features, anchors_emb], dim=-1)
            features = self.decoder_ca_layers[i](q_ca, src_with_pos, src)
            q_sa = torch.concat([features, anchors_emb], dim=-1)
            features = self.ln(self.decoder_sa_layers[i](q_sa, q_sa, features))
            anchor_shift = F.tanh(self.anchor_shift_reg(features)) / (anchor_max_size / anchor_stride) / 2
            anchors = anchors + anchor_shift
            anchors_emb = self.pos_emb_m(anchors)

        cls_logits = self.cls_reg(features)
        return anchors, cls_logits


class DETR(nn.Module):

    def __init__(self, d_enc, d_coord_emb=64, n_enc_head=8, n_dec_head=8, n_enc_layer=16, n_dec_layer=6,
                 exam_diff=True):
        super().__init__()
        d_enc_cont = d_enc - 2 * d_coord_emb
        self.cnn = nn.Conv2d(3, d_enc_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_enc_cont)

        self.encoder = DetrEncoder(n_enc_head, n_enc_layer, d_enc, d_coord_emb)
        self.decoder = DetrDecoder1(n_dec_head, n_dec_layer, d_enc, d_coord_emb)
        self.exam_diff = exam_diff

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        x, pos = self.encoder(x)

        boxes, cls_logits = self.decoder(x, pos)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (x[0] - x[1]).abs().mean()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, enc_diff, logits_diff


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    # detr = DETR(256, device=device)
    # detr.to(device=device)
    # categories, anchors = detr(imgs)
