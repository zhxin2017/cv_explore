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
        return x, pos_emb


class DetrDecoder(nn.Module):
    def __init__(self, n_head, n_dec_layer, d_src, d_enc_coord_emb, d_dec_coord_emb, anchors):
        super().__init__()
        self.n_dec_layer = n_dec_layer

        self.ln = nn.LayerNorm(d_src)
        d_src_p = d_src + d_enc_coord_emb * 2

        self.dec_pos_emb_m = pe.Sinusoidal(d_dec_coord_emb)

        self.anchors = anchors
        self.n_anchor = len(self.anchors)

        self.content_matcher = tsfm.AttnLayer(d_dec_coord_emb * 4, d_src_p, d_src, n_head, residual=False)

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()
        for i in range(n_dec_layer):
            ca_layer = tsfm.AttnLayer(d_src_p, d_src_p, d_src_p, n_head)
            self.ca_layers.append(ca_layer)

            sa_layer = tsfm.AttnLayer(d_src_p, d_src_p, d_src_p, n_head)
            self.sa_layers.append(sa_layer)

        self.ln_with_pos = nn.LayerNorm(d_src_p)
        self.anchor_shift_reg = base.MLP(d_src_p, d_src_p * 2, 4, 2)
        self.cls_reg = nn.Linear(d_src_p, n_cls)

    def forward(self, src, src_pos_emb):
        B = src.shape[0]
        src_with_pos = torch.concat([self.ln(src), src_pos_emb], dim=-1)
        anchors = self.anchors.unsqueeze(0).repeat(B, 1, 1)
        anchors_emb = self.dec_pos_emb_m(anchors).view(B, self.n_anchor, -1)

        contents = self.content_matcher(anchors_emb, src_with_pos, src)
        tgt = torch.concat([self.ln(contents), anchors_emb], dim=-1)

        for i in range(self.n_dec_layer):
            tgt = self.ca_layers[i](tgt, src_with_pos, src_with_pos)
            tgt = self.sa_layers[i](tgt, tgt, tgt)

        anchor_shift = F.tanh(self.anchor_shift_reg(self.ln_with_pos(tgt))) / (anchor_max_size / anchor_stride) / 2
        anchors = anchors + anchor_shift
        cls_logits = self.cls_reg(self.ln_with_pos(tgt))
        return anchors, cls_logits


class DETR(nn.Module):

    def __init__(self, d_enc, d_enc_coord_emb,  d_dec_coord_emb, n_enc_head, n_dec_head, n_enc_layer, n_dec_layer, anchors,
                 exam_diff=True, add_pos_to_src=False):
        super().__init__()
        self.add_pos_to_src = add_pos_to_src
        d_enc_cont = d_enc - 2 * d_enc_coord_emb
        self.cnn = nn.Conv2d(3, d_enc_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_enc_cont)

        self.encoder = DetrEncoder(n_enc_head, n_enc_layer, d_enc, d_enc_coord_emb)
        self.decoder = DetrDecoder(n_dec_head, n_dec_layer, d_enc, d_enc_coord_emb, d_dec_coord_emb, anchors)
        self.exam_diff = exam_diff

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        x, pos_emb = self.encoder(x)

        boxes, cls_logits = self.decoder(x, pos_emb)

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
