import random

import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
from od import anchor, box
from od.config import n_cls, anchor_stride, anchor_max_size
from common.config import patch_size, grid_size_y, grid_size_x


class DetrEncoder(nn.Module):
    def __init__(self, d_enc, d_coord_emb, n_head, n_enc_layer):
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


class DetrDecoder2(nn.Module):
    def __init__(self, d_src, d_extremity, d_coord_emb):
        super().__init__()
        self.d_extremity = d_extremity
        self.d_src = d_src
        self.d_cont = self.d_src - d_extremity
        self.d_coord_emb= d_coord_emb
        self.d_anchor = d_coord_emb * 4

        self.anchor_center_adjust_reg = base.MLP(self.d_src, self.d_src * 4, 2, 2)
        self.anchor_extremity_adjust_reg = base.MLP(self.d_src, self.d_src * 4, 1, 2)

        self.pos_emb_m = pe.Sinusoidal(d_coord_emb)

        self.extremity_emb_m = pe.Embedding1D(1, d_extremity)

        self.anchors = anchor.generate_anchors()
        self.n_anchor = len(self.anchors)

        self.ln = nn.LayerNorm(self.d_cont)
        self.cls_reg = nn.Linear(self.d_cont, n_cls)

    def forward(self, src, src_pos):
        B = src.shape[0]

        src_pos_emb = self.pos_emb_m(src_pos)

        anchors = torch.tensor(self.anchors, device=src.device).unsqueeze(0).repeat(B, 1, 1)
        anchors_center = box.xyxy_to_cxcy(anchors)
        anchors_center_emb = self.pos_emb_m(anchors_center)
        features_attn = tsfm.attention(anchors_center_emb, src_pos_emb)
        center_features = features_attn @ src

        contents = center_features[..., :self.d_cont]
        cls_logits1 = self.cls_reg(self.ln(contents))

        anchor_center_shift = F.tanh(self.anchor_center_adjust_reg(center_features)) / (anchor_max_size / anchor_stride) / 2
        anchors = anchors + anchor_center_shift.repeat(1, 1, 2)
        anchors_emb = self.pos_emb_m(anchors).view(B, self.n_anchor, 4, self.d_coord_emb).transpose(1, 2)

        contents = contents.view(B, 1, self.n_anchor, self.d_cont).expand(B, 4, self.n_anchor, self.d_cont)
        extremity_emb = self.extremity_emb_m(anchors).view(B, 1, 1, self.d_extremity).expand(B, 4, self.n_anchor, self.d_extremity)
        q_extremity = torch.concat([contents, extremity_emb, anchors_emb], dim=-1)

        l_src = grid_size_y * grid_size_x
        src = src.view(B, 1, l_src, self.d_src).expand(B, 4, l_src, self.d_src)
        src_pos_emb = torch.concat([src_pos_emb, src_pos_emb], dim=-1).view(B, l_src, 4, self.d_coord_emb).transpose(1, 2)
        k_extremity = torch.concat([src, src_pos_emb], dim=-1)

        extremity_features_attn = tsfm.attention(q_extremity, k_extremity)
        extremity_features = extremity_features_attn @ src

        extremity_features_ = extremity_features.mean(dim=1)
        cls_logits2 = self.cls_reg(self.ln(extremity_features_[..., :self.d_cont]))

        extremity_features = extremity_features.transpose(1, 2).reshape(B, self.n_anchor * 4, -1)
        anchor_shift = self.anchor_extremity_adjust_reg(extremity_features).view(B, self.n_anchor, 4)
        anchors = anchors + anchor_shift

        return anchors, cls_logits1, cls_logits2, center_features, extremity_features, extremity_emb, features_attn, extremity_features_attn


class DETR(nn.Module):

    def __init__(self, d_enc, d_extremity=64, d_coord_emb=64, n_enc_head=8, n_enc_layer=16, exam_diff=True):
        super().__init__()
        d_enc_cont = d_enc - 2 * d_coord_emb
        self.cnn = nn.Conv2d(3, d_enc_cont, (patch_size, patch_size), stride=(patch_size, patch_size))
        self.cnn_ln = nn.LayerNorm(d_enc_cont)

        self.encoder = DetrEncoder(d_enc, d_coord_emb, n_enc_head, n_enc_layer)

        self.decoder = DetrDecoder2(d_enc, d_extremity, d_coord_emb)
        self.exam_diff = exam_diff

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)
        x, pos = self.encoder(x)

        boxes, cls_logits1, cls_logits2, center_features, extremity_features, extremity_emb, features_attn, extremity_features_attn = self.decoder(x, pos)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (x[0] - x[1]).abs().mean()
            logits_diff = (cls_logits2[0] - cls_logits2[1]).abs().mean()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits1, cls_logits2, enc_diff, logits_diff, center_features, extremity_features, extremity_emb, features_attn, extremity_features_attn


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
