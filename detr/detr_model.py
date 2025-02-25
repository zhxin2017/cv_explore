import torch
from torch import nn
import torch.nn.functional as F
from tsfm import base, pe, transformer, enc
from detr.config import n_cls


class DetrDecoder(nn.Module):
    def __init__(self, n_dec_layer, d_cont, d_head, d_src_coord_emb, d_tgt_coord_emb):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        self.d_src_coord_emb = d_src_coord_emb
        self.dec_pos_emb_m = pe.Sinusoidal(d_tgt_coord_emb)
        self.dec_anchor_cls_emb_m = pe.Embedding1D(6, d_cont)
        self.src_ln = nn.LayerNorm(d_cont)

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()

        dq = d_cont + 4 * d_tgt_coord_emb

        n_head = dq // d_head
        dk = d_cont + 2 * d_src_coord_emb
        for i in range(n_dec_layer):

            ca_layer = tsfm.Block(dq, dk, dk, n_head)
            self.ca_layers.append(ca_layer)

            if i < n_dec_layer:

                sa_layer = tsfm.Block(dq, dq, dk, n_head)
                self.sa_layers.append(sa_layer)

        self.anchor_shift_reg = base.MLP(dk, dk * 2, 4, 2)
        self.out_ln = nn.LayerNorm(dk)
        self.cls_reg = nn.Linear(dk, n_cls, bias=False)

    def forward(self, src, src_pos_emb, anchors_pos):
        B = src.shape[0]
        n_anchor_pos = anchors_pos.shape[0]
        anchors_pos = anchors_pos.view(1, n_anchor_pos, 1, -1).repeat(B, 1, 6, 1).reshape(B, n_anchor_pos * 6, -1)
        anchors_pos_emb = self.dec_pos_emb_m(anchors_pos)
        anchors_cls_emb = self.dec_anchor_cls_emb_m(src).view(B, 1, 6, -1).repeat(1, n_anchor_pos, 1, 1)\
            .reshape(B, n_anchor_pos * 6, -1)
        q = torch.concat([anchors_cls_emb, anchors_pos_emb], dim=-1)
        src_with_pos = torch.concat([self.src_ln(src), src_pos_emb], dim=-1)

        for i in range(self.n_dec_layer):

            q = self.ca_layers[i](q, src_with_pos, src_with_pos, q)

            if i < self.n_dec_layer - 1:
                q = self.sa_layers[i](q, q, q, q)

        anchor_shift = F.tanh(self.anchor_shift_reg(self.out_ln(q))) / 3
        boxes = anchors_pos + anchor_shift
        cls_logits = self.cls_reg(self.out_ln(q))
        return boxes, cls_logits


class DETR(nn.Module):

    def __init__(self, d_cont, d_head, d_enc_coord_emb, d_dec_coord_emb, n_enc_layer, n_dec_layer,
                 exam_diff=True):
        super().__init__()
        self.encoder = enc.Encoder(n_enc_layer, d_cont, d_head, d_enc_coord_emb)
        self.decoder = DetrDecoder(n_dec_layer, d_cont, d_head, d_enc_coord_emb, d_dec_coord_emb)
        self.exam_diff = exam_diff

    def forward(self, x, anchors):

        src, src_pos_emb = self.encoder(x)

        boxes, cls_logits = self.decoder(src, src_pos_emb, anchors)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (src[0] - src[1]).abs().mean().item()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean().item()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, enc_diff, logits_diff


if __name__ == '__main__':
    device = torch.device('mps')
    B = 2
    imgs = torch.rand([B, 3, 512, 512], device=device)
