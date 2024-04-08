import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm, enc
from od.config import n_cls
from od import box, anchor


class DetrDecoder(nn.Module):
    def __init__(self, n_dec_layer, d_cont, d_head, d_src_coord_emb, d_tgt_coord_emb, d_assist):
        super().__init__()
        self.n_dec_layer = n_dec_layer
        self.d_src_coord_emb = d_src_coord_emb
        self.dec_pos_emb_m = pe.Sinusoidal(d_tgt_coord_emb)

        self.ca_layers = nn.ModuleList()
        self.sa_layers = nn.ModuleList()
        self.assist_emb_ms = nn.ModuleList()

        self.anchor_shift_reg = base.MLP(d_cont, d_cont * 2, 4, 2)

        self.d_src = d_cont
        dq_with_assist = d_cont + 4 * d_tgt_coord_emb + d_assist
        dq_with_pos = d_cont + 4 * d_tgt_coord_emb
        n_head_with_assist = dq_with_assist // d_head
        n_head_with_pos = dq_with_pos // d_head
        dk = d_cont + 2 * d_src_coord_emb
        for i in range(n_dec_layer):
            assist_emb_m = pe.Embedding1D(1, d_assist)
            self.assist_emb_ms.append(assist_emb_m)

            ca_layer = tsfm.AttnLayer(dq_with_assist, dk, d_cont, n_head_with_assist)
            self.ca_layers.append(ca_layer)

            sa_layer = tsfm.AttnLayer(dq_with_pos, dq_with_pos, d_cont, n_head_with_pos)
            self.sa_layers.append(sa_layer)


        self.src_ln = nn.LayerNorm(d_cont)
        self.cls_reg = nn.Linear(d_cont, n_cls, bias=False)

    def forward(self, src, src_pos_emb, anchors):
        B = src.shape[0]
        n_anchor = anchors.shape[0]
        anchors = anchors.unsqueeze(0).repeat(B, 1, 1)
        anchors_emb = self.dec_pos_emb_m(anchors).view(B, n_anchor, -1)
        src_with_pos = torch.concat([self.src_ln(src), src_pos_emb], dim=-1)

        q = torch.zeros([B, n_anchor, self.d_src], device=src.device)
        for i in range(self.n_dec_layer):
            assist_emb = self.assist_emb_ms[i](src).repeat(1, n_anchor, 1)

            q_with_assist = torch.concat([self.src_ln(q), assist_emb, anchors_emb], dim=-1)
            q = self.ca_layers[i](q_with_assist, src_with_pos, src, q_with_assist)

            if i == 1:
                anchor_shift_limit = 1 / 8
                anchor_shift = F.tanh(self.anchor_shift_reg(self.src_ln(q))) * anchor_shift_limit

                anchors = anchors + anchor_shift
                anchors_emb = self.dec_pos_emb_m(anchors).view(B, n_anchor, -1)

            q_with_pos = torch.concat([self.src_ln(q), anchors_emb], dim=-1)
            q = self.sa_layers[i](q_with_pos, q_with_pos, q, q_with_pos)

        cls_logits = self.cls_reg(self.src_ln(q))
        return anchors, cls_logits


class DETR(nn.Module):

    def __init__(self, d_cont, d_head, d_enc_coord_emb, d_dec_coord_emb, d_assist, n_enc_layer, n_dec_layer,
                 exam_diff=True):
        super().__init__()
        self.encoder = enc.Encoder(n_enc_layer, d_cont, d_head, d_enc_coord_emb)
        self.decoder = DetrDecoder(n_dec_layer, d_cont, d_head, d_enc_coord_emb, d_dec_coord_emb, d_assist)
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
