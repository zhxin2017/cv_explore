import torch
from torch import nn
import torch.nn.functional as F
from tsfm import base, pe, transformer, enc
from detr.config import n_cls
from common.config import img_size


class DetrDecoder(nn.Module):
    def __init__(self, d_cont, d_enc_coord_emb, d_head, n_dec_layer, n_query):
        super().__init__()

        dq_dec = dk_dec = dv_dec = d_cont + d_enc_coord_emb * 2
        self.n_query = n_query
        self.query_emb_m = nn.Embedding(n_query, dq_dec)
        self.dec_ca_layers = nn.ModuleList()
        self.dec_sa_layers = nn.ModuleList()

        n_head = dq_dec // d_head
        for i in range(n_dec_layer):
            dec_ca_layer = tsfm.Block(dq_dec, dk_dec, dv_dec, n_head)
            dec_sa_layer = tsfm.Block(dq_dec, dk_dec, dv_dec, n_head)
            self.dec_ca_layers.append(dec_ca_layer)
            if i < n_dec_layer - 1:
                self.dec_sa_layers.append(dec_sa_layer)
        self.n_dec_layer = n_dec_layer
        # regression
        self.ln = nn.LayerNorm(dv_dec)
        self.box_reg = base.MLP(dv_dec, dv_dec * 2, 4, 2)
        self.cls_reg = nn.Linear(dv_dec, n_cls, bias=False)

    def forward(self, src, src_pos_emb):
        src_with_pos = torch.concat([src, src_pos_emb], dim=-1)
        query_indices = torch.arange(self.n_query, device=src.device).view(1, self.n_query).repeat(src.shape[0], 1)
        q_dec = self.query_emb_m(query_indices)

        for i in range(self.n_dec_layer):
            q_dec = self.dec_ca_layers[i](q_dec, src_with_pos, src_with_pos, q_dec)
            if i < self.n_dec_layer - 1:
                q_dec = self.dec_sa_layers[i](q_dec, q_dec, q_dec, q_dec)

        boxes = F.sigmoid(self.box_reg(self.ln(q_dec)))
        cls_logits = self.cls_reg(self.ln(q_dec))

        return boxes, cls_logits


class DETR(nn.Module):
    def __init__(self, d_cont, d_enc_coord_emb, d_head, n_enc_layer, n_dec_layer, n_query, exam_diff=True):
        super().__init__()

        self.encoder = enc.Encoder(n_enc_layer, d_cont, d_head, d_enc_coord_emb)
        self.decoder = DetrDecoder(d_cont, d_enc_coord_emb, d_head, n_dec_layer, n_query)

        self.exam_diff = exam_diff

    def forward(self, x):

        # encoder
        src, pos_emb = self.encoder(x)

        # regression
        boxes, cls_logits = self.decoder(src, pos_emb)

        if self.exam_diff and x.shape[0] > 1:
            enc_diff = (src[0] - src[1]).abs().mean()
            logits_diff = (cls_logits[0] - cls_logits[1]).abs().mean()
        else:
            enc_diff = 0
            logits_diff = 0

        return boxes, cls_logits, enc_diff, logits_diff


if __name__ == '__main__':
    img = torch.randn([4, 3, img_size[0], img_size[1]])
    model = DETR(d_cont=256, d_pos=128, d_head=64, n_dec_layer=4, n_enc_layer=4, n_query=100)
    print(model(img)[0].shape)
