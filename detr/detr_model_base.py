import torch
from torch import nn
import torch.nn.functional as F
from tsfm import base, pe, transformer, enc
from detr.config import n_cls
from common.config import img_size, patch_size



class DetrEncoder(nn.Module):
    def __init__(self, base_encoder, n_det_enc_layer):
        super().__init__()
        self.base_encoder = base_encoder
        dmodel = self.base_encoder.dmodel
        dhead = self.base_encoder.dhead
        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel
        self.n_det_enc_layer = n_det_enc_layer
        self.det_enc_layers = nn.ModuleList()
        for i in range(n_det_enc_layer):
            self.enc_layers.append(transformer.TsfmLayer(dmodel, dhead))

    def forward(self, x):
        x = self.base_encoder(x)
        for enc_layer in self.det_enc_layers:
            x = enc_layer(x, x, x)
        return x


class DetrDecoder(nn.Module):
    def __init__(self, dmodel, dhead, n_dec_layer, n_query):
        super().__init__()

        self.n_query = n_query
        self.query_emb_m = nn.Embedding(n_query, dmodel)
        self.dec_ca_layers = nn.ModuleList()
        self.dec_sa_layers = nn.ModuleList()

        for i in range(n_dec_layer):
            dec_ca_layer = transformer.TsfmLayer(dmodel, dhead)
            dec_sa_layer = transformer.TsfmLayer(dmodel, dhead)
            self.dec_ca_layers.append(dec_ca_layer)
            self.dec_sa_layers.append(dec_sa_layer)
        self.n_dec_layer = n_dec_layer
        # regression
        self.ln = nn.LayerNorm(dmodel)
        self.box_reg = base.MLP(dmodel, dmodel * 2, 4, 2)
        self.cls_reg = nn.Linear(dmodel, n_cls, bias=False)

    def forward(self, src):
        query_indices = torch.arange(self.n_query, device=src.device).view(1, self.n_query).repeat(src.shape[0], 1)
        q_dec = self.query_emb_m(query_indices)

        for i in range(self.n_dec_layer):
            q_dec = self.dec_ca_layers[i](q_dec, src, src)
            q_dec = self.dec_sa_layers[i](q_dec, q_dec, q_dec)

        boxes = F.sigmoid(self.box_reg(self.ln(q_dec)))
        cls_logits = self.cls_reg(self.ln(q_dec))

        return boxes, cls_logits


class DETR(nn.Module):
    def __init__(self, dmodel, dhead, n_base_enc_layer, n_det_enc_layer, n_dec_layer, n_query):
        super().__init__()
        base_encoder = transformer.Encoder(n_base_enc_layer, dmodel, dhead, patch_size)
        self.det_enc = DetrEncoder(base_encoder, n_det_enc_layer)
        self.det_dec = DetrDecoder(dmodel, dhead, n_dec_layer, n_query)


    def forward(self, x):
        # encoder
        x = self.det_enc(x)
        # regression
        boxes, cls_logits = self.det_dec(x)
        return boxes, cls_logits


if __name__ == '__main__':
    img = torch.randn([4, 3, img_size[0], img_size[1]])
    model = DETR(d_cont=256, d_pos=128, d_head=64, n_dec_layer=4, n_enc_layer=4, n_query=100)
    print(model(img)[0].shape)
