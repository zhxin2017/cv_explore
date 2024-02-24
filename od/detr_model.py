import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
import util
from od import config


class DetrDecoderLayer(nn.Module):
    def __init__(self, n_head, d_query, d_cont, omit_sa=False):
        super().__init__()
        self.omit_sa = omit_sa
        self.sa = tsfm.CrossAttention(n_head, d_query, d_query)
        self.ca = tsfm.CrossAttention(n_head, d_query, d_query)
        self.ffn = base.MLP(d_query, d_query * 2, d_cont, 2)
        self.ln = nn.LayerNorm(d_cont)

    def forward(self, q_tgt_cont, q_tgt_x1y1, q_tgt_x2y2, k_src_xy, v_src_cont):
        q_ca = torch.concat((q_tgt_cont, q_tgt_x2y2, q_tgt_x1y1), dim=-1)
        k_ca = torch.concat((v_src_cont, k_src_xy, k_src_xy), dim=-1)
        v_ca = k_ca
        ca_out = self.ca(q_ca, k_ca, v_ca)

        if not self.omit_sa:
            sa_out = self.sa(ca_out, ca_out, ca_out)
        else:
            sa_out = ca_out

        out = self.ln(F.relu(self.ffn(sa_out)))
        return out


class DETR(nn.Module):
    def __init__(self, d_cont, n_head=8, n_enc_layer=6, n_dec_layer=6, n_query=300, device=torch.device('mps')):
        super().__init__()
        self.d_cont = d_cont
        self.d_coord = d_cont // 2
        d_dec = d_cont * 2
        self.n_query = n_query
        self.n_dec_layer = n_dec_layer
        self.device = device
        self.cnn1 = nn.Conv2d(3, d_cont, (2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn3 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn_ln = nn.LayerNorm(d_cont)

        encoder_layers = []
        for i in range(n_enc_layer):
            encoder_layers.append(tsfm.EncoderLayer(d_cont + self.d_coord, n_head))
        self.enc_dec_proj = nn.Linear(d_cont + self.d_coord, d_cont)
        self.encoder = nn.ModuleList(encoder_layers)

        self.yx_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.hw_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.anchor_yx_emb = pe.Embedding1D(n_query, 2, device)
        self.anchor_hw_emb = pe.Embedding1D(n_query, 2, device)
        self.classify_mlp = base.MLP(d_cont, 256, config.category_num, 2)

        decoder_layers = []
        for i in range(n_dec_layer):
            decoder_layers.append(DetrDecoderLayer(n_head, d_dec, d_cont, omit_sa=(i == n_dec_layer - 1)))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        B, H, W, C = x.shape

        enc_yx = pe.gen_pos_2d(x, self.device).view(B, H * W, 2)
        enc_yx_emb = pe.sinusoidal_encoding(enc_yx, self.d_coord // 2, device=self.device)

        x = x.view(B, H * W, self.d_cont)
        x = torch.concat((x, enc_yx_emb), dim=-1)

        for enc_layer in self.encoder:
            x = enc_layer(x)

        x = F.layer_norm(F.relu(self.enc_dec_proj(x)), [self.d_cont])

        y1x1 = self.anchor_yx_emb(B)
        hw = self.anchor_hw_emb(B)
        y2x2 = y1x1 + hw

        q_tgt_y1x1 = pe.sinusoidal_encoding(y1x1, self.d_coord // 2, device=self.device)
        q_tgt_y2x2 = pe.sinusoidal_encoding(y2x2, self.d_coord // 2, device=self.device)
        q_tgt_cont = torch.zeros(B, self.n_query, self.d_cont, device=self.device)

        k_src_yx = enc_yx_emb

        for i, dec_layer in enumerate(self.decoder):
            v_src_cont = x
            q_tgt_cont = dec_layer(q_tgt_cont, q_tgt_y1x1, q_tgt_y2x2, k_src_yx, v_src_cont)
            tgt_y1x1_delta = self.yx_delta_mlp(q_tgt_cont)
            tgt_hw_delta = self.hw_delta_mlp(q_tgt_cont)
            y1x1 = F.sigmoid(util.inverse_sigmoid(y1x1) + tgt_y1x1_delta)
            hw = F.sigmoid(util.inverse_sigmoid(hw) + tgt_hw_delta)
            y2x2 = y1x1 + hw

            if i < self.n_dec_layer - 1:
                q_tgt_y1x1 = pe.sinusoidal_encoding(y1x1, self.d_coord // 2, device=self.device)
                q_tgt_y2x2 = pe.sinusoidal_encoding(y2x2, self.d_coord // 2, device=self.device)

        cls_logits = self.classify_mlp(q_tgt_cont)

        xy_index = [1, 0]
        x1y1 = y1x1[..., xy_index]
        x2y2 = y2x2[..., xy_index]
        boxes = torch.concat((x1y1, x2y2), dim=-1)
        return boxes, cls_logits


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
