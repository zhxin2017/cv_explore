import torch
from torch import nn
import torch.nn.functional as F
from model import base, pe, tsfm
import util
from od import config


class DetrDecoderLayer(nn.Module):
    def __init__(self, n_head, q_dim, omit_sa=False):
        super().__init__()
        self.omit_sa = omit_sa
        out_dim = q_dim // 2
        self.sa = tsfm.CrossAttention(n_head, q_dim, q_dim, out_dim=out_dim)
        self.ca = tsfm.CrossAttention(n_head, q_dim, q_dim, out_dim=out_dim)
        self.ffn = base.FFN(out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, q_tgt_cont, q_tgt_xy, q_tgt_wh, k_src_xy, k_src_wh, v_src_cont):
        q_sa = torch.concat((q_tgt_cont, q_tgt_xy, q_tgt_wh), dim=-1)
        if not self.omit_sa:
            sa_out = self.sa(q_sa, q_sa, q_sa)
            q_ca = torch.concat((sa_out, q_tgt_xy, q_tgt_wh), dim=-1)
        else:
            q_ca = q_sa

        k_ca = torch.concat((v_src_cont, k_src_xy, k_src_wh), dim=-1)
        ca_out = self.ca(q_ca, k_ca, k_ca)
        out = self.ln(self.ffn(ca_out))
        return out


class DETR(nn.Module):
    def __init__(self, d_cont, n_head=8, n_enc_layer=6, n_dec_layer=6, n_query=300, device=torch.device('mps')):
        super().__init__()
        self.d_cont = d_cont
        self.d_xy_emb = d_cont // 2
        self.d_wh_emb = d_cont // 2
        d_dec = d_cont * 2
        self.n_query = n_query
        self.n_dec_layer = n_dec_layer
        self.device = device
        self.xy_proj = base.MLP(2, 256, self.d_xy_emb, 2)
        self.hw_proj = base.MLP(2, 256, self.d_wh_emb, 2)
        self.cnn1 = nn.Conv2d(3, d_cont, (2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn3 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn_ln = nn.LayerNorm(d_cont)

        encoder_layers = []
        for i in range(n_enc_layer):
            encoder_layers.append(tsfm.EncoderLayer(d_cont + self.d_xy_emb, n_head))
        self.enc_dec_proj = nn.Linear(d_cont + self.d_xy_emb, d_cont)
        self.encoder = nn.ModuleList(encoder_layers)

        self.xy_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.wh_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.src_wh_mlp = base.MLP(d_cont, 256, self.d_wh_emb, 2)
        self.anchor_emb = pe.Embedding1D(n_query, 4, device)
        self.classify_mlp = base.MLP(d_cont, 256, config.category_num, 2)

        decoder_layers = []
        for i in range(n_dec_layer):
            decoder_layers.append(DetrDecoderLayer(n_head, q_dim=d_dec, omit_sa=(i == 0)))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        B, H, W, C = x.shape

        positions = pe.gen_pos_2d(x, self.device).view(B, H * W, 2)
        xy_emb = self.xy_proj(positions)

        x = x.view(B, H * W, self.d_cont)
        x = torch.concat((x, xy_emb), dim=-1)

        for enc_layer in self.encoder:
            x = enc_layer(x)

        x = F.layer_norm(F.relu(self.enc_dec_proj(x)), [self.d_cont])

        anchors = self.anchor_emb(B)

        xy = anchors[..., :2] + 0
        wh = anchors[..., 2:] + 0

        q_tgt_xy = self.xy_proj(xy)
        q_tgt_wh = self.hw_proj(wh)
        q_tgt_cont = torch.zeros(B, self.n_query, self.d_cont, device=self.device)

        k_src_xy = xy_emb
        k_src_wh = self.src_wh_mlp(x)

        for i, dec_layer in enumerate(self.decoder):
            v_src_cont = x
            q_tgt_cont = dec_layer(q_tgt_cont, q_tgt_xy, q_tgt_wh, k_src_xy, k_src_wh, v_src_cont)
            tgt_xy_delta = self.xy_delta_mlp(q_tgt_cont)
            tgt_wh_delta = self.wh_delta_mlp(q_tgt_cont)
            xy = F.sigmoid(util.inverse_sigmoid(xy) + tgt_xy_delta)
            wh = F.sigmoid(util.inverse_sigmoid(wh) + tgt_wh_delta)

            if i < self.n_dec_layer - 1:
                q_tgt_xy = self.xy_proj(xy)
                q_tgt_wh = self.hw_proj(wh)

        cls_logits = self.classify_mlp(q_tgt_cont)
        boxes = torch.concat((xy, wh), dim=-1)
        return boxes, cls_logits


if __name__ == '__main__':
    device = torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)
