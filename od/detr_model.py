import torch
from torch import nn
import torch.nn.functional as F
from model import base
import util
from od import pe, config_category


def attention(q, k, v):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = F.softmax(q @ k / d ** 0.5, dim=-1) @ v
    return attn


class CrossAttention(nn.Module):
    def __init__(self, n_head, q_dim, v_dim):
        super().__init__()
        self.n_head = n_head
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.q_head_dim = q_dim // n_head
        self.v_head_dim = v_dim // n_head
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(q_dim, q_dim)
        self.v_proj = nn.Linear(v_dim, v_dim)
        self.out_proj = nn.Linear(v_dim, v_dim)
        self.ln = nn.LayerNorm(v_dim)

    def forward(self, q, k, v):
        q_ = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        b, lq, lv = q_.shape[0], q_.shape[1], v.shape[1]

        q_ = q_.view(b, lq, self.n_head, self.q_head_dim).permute([0, 2, 1, 3])
        k = k.view(b, lv, self.n_head, self.q_head_dim).permute([0, 2, 1, 3])
        v = v.view(b, lv, self.n_head, self.v_head_dim).permute([0, 2, 1, 3])

        out = attention(q_, k, v)
        out = out.permute([0, 2, 1, 3]).contiguous().view(b, lq, self.n_head * self.v_head_dim)

        if self.v_dim < self.q_dim:
            q = q[..., :self.v_dim]
        out = self.ln(q + self.out_proj(out))
        return out


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.sa = CrossAttention(n_head, dim, dim)
        self.ffn = base.FFN(dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.sa(x, x, x)
        out = self.ffn(out)
        out = self.ln(x + out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, n_head, q_dim, v_dim, omit_sa=False):
        super().__init__()
        self.omit_sa = omit_sa
        self.sa = CrossAttention(n_head, q_dim, v_dim)
        self.ca = CrossAttention(n_head, q_dim, v_dim)
        self.ffn = base.FFN(v_dim)
        self.ln = nn.LayerNorm(v_dim)

    def forward(self, q_tgt_cont, q_tgt_pos, q_tgt_hw, k_src_pos, k_src_hw, v_src_cont):
        q_sa = torch.concat((q_tgt_cont, q_tgt_pos, q_tgt_hw), dim=-1)
        if not self.omit_sa:
            k_sa = q_sa
            v_sa = q_tgt_cont
            sa_out = self.sa(q_sa, k_sa, v_sa)
            q_ca = torch.concat((sa_out, q_tgt_pos, q_tgt_hw), dim=-1)
        else:
            q_ca = q_sa

        k_ca = torch.concat((v_src_cont, k_src_pos, k_src_hw), dim=-1)
        ca_out = self.ca(q_ca, k_ca, v_src_cont)
        out = self.ln(self.ffn(ca_out))
        return out


class DETR(nn.Module):
    def __init__(self, d_cont, n_head=8, n_enc_layer=6, n_dec_layer=6, n_query=300, device=torch.device('mps')):
        super().__init__()
        self.d_cont = d_cont
        self.d_pos_emb = d_cont // 2
        self.n_query = n_query
        self.n_dec_layer = n_dec_layer
        self.device = device
        self.pe_proj = base.MLP(2, 256, self.d_pos_emb, 2)
        self.hw_proj = base.MLP(2, 256, self.d_pos_emb, 2)
        self.cnn1 = nn.Conv2d(3, d_cont, (2, 2), stride=(2, 2))
        self.cnn2 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn3 = nn.Conv2d(d_cont, d_cont, (2, 2), stride=(2, 2))
        self.cnn_ln = nn.LayerNorm(d_cont)

        encoder_layers = []
        for i in range(n_enc_layer):
            encoder_layers.append(EncoderLayer(d_cont + self.d_pos_emb, n_head))
        self.enc_dec_proj = nn.Linear(d_cont + self.d_pos_emb, d_cont)
        self.encoder = nn.ModuleList(encoder_layers)

        self.pos_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.hw_delta_mlp = base.MLP(d_cont, 256, 2, 2)
        self.src_hw_mlp = base.MLP(d_cont, 256, self.d_pos_emb, 2)
        self.anchor_emb = pe.Embedding1D(n_query, 4, device)
        self.classify_mlp = base.MLP(d_cont, 256, config_category.category_num, 2)

        decoder_layers = []
        for i in range(n_dec_layer):
            if i == 0:
                decoder_layers.append(DecoderLayer(n_head, self.d_cont * 2, d_cont, omit_sa=True))
            else:
                decoder_layers.append(DecoderLayer(n_head, self.d_cont * 2, d_cont))
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        B, H, W, C = x.shape

        positions = pe.gen_pos_2d(x, self.device).view(B, H * W, 2)
        pos_emb = self.pe_proj(positions)

        x = x.view(B, H * W, self.d_cont)
        x = torch.concat((x, pos_emb), dim=-1)

        for enc_layer in self.encoder:
            x = enc_layer(x)

        x = F.layer_norm(F.relu(self.enc_dec_proj(x)), [self.d_cont])

        anchors = self.anchor_emb(B)

        xy = anchors[..., :2] + 0
        wh = anchors[..., 2:] + 0

        q_tgt_pos = self.pe_proj(xy)
        q_tgt_hw = self.hw_proj(wh)
        q_tgt_cont = torch.zeros(B, self.n_query, self.d_cont, device=self.device)

        k_src_pos = pos_emb
        k_src_hw = self.src_hw_mlp(x)

        for i, dec_layer in enumerate(self.decoder):
            v_src_cont = x
            q_tgt_cont = dec_layer(q_tgt_cont, q_tgt_pos, q_tgt_hw, k_src_pos, k_src_hw, v_src_cont)
            tgt_pos_delta = self.pos_delta_mlp(q_tgt_cont)
            tgt_hw_delta = self.hw_delta_mlp(q_tgt_cont)
            xy = F.sigmoid(util.inverse_sigmoid(xy) + tgt_pos_delta)
            wh = F.sigmoid(util.inverse_sigmoid(wh) + tgt_hw_delta)

            if i < self.n_dec_layer - 1:
                q_tgt_pos = self.pe_proj(xy)
                q_tgt_hw = self.hw_proj(wh)

        cls_logits = self.classify_mlp(q_tgt_cont)
        boxes = torch.concat((xy, wh), dim=-1)
        return boxes, cls_logits


if __name__ == '__main__':

    device= torch.device('mps')

    imgs = torch.rand([2, 3, 512, 512], device=device)
    detr = DETR(256, device=device)
    detr.to(device=device)
    categories, anchors = detr(imgs)