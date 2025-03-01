from torch import nn
import torch
from tsfm import base
import torch.nn.functional as F
from common.config import max_grid_h, max_grid_w


def attention(q, k, mask=None):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = q @ k / d ** 0.5
    if mask is not None:
        attn = attn.masked_fill(mask.view(-1, 1, q.shape[-2], k.shape[-1]) == 0, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return attn


class MHA(nn.Module):
    def __init__(self, dmodel, dhead):
        super().__init__()
        self.n_head = dmodel // dhead
        self.dhead = dhead
        self.q_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.k_proj = nn.Linear(dmodel, dmodel, bias=False)
        self.v_proj = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        b, lq, lv = q.shape[0], q.shape[1], v.shape[1]

        q = q.view(b, lq, self.n_head, self.dhead).transpose(1, 2)
        k = k.view(b, lv, self.n_head, self.dhead).transpose(1, 2)
        v = v.view(b, lv, self.n_head, self.dhead).transpose(1, 2)

        attn = attention(q, k, mask)
        v = attn @ v
        v = v.transpose(1, 2).contiguous().view(b, lq, -1)
        return v


class TsfmLayer(nn.Module):
    def __init__(self, dmodel, dhead):
        super().__init__()
        self.dmodel = dmodel
        self.dhead = dhead
        self.ln_q = nn.LayerNorm(dmodel)
        self.ln_k = nn.LayerNorm(dmodel)
        self.ln_v = nn.LayerNorm(dmodel)
        self.mha = MHA(dmodel, dhead)

        self.ln_ffn = nn.LayerNorm(dmodel)
        self.ffn = base.FFN(dmodel)

    def forward(self, q, k, v, mask=None):
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)
        q = q + self.mha(q, k, v, mask)
        q = q + self.ffn(self.ln_ffn(q))
        return q


class Encoder(nn.Module):
    def __init__(self, nlayer, dmodel, dhead, patch_size):
        super().__init__()
        self.proj = nn.Linear(patch_size * patch_size * 3, dmodel)

        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        self.pos_y_emb_m = nn.Embedding(max_grid_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, dmodel)

        self.n_enc_layer = nlayer
        self.enc_layers = nn.ModuleList()
        for i in range(nlayer):
            self.enc_layers.append(TsfmLayer(dmodel, dhead))

    def forward(self, x, x_shift=0, y_shift=0, mask=None):
        b, patch_h, patch_w, c = x.shape
        num_patch = patch_h * patch_w
        x = x.view(b, num_patch, c)
        x = self.proj(x)
        y_indices = torch.arange(patch_h, device=x.device) + y_shift
        x_indices = torch.arange(patch_w, device=x.device) + x_shift
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, patch_h, 1, self.dmodel).
                     repeat(b, 1, patch_w, 1).view(b, num_patch, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, patch_w, self.dmodel).
                     repeat(b, patch_h, 1, 1).view(b, num_patch, self.dmodel))
        pos_emb = pos_y_emb + pos_x_emb
        x = x + pos_emb

        for enc_layer in self.enc_layers:
            x = enc_layer(x, x, x, mask)
        return x, pos_emb


if __name__ == '__main__':
    encoder = Encoder(4, 64, 8, 16)
    imgs = torch.randn(1, 3, 256, 256)
    x, pos_emb = encoder(imgs)
    print(x.shape)