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
