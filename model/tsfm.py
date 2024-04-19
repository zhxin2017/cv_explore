from torch import nn
import torch
import torch.nn.functional as F
from model import base, pe


def attention(q, k, mask=None):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = q @ k / d ** 0.5
    if mask is not None:
        attn = attn.masked_fill(mask.view(-1, 1, q.shape[-2], k.shape[-1]) == 0, float('-inf'))

    attn = F.softmax(attn, dim=-1)
    return attn


class MHA(nn.Module):
    def __init__(self, dq, dk, dv, n_head, d_match=None, project_v=True):
        super().__init__()
        self.n_head = n_head
        if d_match is None:
            d_match = max(dq, dk, dv)

        self.q_proj = nn.Linear(dq, d_match, bias=False)
        self.k_proj = nn.Linear(dk, d_match, bias=False)

        self.project_v = project_v
        if project_v:
            self.v_proj = nn.Linear(dv, d_match, bias=False)
            self.out_proj = nn.Linear(d_match, dv, bias=False)

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)

        b, lq, lv = q.shape[0], q.shape[1], v.shape[1]

        q = q.view(b, lq, self.n_head, -1).transpose(1, 2)
        k = k.view(b, lv, self.n_head, -1).transpose(1, 2)
        attn = attention(q, k, mask)

        if self.project_v:
            v = self.v_proj(v)
            v = v.view(b, lv, self.n_head, -1).transpose(1, 2)
            v = attn @ v
            v = v.transpose(1, 2).contiguous().view(b, lq, -1)
            v = self.out_proj(v)
        else:
            v = v.view(b, 1, lv, -1)
            v = attn @ v
            v = v.transpose(1, 2).contiguous()
        return v


class AttnLayer(nn.Module):
    def __init__(self, dq, dk, dv, n_head, lp=0):
        super().__init__()
        self.dq = dq
        self.dv = dv
        self.q_ln = nn.LayerNorm(dq)
        self.k_ln = nn.LayerNorm(dk)
        self.v_ln = nn.LayerNorm(dv)

        self.lp = lp
        if lp > 0:
            self.plugin_k_emb_m = pe.Embedding1D(lp, dq)
            self.plugin_v_emb_m = pe.Embedding1D(lp, dq)
            self.plugin_attn = MHA(dq, dq, dq, n_head)

        self.self_attn = MHA(dq, dk, dv, n_head)

        self.out_ln = nn.LayerNorm(dv)
        self.ffn = base.FFN(dv)

    def forward(self, q, k, v, q_res=0, mask=None):
        if self.lp > 0:
            q = self.q_ln(q)
            plugin_k = self.q_ln(self.plugin_k_emb_m(q))
            plugin_v = self.q_ln(self.plugin_v_emb_m(q))
            q = q + self.plugin_attn(q, plugin_k, plugin_v)
        q = self.q_ln(q)
        k = self.k_ln(k)
        v = self.v_ln(v)

        x = q_res[..., :self.dv] + self.self_attn(q, k, v, mask)
        x = x + self.ffn(self.out_ln(x))
        return x
