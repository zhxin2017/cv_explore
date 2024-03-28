from torch import nn
import torch
import torch.nn.functional as F
from model import base


def attention(q, k):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = F.softmax(q @ k / d ** 0.5, dim=-1)
    return attn


class MHA(nn.Module):
    def __init__(self, dq, dk, dv, n_head, project_v=True, d_head=None):
        super().__init__()
        self.n_head = n_head
        if d_head is not None:
            d_match = n_head * d_head
        else:
            d_match = max(dq, dk, dv)

        self.q_proj = nn.Linear(dq, d_match, bias=False)
        self.k_proj = nn.Linear(dk, d_match, bias=False)
        self.project_v = project_v
        if project_v:
            self.v_proj = nn.Linear(dv, d_match, bias=False)
            self.out_proj = nn.Linear(d_match, dv, bias=False)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)

        b, lq, lv = q.shape[0], q.shape[1], v.shape[1]

        q = q.view(b, lq, self.n_head, -1).transpose(1, 2)
        k = k.view(b, lv, self.n_head, -1).transpose(1, 2)
        attn = attention(q, k)

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
    def __init__(self, dq, dk, dv, n_head):
        super().__init__()
        self.dq = dq
        self.dv = dv
        self.q_ln = nn.LayerNorm(dq)
        self.k_ln = nn.LayerNorm(dk)
        self.v_ln = nn.LayerNorm(dv)

        self.out_ln = nn.LayerNorm(dv)

        self.self_attn = MHA(dq, dk, dv, n_head)
        self.ffn = base.FFN(dv)

    def forward(self, q, k, v, q_res=0):
        q = self.q_ln(q)
        k = self.k_ln(k)
        v = self.v_ln(v)

        x = q_res + self.self_attn(q, k, v)
        x = x + self.ffn(self.out_ln(x))
        return x
