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
            d_match = min(dq, dk)

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
        k = k.view(b, lq, self.n_head, -1).transpose(1, 2)
        attn = attention(q, k)

        if self.project_v:
            v = self.v_proj(v)
            v = v.view(b, lv, self.n_head, -1).transpose(1, 2)
            v = attn @ v
            v = v.transpose(1, 2).contiguous().view(b, lq, -1)
            v = self.out_proj(v)
        else:
            attn = attn.mean(dim=1)
            v = attn @ v
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

        if dq != dv:
            self.q_residual_proj = nn.Linear(dq, dv, bias=False)

    def forward(self, q, k, v):
        q = self.q_ln(q)
        k = self.k_ln(k)
        v = self.v_ln(v)

        if self.dq != self.dv:
            q_residual = self.q_residual_proj(q)
        else:
            q_residual = q

        x = q_residual + self.self_attn(q, k, v)
        x = x + self.ffn(self.out_ln(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_q, d_k, d_v, n_head, ommit_sa=False, ca_residual=True):
        super().__init__()
        self.ommit_sa = ommit_sa
        self.ca_residual = ca_residual
        self.d_v = d_v
        self.q_sa_ln = nn.LayerNorm(d_q)
        self.self_attn = MHA(d_q, d_q, d_q, n_head)
        self.q_ca_ln = nn.LayerNorm(d_q)
        self.k_ca_ln = nn.LayerNorm(d_k)
        self.v_ca_ln = nn.LayerNorm(d_v)
        self.cross_attn = MHA(d_q, d_k, d_v, n_head)
        self.out_ln = nn.LayerNorm(d_v)
        self.ffn = base.FFN(d_v)

    def forward(self, q, k, v):
        if not self.ommit_sa:
            q = self.q_sa_ln(q)
            q = q + self.self_attn(q, q, q)
        q = self.q_ca_ln(q)
        k = self.k_ca_ln(k)
        v = self.v_ca_ln(v)
        if self.ca_residual:
            q = q[..., :self.d_v] + self.cross_attn(q, k, v)
        else:
            q = self.cross_attn(q, k, v)
        q = q + self.ffn(self.out_ln(q))
        return q
