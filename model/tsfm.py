from torch import nn
import torch
import torch.nn.functional as F
from model import base


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        d = q.shape[-1]
        k = torch.transpose(k, -2, -1)
        attn = F.softmax(q @ k / d ** 0.5, dim=-1) @ v
        return attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, q_dim, k_dim, v_dim):
        super().__init__()
        self.n_head = n_head
        self.q_proj = nn.Linear(q_dim, k_dim, bias=False)
        self.k_proj = nn.Linear(k_dim, k_dim, bias=False)
        self.v_proj = nn.Linear(v_dim, v_dim, bias=False)
        self.attn = Attention()
        self.out_proj = nn.Linear(v_dim, v_dim, bias=False)

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        b, lq, lv = q.shape[0], q.shape[1], v.shape[1]

        q = q.view(b, lq, self.n_head, -1).transpose(1, 2)
        k = k.view(b, lv, self.n_head, -1).transpose(1, 2)
        v = v.view(b, lv, self.n_head, -1).transpose(1, 2)

        attn = self.attn(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(b, lq, -1)

        out = self.out_proj(attn)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, q_dim, v_dim, n_head):
        super().__init__()
        self.v_dim = v_dim
        self.q_ln = nn.LayerNorm(q_dim)
        self.v_ln = nn.LayerNorm(v_dim)
        self.self_attn = MultiheadAttention(n_head, q_dim, q_dim, v_dim)
        self.out_ln = nn.LayerNorm(v_dim)
        self.ffn = base.FFN(v_dim)

    def forward(self, q, v):
        q = self.q_ln(q)
        v = self.v_ln(v)
        x = q[..., :self.v_dim] + self.self_attn(q, q, v)
        x = x + self.ffn(self.out_ln(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, n_head, ommit_sa=False, residual=True):
        super().__init__()
        self.ommit_sa = ommit_sa
        self.residual = residual
        self.v_dim = v_dim
        self.q_sa_ln = nn.LayerNorm(q_dim)
        self.self_attn = MultiheadAttention(n_head, q_dim, q_dim, q_dim)
        self.q_ca_ln = nn.LayerNorm(q_dim)
        self.k_ca_ln = nn.LayerNorm(k_dim)
        self.v_ca_ln = nn.LayerNorm(v_dim)
        self.cross_attn = MultiheadAttention(n_head, q_dim, k_dim, v_dim)
        self.out_ln = nn.LayerNorm(v_dim)
        self.ffn = base.FFN(v_dim)

    def forward(self, q, k, v):
        if not self.ommit_sa:
            q = self.q_sa_ln(q)
            q = q + self.self_attn(q, q, q)
        q = self.q_ca_ln(q)
        k = self.k_ca_ln(k)
        v = self.v_ca_ln(v)
        if self.residual:
            q = q[..., :self.v_dim] + self.cross_attn(q, k, v)
        else:
            q = self.cross_attn(q, k, v)
        q = q + self.ffn(self.out_ln(q))
        return q
