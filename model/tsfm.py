from torch import nn
import torch
import torch.nn.functional as F
from model import base


def attention(q, k, v):
    d = q.shape[-1]
    k = torch.transpose(k, -2, -1)
    attn = F.softmax(q @ k / d ** 0.5, dim=-1) @ v
    return attn


class CrossAttention(nn.Module):
    def __init__(self, n_head, q_dim, v_dim, out_dim=None):
        super().__init__()
        self.n_head = n_head
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.q_head_dim = q_dim // n_head
        self.v_head_dim = v_dim // n_head
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(q_dim, q_dim)
        self.v_proj = nn.Linear(v_dim, v_dim)
        if out_dim is None:
            self.out_dim = v_dim
        else:
            self.out_dim = out_dim
        self.q_add_proj = nn.Linear(q_dim, self.out_dim)
        self.out_proj = nn.Linear(v_dim, self.out_dim)
        self.ln = nn.LayerNorm(self.out_dim)

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

        out = self.ln(self.q_add_proj(q) + self.out_proj(out))
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
