from torch import nn
import torch


class FilteredLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.filter_linear = nn.Linear(in_dim, out_dim)
        self.filter_sigmoid = nn.Sigmoid()
        self.forward_linear = nn.Linear(in_dim, out_dim)
        self.forward_relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x):
        filter_ = self.filter_sigmoid(self.filter_linear(x))
        forward = self.forward_relu(self.forward_linear(x))
        x = self.ln(filter_ * forward)
        return x


class SelectiveAttention(nn.Module):
    def __init__(self, lv, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.seq_len = lv
        self.select_linear = nn.Linear(in_dim, lv)
        self.select_sigmoid = nn.Sigmoid()
        self.ln = nn.LayerNorm(in_dim)
        self.proj = FilteredLinear(in_dim, out_dim)

    def forward(self, v, q):
        attn = self.select_linear(q)
        q = torch.bmm(attn, v)
        q = self.ln(q)
        q = self.proj(q)
        return q


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d, lv):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.lv = lv
        self.heads = nn.ModuleList()
        self.head_dim = d // n_head
        for i in range(n_head):
            head = SelectiveAttention(self.lv, self.d, self.head_dim)
            self.heads.append(head)
        self.linear = FilteredLinear(d, d)

    def forward(self, v, q):
        attn_outs = []
        for head in self.heads:
            attn_out = head(v, q)
            attn_outs.append(attn_out)
        attn_outs = torch.concat(attn_outs, dim=-1)
        return attn_outs

class Encoder(nn.Module):
    def __init__(self, n_head, n_layer, d, lv):
        super().__init__()
        self.n_head = n_head
        self.n_layer = n_layer
        self.d = d
        self.lv = lv
        self.attns = nn.ModuleList()
        self.head_dim = d // n_head
        for i in range(n_layer):
            attn = MultiHeadAttention(self.n_head, self.d, self.lv)
            self.attns.append(attn)

    def forward(self, v):
        for attn in self.attns:
            v = attn(v, v)
        return v


if __name__ == '__main__':
    len_q = 8
    len_v = 16
    d = 256
    batch = 3
    q = torch.rand([batch, len_q, d])
    v = torch.rand([batch, len_v, d])
    encoder = Encoder(4, 5, d, len_v)
    # mha = MultiHeadAttention(4, d, len_v)
    # x = mha(v, q)
    x = encoder(v)
    print(x.shape)
