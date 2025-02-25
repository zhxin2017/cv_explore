from torch import nn
import torch
import torch.nn.functional as F
from tsfm import base, pe
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
        attn = attention(q, k, mask)

        v = v.view(b, lv, self.n_head, -1).transpose(1, 2)
        v = attn @ v
        v = v.transpose(1, 2).contiguous().view(b, lq, -1)
        return v


class Block(nn.Module):
    def __init__(self, dmodel, dhead):
        super().__init__()
        self.dmodel = dmodel
        self.dhead = dhead
        self.mha_ln = nn.LayerNorm(dmodel)
        self.mha = MHA(dmodel, dhead)

        self.ffn_ln = nn.LayerNorm(dmodel)
        self.ffn = base.FFN(dmodel)

    def forward(self, q, k, v, mask=None):
        q = self.mha_ln(q)
        k = self.mha_ln(k)
        v = self.mha_ln(v)

        q = q + self.mha(q, k, v, mask)
        q = q + self.ffn(self.ffn_ln(q))
        return q


class Encoder2D(nn.Module):
    def __init__(self, nlayer, dmodel, dhead, patch_size):
        super().__init__()

        self.cnn = nn.Conv2d(3, dmodel, (patch_size, patch_size),
                             stride=(patch_size, patch_size))
        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        self.pos_y_emb_m = nn.Embedding(max_grid_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, dmodel)

        self.n_enc_layer = nlayer
        self.enc_layers = nn.ModuleList()
        for i in range(nlayer):
            self.enc_layers.append(Block(dmodel, dhead))

    def forward(self, x, x_shift=0, y_shift=0, mask=None):
        x = F.relu(self.cnn(x))
        x = x.permute([0, 2, 3, 1])
        x = self.ln(x)

        b, h, w, c = x.shape
        num_grid = h * w
        y_indices = torch.arange(h, device=x.device) + y_shift
        x_indices = torch.arange(w, device=x.device) + x_shift
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, h, 1, self.dmodel).
                     repeat(b, 1, w, 1).view(b, num_grid, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, w, self.dmodel).
                     repeat(b, h, 1, 1).view(b, num_grid, self.dmodel))
        pos_emb = pos_y_emb + pos_x_emb
        x = x.view(b, num_grid, -1)

        x = x + pos_emb

        for enc_layer in self.enc_layers:
            x = enc_layer(x, x, x, mask)
        return x, pos_emb


class Encoder(nn.Module):
    def __init__(self, n_enc_layer, dmodel, dhead, patch_size, pretrain=False):
        super().__init__()

        self.cnn = nn.Conv2d(3, dmodel, (patch_size, patch_size),
                             stride=(patch_size, patch_size))
        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        self.pos_y_emb_m = nn.Embedding(max_grid_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, dmodel)

        self.n_enc_layer = n_enc_layer
        self.enc_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.enc_layers.append(Block(dmodel, dhead))
        self.pretrain = pretrain
        if pretrain:
            self.next_token_emb_m = pe.Embedding1D(1, dmodel)

    def forward(self, x, x_shift=0, y_shift=0, mask=None):
        x = F.relu(self.cnn(x))
        x = x.permute([0, 2, 3, 1])
        x = self.ln(x)

        b, h, w, c = x.shape
        num_grid = h * w
        y_indices = torch.arange(h, device=x.device) + y_shift
        x_indices = torch.arange(w, device=x.device) + x_shift
        pos_y_emb = (self.pos_y_emb_m(y_indices).view(1, h, 1, self.dmodel).
                     repeat(b, 1, w, 1).view(b, num_grid, self.dmodel))
        pos_x_emb = (self.pos_x_emb_m(x_indices).view(1, 1, w, self.dmodel).
                     repeat(b, h, 1, 1).view(b, num_grid, self.dmodel))
        pos_emb = self.ln(pos_y_emb + pos_x_emb)
        x = x.view(b, num_grid, -1)

        if self.pretrain:
            next_token_emb = self.ln(self.next_token_emb_m(x).repeat(1, num_grid, 1))
            x = torch.concat([x, next_token_emb], dim=1)
            pos_emb = torch.concat([pos_emb, pos_emb], dim=1)
            mask_11 = torch.tril(torch.ones([num_grid, num_grid], device=x.device))
            mask_12 = torch.diag(torch.ones([num_grid - 1], device=x.device), diagonal=1)
            mask_21 = torch.tril(torch.ones([num_grid, num_grid], device=x.device), diagonal=-1)
            mask_22 = torch.diag(torch.ones([num_grid], device=x.device))
            mask_1 = torch.concat([mask_11, mask_12], dim=-1)
            mask_2 = torch.concat([mask_21, mask_22], dim=-1)
            mask = torch.concat([mask_1, mask_2], dim=0)  # 2 * num_grid, 2 * num_grid
        else:
            mask = mask

        x = x + pos_emb

        for enc_layer in self.enc_layers:
            x = enc_layer(x, x, x, x, mask)
        return x, pos_emb
