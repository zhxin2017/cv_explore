import torch
from torch import nn
import torch.nn.functional as F
from model import pe, tsfm
from common.config import num_grid


class Encoder(nn.Module):
    def __init__(self, n_enc_layer, d_cont, d_head, d_coord_emb, pretrain=False):
        super().__init__()

        self.cnn1 = nn.Conv2d(3, 1024, (4, 4), stride=(4, 4))
        self.cnn2 = nn.Conv2d(1024, d_cont, (4, 4), stride=(4, 4))
        self.cnn_ln = nn.LayerNorm(d_cont)

        dq = d_cont + 2 * d_coord_emb
        n_head = dq // d_head

        self.pos_emb_m = pe.Sinusoidal(d_coord_emb)

        self.n_enc_layer = n_enc_layer
        self.attn_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.attn_layers.append(tsfm.AttnLayer(dq, dq, d_cont, n_head))

        self.pretrain = pretrain
        if pretrain:
            self.next_token_emb_m = pe.Embedding1D(1, d_cont)

    def forward(self, x, mask=None):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = x.permute([0, 2, 3, 1])
        x = self.cnn_ln(x)

        bsz = x.shape[0]
        pos = pe.gen_pos_2d(x, pos='center').view(bsz, num_grid, 2)
        pos_emb = self.pos_emb_m(pos)
        x = x.view(bsz, num_grid, -1)

        if self.pretrain:
            next_token_emb = self.next_token_emb_m(x).repeat(1, num_grid, 1)
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

        for i in range(self.n_enc_layer):
            q = torch.concat([x, pos_emb], dim=-1)
            x = self.attn_layers[i](q, q, x, q, mask)
        return x, pos_emb
