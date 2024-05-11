import torch
from torch import nn
import torch.nn.functional as F
from model import pe, tsfm
from common.config import max_grid_h, max_grid_w, patch_size


class Encoder(nn.Module):
    def __init__(self, n_enc_layer, dmodel, dhead, pretrain=False):
        super().__init__()

        self.cnn = nn.Conv2d(3, dmodel, (patch_size, patch_size),
                             stride=(patch_size, patch_size))
        self.ln = nn.LayerNorm(dmodel)
        self.dmodel = dmodel

        n_head = dmodel // dhead

        self.pos_y_emb_m = nn.Embedding(max_grid_h, dmodel)
        self.pos_x_emb_m = nn.Embedding(max_grid_w, dmodel)

        self.n_enc_layer = n_enc_layer
        self.enc_layers = nn.ModuleList()
        for i in range(n_enc_layer):
            self.enc_layers.append(tsfm.AttnLayer(dmodel, dmodel, dmodel, n_head))
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

