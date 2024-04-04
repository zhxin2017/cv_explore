import torch
from common.config import img_size
import torch.nn.functional as F
from model import pe, tsfm, enc, base
from torch import nn
from common.config import grid_size_y, grid_size_x, num_grid, img_size, patch_size


class SSL(nn.Module):
    def __init__(self, d_cont, d_coord_emb, d_head, n_enc_layer):
        super().__init__()
        self.encoder = enc.Encoder(n_enc_layer, d_cont, d_head, d_coord_emb, pretrain=True)
        self.reg = base.MLP(d_cont, d_cont * 4, 3 * patch_size**2, 2)

    def forward(self, x):
        x = self.encoder(x)
        pred = F.sigmoid(self.reg(x[:, num_grid + 1:]))
        pred = pred.view(-1, patch_size**2, 3)
        return pred

