import torch
import torch.nn.functional as F
from model import enc, base
from torch import nn
from common.config import num_grid, patch_size


class SSL(nn.Module):
    def __init__(self, d_cont, d_coord_emb, d_head, n_enc_layer):
        super().__init__()
        self.encoder = enc.Encoder(n_enc_layer, d_cont, d_head, d_coord_emb, pretrain=True)
        self.ln = nn.LayerNorm(d_cont)
        self.reg = base.MLP(d_cont, d_cont * 4, 3 * patch_size ** 2, 2)

    def forward(self, x):
        x, _ = self.encoder(x)
        pred = F.sigmoid(self.reg(self.ln(x[:, num_grid + 1:])))
        pred = pred.view(x.shape[0], num_grid - 1, patch_size, patch_size, 3)
        return pred


if __name__ == '__main__':
    x = torch.rand([4, 3, 512, 512])
    model = SSL(d_cont=384, d_coord_emb=64, d_head=64, n_enc_layer=20)
    pred = model(x)
    print(pred.shape)
