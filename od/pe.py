import torch
from torch import nn


class PositionEmbedding2D(nn.Module):
    def __init__(self, H, W, num_feat, device=torch.device('mps')):
        super().__init__()
        self.device = device
        self.H = H
        self.W = W
        self.row_embed = nn.Embedding(H, num_feat)
        self.col_embed = nn.Embedding(W, num_feat)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, B, H_range=None, W_range=None):
        x = torch.arange(self.W, device=self.device)
        y = torch.arange(self.H, device=self.device)
        x_emb = self.col_embed(x)
        y_emb = self.row_embed(y)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(self.H, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, self.W, 1),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .repeat(B, 1, 1, 1)
        )
        return pos
