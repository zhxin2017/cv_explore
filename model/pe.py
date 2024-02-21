import torch
from torch import nn
from common.config import img_sz


class Embedding2D(nn.Module):
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


class Embedding1D(nn.Module):
    def __init__(self, N, num_feat, device=torch.device('mps')):
        super().__init__()
        self.device = device
        self.N = N
        self.embed = nn.Embedding(N, num_feat)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, B):
        indices = torch.arange(self.N, device=self.device)
        emb = self.embed(indices)
        return emb.unsqueeze(0).repeat(B, 1, 1)


def gen_pos_2d(x, device=torch.device('mps'), feature_map_size=max(img_sz) // 8):
    B, H, W, C = x.shape
    row_indices = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).unsqueeze(-1)
    col_indices = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).unsqueeze(-1)
    positions = torch.concat(((col_indices + 0.5) / feature_map_size, (row_indices + 0.5) / feature_map_size), dim=-1)
    positions = positions.unsqueeze(0).repeat(B, 1, 1, 1)
    return positions

