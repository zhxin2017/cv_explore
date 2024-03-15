import torch
from torch import nn
from common.config import img_size


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
    def __init__(self, N, num_feat, uniform=False):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(N, num_feat)
        if uniform:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        B = x.shape[0]
        indices = torch.arange(self.N, device=x.device)
        emb = self.embed(indices)
        return emb.unsqueeze(0).repeat(B, 1, 1)


def gen_pos_2d(x):
    B, H, W, C = x.shape
    max_size = max(H, W)
    row_indices = torch.arange(H, device=x.device).unsqueeze(1).repeat(1, W).unsqueeze(-1)
    col_indices = torch.arange(W, device=x.device).unsqueeze(0).repeat(H, 1).unsqueeze(-1)
    positions = torch.concat(((col_indices + 0.5) / max_size, (row_indices + 0.5) / max_size), dim=-1)
    positions = positions.unsqueeze(0).repeat(B, 1, 1, 1)
    return positions


class Sinusoidal(nn.Module):
    def __init__(self, d, temperature=0.1, norm=True):
        super().__init__()
        self.d = d
        self.temperature = temperature
        self.norm = norm
        self.ln = nn.LayerNorm(d)

    def forward(self, pos):
        half_d = self.d // 2
        half_range = list(range(half_d))
        sin_indices = torch.tensor([2 * i for i in half_range], device=pos.device)
        cos_indices = torch.tensor([2 * i + 1 for i in half_range], device=pos.device)
        bsz, seq_len, pos_dim = pos.shape
        emb = pos.view(bsz, seq_len, pos_dim, 1).expand(bsz, seq_len, pos_dim, self.d)
        emb[..., sin_indices] = torch.sin(emb[..., sin_indices] / self.temperature ** (sin_indices / self.d))
        emb[..., cos_indices] = torch.cos(emb[..., cos_indices] / self.temperature ** ((cos_indices - 1) / self.d))
        if self.norm:
            emb = self.ln(emb)
        return emb.view(bsz, seq_len, pos_dim * self.d)




if __name__ == '__main__':
    from od import anchor
    anchors = anchor.generate_anchors()
    anchors = torch.tensor(anchors).unsqueeze(0)
    # emb = sinusoidal_encoding(anchors, 64)
    pass