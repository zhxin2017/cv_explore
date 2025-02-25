import torch
from torch import nn
from common.config import grid_size_x, grid_size_y, num_grid


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


def gen_pos_2d(x, pos='center'):
    assert pos in ['center', 'x1y1', 'x2y2']
    max_size = max(grid_size_y, grid_size_x)
    row_indices = torch.arange(grid_size_y, device=x.device).unsqueeze(1).repeat(1, grid_size_x).unsqueeze(-1)
    col_indices = torch.arange(grid_size_x, device=x.device).unsqueeze(0).repeat(grid_size_y, 1).unsqueeze(-1)
    if pos == 'center':
        positions = torch.concat(((col_indices + 0.5) / max_size, (row_indices + 0.5) / max_size), dim=-1)
    elif pos == 'x1y1':
        positions = torch.concat((col_indices / max_size, row_indices / max_size), dim=-1)
    else:
        positions = torch.concat(((col_indices + 1) / max_size, (row_indices + 1) / max_size), dim=-1)
    positions = positions.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
    return positions


def gen_pos_indices(device):
    row_indices = torch.arange(grid_size_y, device=device).unsqueeze(1).repeat(1, grid_size_x).unsqueeze(-1)
    col_indices = torch.arange(grid_size_x, device=device).unsqueeze(0).repeat(grid_size_y, 1).unsqueeze(-1)
    pos_indices = torch.concat((col_indices, row_indices), dim=-1)
    pos_indices = pos_indices.view(num_grid, 2)
    return pos_indices


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
        emb_ = torch.clone(emb)
        emb_[..., sin_indices] = torch.sin(emb[..., sin_indices] / self.temperature ** (sin_indices / self.d))
        emb_[..., cos_indices] = torch.cos(emb[..., cos_indices] / self.temperature ** ((cos_indices - 1) / self.d))
        if self.norm:
            emb_ = self.ln(emb_)
        return emb_.view(bsz, seq_len, pos_dim * self.d)


if __name__ == '__main__':
    from detr import anchor
    import random
    from matplotlib import pyplot as plt

    anchors = anchor.generate_anchors()
    ln = nn.LayerNorm(64)

    fig, axes = plt.subplots(1, 7, figsize=(35, 5))
    random.shuffle(anchors)
    anchor = anchors[0]

    img = torch.ones([128, 128, 3]) * 0.6
    axes[0].imshow(img)
    x1, y1, x2, y2 = anchors[0]
    x1 = x1 * 128
    y1 = y1 * 128
    x2 = x2 * 128
    y2 = y2 * 128
    axes[0].add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', lw=1))

    pos_emb = Sinusoidal(64)
    x = torch.rand([1, grid_size_y, grid_size_x, 2])
    coord = gen_pos_2d(x).view(1, num_grid, 2)
    pos_emb_ = pos_emb(coord).view(num_grid, 128)

    anchor = torch.tensor(anchor).view(1, 1, 4)
    anchor_emb = pos_emb(anchor).view(1, 256)

    x1_attn = (anchor_emb[:, :64] @ pos_emb_[:, :64].transpose(0, 1)).softmax(dim=-1).view(grid_size_y, grid_size_x)
    axes[1].imshow(x1_attn.detach().numpy())

    y1_attn = (anchor_emb[:, 64:128] @ pos_emb_[:, 64:].transpose(0, 1)).softmax(dim=-1).view(grid_size_y, grid_size_x)
    axes[2].imshow(y1_attn.detach().numpy())

    x2_attn = (anchor_emb[:, 128:192] @ pos_emb_[:, :64].transpose(0, 1)).softmax(dim=-1).view(grid_size_y, grid_size_x)
    axes[3].imshow(x2_attn.detach().numpy())

    y2_attn = (anchor_emb[:, 192:] @ pos_emb_[:, 64:].transpose(0, 1)).softmax(dim=-1).view(grid_size_y, grid_size_x)
    axes[4].imshow(y2_attn.detach().numpy())

    axes[5].imshow((x1_attn + y1_attn + x2_attn + y2_attn).detach().numpy())

    x_middle = ln(anchor_emb[:, 128:192] * .8 + anchor_emb[:, :64] * .2)
    print(x_middle.std())
    x1, y1, x2, y2 = anchor[0, 0] * grid_size_x
    print(x1, y1, x2, y2)
    print(x2 * .8 + x1 * .2)
    x_middle_attn = ((x_middle) @ pos_emb_[:, :64].transpose(0, 1)).softmax(dim=-1).view(grid_size_y, grid_size_x)
    axes[6].imshow(x_middle_attn.detach().numpy())
    plt.pause(0)
