import torch
import torch.nn.functional as F
from tsfm import transformer, base
from torch import nn
from common.config import patch_size


class NextTokenPredictor(nn.Module):
    def __init__(self, dmodel, dhead, nlayer):
        super().__init__()
        self.next_token_embedding_m = nn.Embedding(1, patch_size**2 * 3)
        self.encoder = transformer.Encoder(nlayer, dmodel, dhead, patch_size)
        self.linear1 = nn.Linear(dmodel, dmodel * 4)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(dmodel * 4, patch_size ** 2 * 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, next_token_mask):
        b, ph, pw, c = x.shape
        next_token_embedding = self.next_token_embedding_m(torch.zeros([b, ph, pw], dtype=torch.long, device=x.device))
        x = x * ~next_token_mask + next_token_embedding * next_token_mask
        x = self.encoder(x, mask=mask)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    x = torch.rand([4, 3, 512, 512])
    model = NextTokenPredictor(d_coord_emb=64, dhead=64, nlayer=20)
    pred = model(x)
    print(pred.shape)
