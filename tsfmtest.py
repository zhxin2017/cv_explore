import time
from tsfm import selective_tsfm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.datasets import MNIST


mps_device = torch.device("mps")
train_mnist = MNIST(root="mnist", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_mnist = MNIST(root="mnist", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))


class Cls4(nn.Module):
    def __init__(self, lv, in_dim, dim=128):
        super().__init__()
        self.lv = lv
        self.dim = dim
        self.proj = nn.Linear(in_dim, dim)
        self.tsfm = selective_tsfm.Encoder(4, 4, dim, lv + 1)
        self.pe_em = nn.Embedding(lv, dim)
        self.query_em = nn.Embedding(1, dim)
        self.cls_linear = nn.Linear(dim, 10)

    def forward(self, x):
        b, h, w = x.shape
        x = self.proj(x)
        pe = self.pe_em(torch.arange(self.lv))
        x = x + pe.view(1, self.lv, self.dim)
        query = self.query_em(torch.zeros(b, 1, dtype=torch.int))
        x = torch.concat([x, query], dim=1)
        x = self.tsfm(x)
        x = self.cls_linear(x[:, -1])
        return x


model = Cls4(28, 28, 128)
# tsfm.to(mps_device)

train_dl = DataLoader(train_mnist, batch_size=4)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
closs_fun = nn.CrossEntropyLoss(reduction='none')


losses = [100] * 20
accus = [0] * 20

s = time.time()
ok = False

for i in range(100):
    cnt = 0
    if ok:
        break
    for img, cls in train_dl:
        cnt += 1
        img = torch.permute(img, [0, 2, 3, 1])
        b, h, w, c = img.shape
        logits = model(img.view(b, h, w))
        loss = closs_fun(logits, cls)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.argmax(logits, dim=-1)
        tp = (pred == cls).sum()
        accu = tp / b
        losses.append(loss.item())
        losses.pop(0)
        accus.append(accu)
        accus.pop(0)
        if sum(accus) / 20 >= 0.975 and sum(losses) / 20 < 0.35:
            e = time.time()
            print('used', e - s)
            ok = True
            break
        print(f'|{i}|#{cnt}|loss: {loss.item():.4f}| accuracy {accu}')
        # if cnt == 100:
        #     torch.save(tsfm.state_dict(), 'models/mystfm_mnist.pt')

