import os
import sys
sys.path.append('..')
import torch
from torch import optim
from ssl_ import ssl_model, ssl_dataset
from common.config import model_save_dir, device_type, model_save_stride, patch_size, num_grid, train_ssl_bsz
from common import image
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device(device_type)
model = ssl_model.SSL(d_cont=384, d_coord_emb=64, d_head=64, n_enc_layer=20)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

loss_fn = nn.MSELoss()


def train(epoch, population, batch_size, num_sample):
    ds = ssl_dataset.SslDataset(sample_num=population)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for i in range(epoch):
        for j, img in enumerate(dl):
            img = img.to(device)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

            pred = model(img)

            img_p = image.patchify(img, patch_size=patch_size, channel_first=True)
            img_p = torch.permute(img_p, [0, 1, 2, 4, 5, 3])
            img_p = img_p.contiguous().view(img.shape[0], num_grid, patch_size, patch_size, 3)

            loss = loss_fn(pred, img_p[:, 1:])
            optimizer.zero_grad()
            t = time.time()
            loss.backward()
            t_bp = time.time() - t
            # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
            optimizer.step()

            print(f'smp {num_sample}|epoch {i + 1}/{epoch}|batch {j}|'
                  f'loss {loss.detach().item():.3f}|tbp {t_bp:.3f}|')


if __name__ == '__main__':
    model_files = os.listdir(model_save_dir)
    model_files = [f for f in model_files if f.endswith('.pt') and f.startswith('ssl')]
    if len(model_files) == 0:
        model_path_old = None
        latest_version = 0
    else:
        versions = [int(f.split('.')[0].split('_')[-1]) for f in model_files]
        latest_version = max(versions)
        model_path_old = f'{model_save_dir}/ssl_{latest_version}.pt'
        saved_state = torch.load(model_path_old, map_location=device)
        model.load_state_dict(saved_state)
        # state = tsfm.state_dict()
        # for k in state.keys():
        #     if k in saved_state:
        #         state[k] = saved_state[k]
        # tsfm.load_state_dict(state)

    for i in range(500):
        n_smp = latest_version + 1 + i
        ts = time.time()
        train(1, batch_size=train_ssl_bsz, population=1000, num_sample=n_smp)
        te = time.time()
        print(f'----------------------used {te - ts:.3f} secs---------------------------')
        # train(1000, batch_size=1, population=2, num_sample=i)
        # train(2, batch_size=2, population=2, num_sample=i, weight_recover=.5, gamma=2)
        if n_smp % model_save_stride == 0:
            model_path_new = f'{model_save_dir}/ssl_{n_smp}.pt'
            torch.save(model.state_dict(), model_path_new)
            if model_path_old is not None:
                os.remove(model_path_old)
            model_path_old = model_path_new
        # break
