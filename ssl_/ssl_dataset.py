import torch
import torchvision
from torch.utils.data import Dataset
import os
from common.config import train_img_dir, val_img_dir, img_size
from common import image
import random
from matplotlib import pyplot as plt


def crop_img(img, crop_top_left=None):
    h, w, c = img.shape
    min_size = min(h, w)
    if crop_top_left is None:
        crop_top_left = random.choice([True, False])
    if crop_top_left:
        cropped = img[:min_size, :min_size]
    else:
        cropped = img[-min_size:, -min_size:]
    return cropped


class SslDataset(Dataset):
    def __init__(self, train=True, sample_num=None):
        super().__init__()
        self.img_dir = train_img_dir if train else val_img_dir
        self.file_names = os.listdir(self.img_dir)
        random.shuffle(self.file_names)
        self.resize = torchvision.transforms.Resize(img_size)
        n_imgs = len(self.file_names)
        self.sample_num = sample_num
        if self.sample_num is None or n_imgs < self.sample_num:
            self.sample_num = n_imgs

    def __getitem__(self, item):
        img_file_name = self.file_names[item]
        img_fp = f'{self.img_dir}/{img_file_name}'
        img = image.read_img(img_fp, channel_first=False)
        img = crop_img(img)
        img = torch.tensor(img)
        img = torch.permute(img, [2, 0, 1])
        img = self.resize(img)
        return img

    def __len__(self):
        return self.sample_num


if __name__ == '__main__':
    ds = SslDataset()
    for img in ds:
        img = torch.permute(img, [1, 2, 0])
        plt.imshow(img)
        plt.axis('off')
        print(img.shape)
        plt.pause(0)
        break
# patchified = patchified.view(grid_num_y, grid_num_x, 3, patch_size, patch_size)
# patchified = torch.permute(patchified, [0, 3, 1, 4, 2]).contiguous().view(grid_num_y * patch_size,
#                                                                           grid_num_x * patch_size, -1)
