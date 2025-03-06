from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from common.config import img_size, patch_size
from image import image_util
import random


class PretrainDataset(Dataset):
    def __init__(self, img_dirs, random_padding=True):
        super().__init__()
        self.img_paths = []
        self.random_padding = random_padding
        for img_dir in img_dirs:
            for root, dirs, files in os.walk(img_dir):
                for file in files:
                    if not file.split('.')[-1].lower() in ['jpg', 'jpeg']:
                        continue
                    self.img_paths.append(f'{root}/{file}')

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        img = image_util.load_image(img_path)
        padding_position = 'random' if self.random_padding else 'topleft'
        img_padded = image_util.pad(img, *img_size, position=padding_position)
        patches = image_util.to_patches(img_padded, patch_size).astype(np.float32)
        num_patches = patches.shape[0] * patches.shape[1]
        next_token_idx = random.randint(1, num_patches - 1)
        next_token_mask = np.zeros([num_patches], dtype=np.int8)
        next_token_mask[next_token_idx] = 1
        next_token_mask = next_token_mask.reshape(patches.shape[0], patches.shape[1], 1)
        attn_mask = image_util.mask_patches(patches, next_token_idx)
        return patches, attn_mask, next_token_idx, next_token_mask

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    # imgdirs = ['/Users/zx/Documents/ml/dataset/coco/train2017', 'Users/zx/Documents/ml/dataset/VOCdevkit']
    imgdirs = ['Users/zx/Documents/ml/dataset/VOCdevkit']
    ds = PretrainDataset(imgdirs)
    print(len(ds))
