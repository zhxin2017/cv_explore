import torch

from detr import od_image
from common import image, config
import torchvision

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')
img_id = 632
img_dir = config.val_img_dir

# test padding image
'''
img = od_image.read_img_by_id(img_id, img_dir, channel_first=False)
img01, _, _ = od_image.pad_img(img, channel_first=False)
img10, _, _ = od_image.pad_img(img, channel_first=False, out_ratio=0.6)
img11, _, _ = od_image.pad_img(img, channel_first=False, out_ratio=1.5)
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(img)
axes[0, 0].axis('off')
axes[0, 1].imshow(img01)
axes[0, 1].axis('off')
axes[1, 0].imshow(img10)
axes[1, 0].axis('off')
axes[1, 1].imshow(img11)
axes[1, 1].axis('off')
plt.pause(0)
'''

# test patchify
img = od_image.read_img_by_id(img_id, img_dir)
img, _, _ = image.pad_img(img, channel_first=True)
c, h, w = img.shape
img = img.view([1, c, h, w])

img_size = 512
patch_size = 32
patch_num = img_size // patch_size
img = torchvision.transforms.Resize(img_size)(img)
img = image.patchify(img, patch_size)


img = img.view(patch_num, patch_num, c, patch_size, patch_size)
img = torch.permute(img, [0, 1, 3, 4, 2])
print(img.shape)
fig, axes = plt.subplots(patch_num, patch_num)
for i in range(patch_num):
    for j in range(patch_num):
        axes[i, j].axis('off')
        axes[i, j].imshow(img[i, j])
plt.pause(0)