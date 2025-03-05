import torch
from common.config import device_type, patch_size
from image import image_util
from common.config import img_size, patch_size
import random
import numpy as np
import matplotlib.pyplot as plt
from tsfm import transformer

device = torch.device(device_type)
encoder = transformer.Encoder(nlayer=18, dmodel=384, dhead=64, patch_size=patch_size).to(device)

model_file = 'outputs/pretrain/pretrain_e4_i3600.pt'
saved_state = torch.load(model_file, map_location=device)
encoder_weights = encoder.state_dict()
for k, v in saved_state.items():
    if k.startswith('encoder.'):
        encoder_weights[k[len('encoder.'):]] = v
encoder.load_state_dict(encoder_weights)
encoder.eval()

tesst_img_path = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/JPEGImages/2007_004713.jpg'

img = image_util.load_image(tesst_img_path)
img_padded = image_util.pad(img, *img_size, position='topleft')
patches = image_util.to_patches(img_padded, patch_size).astype(np.float32)
patches = torch.tensor([patches]).to(device)
x, _ = encoder(patches)
print(x.shape)
ln = torch.nn.LayerNorm(x.shape[-1]).to(device)
x = ln(x)
attn = transformer.attention(x, x)
print(attn.shape)

q_idx = random.randint(0, x.shape[1] - 1)
q_h, q_w = q_idx // patches.shape[2], q_idx % patches.shape[2]
print(q_idx)
attn_map = attn[0, q_idx].detach().cpu().numpy().reshape(patches.shape[1], patches.shape[2])
print(attn_map.shape)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
img_padded[q_h * patch_size:(q_h + 1) * patch_size, q_w * patch_size:(q_w + 1) * patch_size] = np.array([1, 0, 0])
axes[0].imshow(img_padded)
axes[1].imshow(attn_map)
plt.show()

# plt.imshow(attn[0].detach().cpu().numpy())
# plt.show()