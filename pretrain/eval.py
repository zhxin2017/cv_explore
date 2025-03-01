from image import image_util
from common.config import img_size, patch_size
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from common.config import model_save_dir, device_type, model_save_stride, pretrain_batch_size


def eval(model, output_dir, epoch, batch):
    eval_dir = '/Users/zx/Documents/ml/dataset/coco/mini'
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    img_files = os.listdir(eval_dir)
    img_files = [file for file in img_files if file.endswith('.jpg')]
    img_paths = [os.path.join(eval_dir, img_file) for img_file in img_files]
    for i, img_path in enumerate(img_paths):
        img = image_util.load_image(img_path)
        img_padded = image_util.pad(img, *img_size, position='topleft')
        patches = image_util.to_patches(img_padded, patch_size).astype(np.float32)
        patch_h, patch_w = patches.shape[0], patches.shape[1]
        num_patches = patch_h * patch_w
        next_token_indices = [1, num_patches // 2, num_patches - 1]
        for j, next_token_idx in enumerate(next_token_indices):
            patches_ = torch.tensor([patches], device=torch.device(device_type))
            attn_mask = image_util.mask_patches(patches, next_token_idx)
            attn_mask = torch.tensor([attn_mask], device=torch.device(device_type))
            next_token_idx = torch.tensor([next_token_idx], device=torch.device(device_type), dtype=torch.int64)
            with torch.no_grad():
                pred_patch = model(patches_, attn_mask, next_token_idx)
            patches_[0, next_token_idx // patch_w, next_token_idx % patch_w] = pred_patch[0, 0]
            patches_ = patches_.view(patch_h, patch_w, patch_size, patch_size, 3)
            patches_ = patches_.permute(0, 2, 1, 3, 4).reshape(patch_h * patch_size, patch_w * patch_size, 3)
            patches_ = patches_.detach().cpu().numpy()
            patches_ = np.clip(patches_, 0, 1)
            patches_ = patches_ * 255
            patches_ = patches_.astype(np.uint8)
            axes[i][j].imshow(patches_)
            axes[i][j].axis('off')
        plt.savefig(f'{output_dir}/eval_e{epoch}_b{batch}.png', dpi=300)









