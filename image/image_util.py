import cv2
import numpy as np
import random


def load_image(image_path):
    return cv2.imread(image_path) / 255


def pad(img, dst_h, dst_w, position='random'):
    h, w, c = img.shape
    if dst_h / h > dst_w / w: # dst img is slimmer, then resize w to dst_w and pad h with 0
        w_resize = dst_w
        h_resize = int(h * dst_w / w)
        img = cv2.resize(img, (w_resize, h_resize))
        h_leftover = dst_h - h_resize
        if position == 'random':
            h_pad_top = random.randint(0, h_leftover)
        else:
            h_pad_top = 0
        h_pad_bottom = h_leftover - h_pad_top
        if h_pad_top > 0:
            padding_head = np.zeros([h_pad_top, dst_w, 3], dtype=img.dtype)
            img_padded = np.concatenate([padding_head, img], axis=0)
        else:
            img_padded = img
        if h_pad_bottom > 0:
            padding_tail = np.zeros([h_pad_bottom, dst_w, 3], dtype=img.dtype)
            img_padded = np.concatenate([img_padded, padding_tail], axis=0)
    else:
        w_resize = int(w * dst_h / h)
        h_resize = dst_h
        img = cv2.resize(img, (w_resize, h_resize))
        w_leftover = dst_w - w_resize
        if position == 'random':
            w_pad_left = random.randint(0, w_leftover)
        else:
            w_pad_left = 0
        w_pad_right = w_leftover - w_pad_left
        if w_pad_left > 0:
            padding_left = np.zeros([dst_h, w_pad_left, 3], dtype=img.dtype)
            img_padded = np.concatenate([padding_left, img], axis=1)
        else:
            img_padded = img
        if w_pad_right > 0:
            padding_right = np.zeros([dst_h, w_pad_right, 3], dtype=img.dtype)
            img_padded = np.concatenate([img_padded, padding_right], axis=1)
    return img_padded

def to_patches(img, patch_size):
    '''
    Args:
        img: [H, W, C], np.ndarray
        patch_size: int
    '''
    h, w, c = img.shape
    patch_h = h // patch_size
    patch_w = w // patch_size
    img = img.reshape(patch_h, patch_size, patch_w, patch_size, c).transpose(0, 2, 1, 3, 4)
    img = img.reshape(patch_h, patch_w, patch_size * patch_size * c)
    return img

def mask_patches(patches, next_token_idx):
    num_patches = patches.shape[0] * patches.shape[1]
    attn_mask = np.zeros([num_patches, num_patches], dtype=np.int32)
    attn_mask[:next_token_idx + 1, :next_token_idx + 1] = 1
    attn_mask[next_token_idx + 1:, next_token_idx + 1:] = 1
    return attn_mask

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # img_path = '/Users/zx/Documents/ml/dataset/coco/train2017/000000000034.jpg'
    img_path = '/Users/zx/Documents/ml/dataset/coco/train2017/000000000064.jpg'
    img = load_image(img_path)
    print(img.shape)
    img_pad = pad(img, 640, 640)
    print(img_pad.shape)
    patch_size = 64
    patches = to_patches(img_pad, patch_size)
    print(patches.shape)    
    nrow = 640 // patch_size
    ncol = 640 // patch_size
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, 10))
    for i in range(nrow):
        for j in range(ncol):
            axes[i][j].imshow(patches[i][j])
            axes[i][j].axis('off')
    # plt.show()
    plt.savefig('/Users/zx/Documents/ml/restart/common/image/test.png', bbox_inches='tight')