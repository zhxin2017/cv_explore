import cv2
import torch
import random


def pad(wider_len, shorter_len, buffer_ratio, out_ratio):
    """
    pad image to the shape of an output ratio
    :param wider_len: length of the wider side of image compared to the output ratio
    :param shorter_len:
    :param buffer_ratio:
    :param out_ratio: target output ratio. note this not same as out_ratio in pad_img()
    :return:
    """
    buffer_wider = int(wider_len * buffer_ratio)
    wider_len_new = buffer_wider + wider_len
    shorter_len_new = int(wider_len_new / out_ratio)
    buffer_shorter = shorter_len_new - shorter_len
    return wider_len_new, shorter_len_new, buffer_wider, buffer_shorter


def pad_img(img, random_offset=True, buffer_ratio=0, content='random', out_ratio=1.0, channel_first=False):
    if channel_first:
        c, h, w = img.shape
    else:
        h, w, c = img.shape
    if h / w > out_ratio:
        h_new, w_new, buffer_h, buffer_w = pad(h, w, buffer_ratio, out_ratio)
    else:
        w_new, h_new, buffer_w, buffer_h = pad(w, h, buffer_ratio, 1 / out_ratio)
    offset_h = random.randint(0, buffer_h) if random_offset else 0
    offset_w = random.randint(0, buffer_w) if random_offset else 0
    shape = [c, h_new, w_new] if channel_first else [h_new, w_new, c]
    if content == 'random':
        padded_img = torch.rand(shape)
    else:
        padded_img = torch.zeros(shape)

    if channel_first:
        padded_img[:, offset_h: offset_h + h, offset_w: offset_w + w] = img
    else:
        padded_img[offset_h: offset_h + h, offset_w: offset_w + w] = img

    return padded_img, offset_h, offset_w


# def crop_image(img, random_offset=True, out_ratio=1.0, channel_first=False):

def patchify(img, patch_size, channel_first=True, combine_patch=False):
    if channel_first:
        b, c, h, w = img.shape
    else:
        b, h, w, c = img.shape

    py, px = h // patch_size, w // patch_size

    if channel_first:
        patch_shape = (b, c, py, patch_size, px, patch_size)
        patch_dim = [0, 2, 4, 1, 3, 5]
    else:
        patch_shape = (b, py, patch_size, px, patch_size, c)
        patch_dim = [0, 1, 3, 2, 4, 5]
    img = img.view(*patch_shape)
    img = torch.permute(img, patch_dim)
    if combine_patch:
        img = img.contiguous().view(b, py, px, patch_size * patch_size * c)
    return img


def read_img(img_fp, channel_first):
    img = cv2.imread(img_fp)
    img = img / 256
    img = img[:, :, [2, 1, 0]]
    img = torch.tensor(img, dtype=torch.float32)
    if channel_first:
        img = torch.permute(img, [2, 0, 1])
    return img
