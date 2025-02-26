import cv2
import torch
import numpy as np
import random

def load_image(image_path):
    return cv2.imread(image_path)


def pad(img, dst_h, dst_w, position='random'):
    h, w, c = img.shape
    if dst_h / h > dst_w / w: # dst img is slimmer, then resize w to dst_w and pad h with 0
        w_resize = dst_w
        h_resize = int(h * dst_w / w)
        img = cv2.resize(img, (w_resize, h_resize))
        if position == 'random':
            h_leftover = dst_h - h_resize
            h_pad_top = random.randint(0, h_leftover)
            print(h_pad_top)
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
            ... # to be specified
    else:
        w_resize = int(w * dst_h / h)
        h_resize = dst_h
        img = cv2.resize(img, (w_resize, h_resize))
        if position == 'random':
            w_leftover = dst_w - w_resize
            w_pad_left = random.randint(0, w_leftover)
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

if __name__ == '__main__':
    img_path = '/Users/zx/Documents/ml/dataset/coco/train2017/000000000034.jpg'
    img = load_image(img_path)
    print(img.shape)
    img_pad = pad(img, 640, 640)
    print(img_pad.shape)
    cv2.imshow('window_name', img_pad)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
