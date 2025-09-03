import os
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from detr import detr_seg_model, match_seg, eval
from config import img_root_dir, xml_root_dir, filelist_files, img_h, img_w, ds1, ds2, \
    batch_size, num_enc_layer, num_dec_layer, dmodel, dhead, num_query, epoch
from data.voc import VocDataset, collate_fn
import time
from data.classes import voc_classes
import numpy as np
from torch.utils.data import DataLoader

num_classes = len(voc_classes)

detr = detr_seg_model.DETR(dmodel, dhead, num_enc_layer, num_dec_layer, num_query, num_classes)

ckpt = 'detr_seg_epoch_1_batch_10500.pt'
detr.load_state_dict(torch.load(ckpt, map_location='cpu'))

device = torch.device('cpu')
detr.to(device)

val_filelist_files = ['/Users/zx/Documents/dataset/VOC2012/ImageSets/Main/val2017.txt']

dataset = VocDataset(img_root_dir, xml_root_dir, val_filelist_files, [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)

ds = ds1 * ds2
paint_ratio = 0.6
colors = np.random.randint(0, 255, size=(num_classes - 1, 3), dtype=np.uint8) * paint_ratio
colors = np.concatenate([np.zeros((1, 3), dtype=np.uint8), colors], axis=0)

if not os.path.exists('view'):
    os.makedirs('view')

for i, (imgs, targets) in enumerate(dataloader):
    imgs = imgs.to(device)
    n = len(targets)
    with torch.no_grad():
        seg_logits, cls_logits = detr(imgs)

    fm_h, fm_w = img_h // ds, img_w // ds
    seg = (seg_logits > 0).view(n, num_query, img_h, img_w) * 1.0
    # seg = F.interpolate(seg, size=(img_h, img_w), mode='nearest')
    seg = seg.cpu().numpy().astype(np.int8)

    cls = torch.argmax(cls_logits, dim=-1)
    cls = cls.cpu().numpy().astype(np.int8)

    for j in range(n):
        img = imgs[j].cpu().numpy() * 255
        for k in range(num_query):
            if cls[j, k] == 0:
                continue
            img_copy = img.copy()
            colors_ = colors[seg[j, k] * cls[j, k]]
            img_copy = img_copy * (1 - seg[j, k].reshape(img_h, img_w, 1) * paint_ratio)
            img_copy = img_copy + colors_
            img_copy = img_copy.astype(np.uint8)
            img_show = np.concatenate([img, img_copy], axis=1)
            cv2.imwrite(f'view/val_seg_{i}_{j}_{k}.jpg', img_show)
            print(f'view/val_seg_{i}_{j}_{k}.jpg saved')