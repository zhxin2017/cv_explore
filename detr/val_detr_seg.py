import os
import cv2
import torch
import torch.nn.functional as F
from torch import optim
from detr import detr_seg_model, match_seg, eval
from config import img_root_dir, xml_root_dir, filelist_files, img_h, img_w, downsample_seg, \
    batch_size, num_enc_layer, num_dec_layer, dmodel, dhead, num_query, epoch
from data.voc import VocDataset, collate_fn
import time
from data.classes import voc_classes
import numpy as np
from torch.utils.data import DataLoader

num_classes = len(voc_classes)

detr = detr_seg_model.DETR(dmodel, dhead, num_enc_layer, num_dec_layer, num_query, num_classes)

ckpt = 'detr_seg_epoch_1_batch_6000.pt'
detr.load_state_dict(torch.load(ckpt, map_location='cpu'))

device = torch.device('cpu')
detr.to(device)

val_filelist_files = ['/Users/zx/Documents/dataset/VOC2012/ImageSets/Main/val2017.txt']

dataset = VocDataset(img_root_dir, xml_root_dir, val_filelist_files, [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)

paint_ratio = 0.4
colors = np.random.randint(0, 255, size=(num_classes - 1, 3), dtype=np.uint8) * paint_ratio
colors = np.concatenate([np.zeros((1, 3), dtype=np.uint8), colors], axis=0)

if not os.path.exists('view'):
    os.makedirs('view')

for i, (imgs, targets) in enumerate(dataloader):
    imgs = imgs.to(device)
    n = len(targets)
    with torch.no_grad():
        seg_logits = detr(imgs)

    seg_cls = torch.argmax(seg_logits, dim=-1)
    seg_cls = F.interpolate(seg_cls * 1.0, size=(img_h, img_w), mode='nearest')

    for j in range(n):
        img = imgs[j].cpu().numpy() * 255
        seg_cls_ = seg_cls[j]
        seg_pos_mask = (seg_cls_ > 0).view(num_query, img_h, img_w, 1) * 1
        seg_query_pos_mask = torch.sum(seg_pos_mask, dim=(1, 2)) > 0
        seg_cls_ = seg_cls_.cpu().numpy().astype(np.int32)
        seg_pos_mask = seg_pos_mask.cpu().numpy()
        for k in range(num_query):
            if seg_query_pos_mask[k]:
                img_copy = img.copy()
                colors_ = colors[seg_cls_[k]]
                img_copy = img_copy * (1 - seg_pos_mask[k] * paint_ratio)
                img_copy = img_copy + colors_
                img_copy = img_copy.astype(np.uint8)
                img_show = np.concatenate([img, img_copy], axis=1)
                cv2.imwrite(f'view/val_seg_{i}_{j}_{k}.jpg', img_show)
                print(f'view/val_seg_{i}_{j}_{k}.jpg saved')