import cv2
from xml.etree import ElementTree as ET
import numpy as np
import random


def load_image(image_path):
    return cv2.imread(image_path) / 255

def parse_xml(xml_path, category_to_idx):
    with open(xml_path, 'r') as xml:
        data = xml.read()
    root = ET.XML(data)
    objs = root.findall('object')
    boxes = []
    cids = []
    for obj in objs:
        name = obj.find('name').text
        idx = category_to_idx[name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        cids.append(idx)
    boxes = np.array(boxes)
    cids = np.array(cids)
    return boxes, cids

def pad_img_and_boxes(img, boxes, dst_h, dst_w):
    '''
    Args:
        img: [H, W, C], np.ndarray
        boxes: [N, 4], np.ndarray, (x1, y1, x2, y2)
    '''
    h, w, _ = img.shape
    if h / dst_h > w / dst_w:
        scale = dst_h / h
        h_ = dst_h
        w_ = int(w * scale)
    else:
        scale = dst_w / w
        h_ = int(h * scale)
        w_ = dst_w
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    x_offset = random.randint(0, dst_w - w_)
    y_offset = random.randint(0, dst_h - h_)
    img = cv2.resize(img, (w_, h_))
    boxes = boxes * scale + np.array([[x_offset, y_offset, x_offset, y_offset]])
    canvas[y_offset:y_offset + h_, x_offset:x_offset + w_] = img
    return canvas, boxes




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
    img_path = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg'
    xml_path = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/Annotations/2007_000027.xml'
    img = load_image(img_path)
    from data.classes import voc_classes
    category_to_idx = {voc_classes[i]: i for i in range(len(voc_classes))}
    boxes, cids = parse_xml(xml_path, category_to_idx)
    print(img.shape)
    img_pad, boxes = pad_img_and_boxes(img, boxes, 288, 512)
    print(img_pad.shape)
    from data import visualize
    img_show = (img_pad * 255).astype(np.uint8)
    img_show = visualize.draw_bbox(img_show, boxes)
    cv2.imwrite('/Users/zx/Documents/ml/restart/test.png', img_show)
    # patch_size = 32
    # patches = to_patches(img_pad, patch_size)
    # print(patches.shape)    
    # nrow = 640 // patch_size
    # ncol = 640 // patch_size
    # fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10, 10))
    # for i in range(nrow):
    #     for j in range(ncol):
    #         axes[i][j].imshow(patches[i][j])
    #         axes[i][j].axis('off')
    # # plt.show()
    # plt.savefig('/Users/zx/Documents/ml/restart/common/image/test.png', bbox_inches='tight')