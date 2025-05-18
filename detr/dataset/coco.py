
import json
import os
from dataset import image_util
from common.config import coco_dir


def load_anno(phase):
    if phase == 'train':
        annos_file = f'{coco_dir}/annotations/od_train_dict.json'
    elif phase == 'val':
        annos_file = f'{coco_dir}/annotations/od_val_dict.json'

    if os.path.exists(annos_file):
        with open(annos_file, 'r') as f:
            annos = json.load(f)
        return annos

    if phase == 'train':
        annotation_file = f'{coco_dir}/annotations/instances_train2017.json'
    elif phase == 'val':
        annotation_file = f'{coco_dir}/annotations/instances_val2017.json'

    with open(annotation_file, 'r') as f:
        instances = json.load(f)
    annos = {}
    for img in instances['images']:
        annos[img['id']] = {'shape': (img['height'], img['width']), 'objs': []}

    for obj in instances['annotations']:
        if annos.get(obj['image_id']) is None:
            continue
        if obj['iscrowd']:
            del annos[obj['image_id']]
            continue
        annos[obj['image_id']]['objs'].append(obj['bbox'] + [obj['category_id']])

    with open(annos_file, 'w') as f:
        f.write(json.dumps(annos))

    return annos

def img_id_to_name(img_id):
    img_id = str(img_id)
    digits = len(img_id)
    total_len = 12
    pad_len = total_len - digits
    padding = '0' * pad_len
    return f'{padding}{img_id}.jpg'


class CocoDataset:
    def __init__(self, phase):
        self.annos = load_anno(phase)
        if phase == 'train':
            img_dir = f'{coco_dir}/train2017'
        else:
            img_dir = f'{coco_dir}/val2017'
        self.img_dir = img_dir
        self.img_ids = list(self.annos.keys())
    
    def __len__(self):
        return len(self.img_ids)

    def load_sample_by_img_id(self, img_id):
        img_name = img_id_to_name(img_id)
        img_fp = f'{self.img_dir}/{img_name}'
        img = image_util.load_image(img_fp)
        boxes = self.annos[img_id]['objs']
        return img, boxes

    def get_sample(self, index):
        img_id = self.img_ids[index]
        return self.load_sample_by_img_id(img_id)

if __name__ == "__main__":
    coco_dataset = CocoDataset('train')
    for i in range(len(coco_dataset)):
        img, boxes = coco_dataset.get_sample(i)
        print(img.shape)
        for box in boxes:
            print(box)
        break
    ...

    