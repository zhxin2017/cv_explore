from data import classes
import xml.etree.ElementTree as ET
import os
from data import preprocess as prep
from torch.utils.data import Dataset
import torch


class VocDataset(Dataset):
    def __init__(self, 
                 img_root_dir, 
                 ann_root_dir, 
                 filelist_files,
                 input_size):
        self.voc_name_to_idx = {name: idx for idx, name in enumerate(classes.voc_classes)}   
        self.h, self.w = input_size
        
        self.filelist = []
        for filelist_file in filelist_files:
            with open(filelist_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.filelist.append(line.strip())
        
        self.img_root_dir = img_root_dir
        self.anno_root_dir = ann_root_dir

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        file_name = self.filelist[index]
        img_path = f'{self.img_root_dir}/{file_name}.jpg'
        anno_path = f'{self.anno_root_dir}/{file_name}.xml'
        boxes, cids = prep.parse_xml(anno_path, self.voc_name_to_idx)
        img = prep.load_image(img_path)
        img, boxes = prep.pad_img_and_boxes(img, boxes, self.h, self.w)
        # num_box = len(boxes)

        return img, boxes, cids


def collate_fn(batch):
    images = [torch.tensor(item[0]) for item in batch]
    targets = [{'boxes': torch.tensor(item[1]), 'cids': torch.tensor(item[2])} for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


if __name__ == '__main__':
    img_root_dir = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/JPEGImages'
    xml_root_dir = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/Annotations'
    filelist_file = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    voc_dataset = VocDataset(img_root_dir, xml_root_dir, [filelist_file], (288, 512), 100)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(voc_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)
    for images, targets in dataloader:
        print(images.shape)
        print(targets[0]['boxes'].shape)
        print(targets[0]['cids'].shape)
        break