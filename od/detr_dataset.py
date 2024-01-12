from od import od_image, anno, box
import image
import torch
import random
from torch.utils.data import Dataset
import torchvision
from od.config_file import train_img_dir, val_img_dir


class OdDataset(Dataset):
    def __init__(self, img_dict, train=True, sample_num=None, resize=(384, 512), n_query=300, random_shift=True):
        self.sample_num = sample_num
        self.resize = resize
        self.random_shift = random_shift
        self.n_query = n_query
        self.img_dict = img_dict
        self.img_dir = train_img_dir if train else val_img_dir

        self.img_ids = []

        for iid, img_info in self.img_dict.items():
            iscrowd = False
            objs = img_info['objs']
            if len(objs) == 0:
                continue
            for obj_ in objs:
                if obj_['iscrowd'] == 1:
                    iscrowd = True
                    break
            if iscrowd:
                continue
            self.img_ids.append(iid)

        random.shuffle(self.img_ids)

        n_imgs = len(self.img_ids)

        if self.sample_num is None or n_imgs < self.sample_num:
            self.sample_num = n_imgs

    def __getitem__(self, item):
        img_id = self.img_ids[item]
        img = od_image.read_img_by_id(img_id, self.img_dir, channel_first=False)
        objs = self.img_dict[img_id]['objs']
        out_ratio = self.resize[0] / self.resize[1]
        img, offset_h, offset_w = image.pad_img(img, random_offset=self.random_shift, out_ratio=out_ratio)

        boxes, cids = [], []
        for o in objs:
            boxes.append(o['bbox'])
            cids.append(o['category_id'])

        boxes = torch.tensor(boxes)

        resize_factor = self.resize[0] / img.shape[0]
        boxes = box.offset_box(boxes, offset_h, offset_w)
        boxes = box.xywh_to_xyxy(boxes) * resize_factor

        bboxes_padded, indices_padded = box.pad_bbox(boxes, cids, self.n_query)

        img = torch.permute(img, [2, 0, 1])

        img = torchvision.transforms.Resize(self.resize)(img)

        return img, bboxes_padded, indices_padded, img_id

    def __len__(self):
        return self.sample_num


if __name__ == '__main__':
    from config_file import val_annotation_file, val_img_od_dict_file

    dicts = anno.build_img_dict(val_annotation_file, val_img_od_dict_file, task='od')
    ds = OdDataset(dicts, train=False, sample_num=10)
    for i in ds:
        print(i)
