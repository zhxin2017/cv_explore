from detr import od_image, anno, box
from common import image
import torch
import random
from torch.utils.data import Dataset
import torchvision
from common.config import train_img_dir, val_img_dir, img_size


def get_gt_by_img_id(img_id, img_dict, img_dir, resize, random_shift, n_query, cid_only=False):
    objs = img_dict[img_id]['objs']
    out_ratio = resize[0] / resize[1]
    if cid_only:
        img = 0
        offset_h, offset_w = 0, 0
        resize_factor = 1
    else:
        img = od_image.read_img_by_id(img_id, img_dir, channel_first=False)
        img, offset_h, offset_w = image.pad_img(img, random_offset=random_shift, content='zero', out_ratio=out_ratio)
        resize_factor = resize[0] / img.shape[0]
        img = torch.permute(img, [2, 0, 1])
        img = torchvision.transforms.Resize(resize)(img)

    boxes, cids = [], []
    for o in objs:
        boxes.append(o['bbox'])
        cids.append(o['category_id'])

    boxes = torch.tensor(boxes)
    boxes = box.offset_box(boxes, offset_h, offset_w)
    boxes = box.xywh_to_xyxy(boxes) * resize_factor
    num_boxes = len(boxes)

    bboxes_padded, indices_padded = box.pad_bbox(boxes, cids, n_query)

    return img, bboxes_padded, indices_padded, num_boxes, img_id


class OdDataset(Dataset):
    def __init__(self, img_dict, n_query, train=True, sample_num=None, resize=img_size, random_shift=False, cid_only=False):
        self.sample_num = sample_num
        self.resize = resize
        self.random_shift = random_shift
        self.n_query = n_query
        self.img_dict = img_dict
        self.img_dir = train_img_dir if train else val_img_dir
        self.img_ids = []
        self.cid_only = cid_only

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
        return get_gt_by_img_id(img_id, self.img_dict, self.img_dir, self.resize, self.random_shift, self.n_query, self.cid_only)

    def __len__(self):
        return self.sample_num


if __name__ == '__main__':
    from common.config import val_annotation_file, val_img_od_dict_file

    dicts = anno.build_img_dict(val_annotation_file, val_img_od_dict_file, task='detr')
    ds = OdDataset(dicts, train=False, sample_num=10)
    for i in ds:
        print(i)
