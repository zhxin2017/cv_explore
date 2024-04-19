from common import config
from detr import od_image, box, anno
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


class DetrProDS(Dataset):
    def __init__(self, img_dir, img_dict, sample_num=None, random_flip='random'):
        super().__init__()
        self.sample_num = sample_num
        self.img_dir = img_dir
        self.img_dict = img_dict
        self.random_flip = random_flip

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
        img = od_image.read_img_by_id(img_id, self.img_dir, channel_first=True)
        H, W = img.shape[1], img.shape[2]

        if max(H, W) > config.max_img_size:
            rescale = config.max_img_size / max(H, W)
            H = int(H * rescale)
            W = int(W * rescale)
        else:
            rescale = 1

        H_ = (H // config.patch_size) * config.patch_size
        W_ = (W // config.patch_size) * config.patch_size

        img = torchvision.transforms.Resize([H_, W_])(img)

        objs = self.img_dict[img_id]['objs']
        boxes, cids = [], []

        for o in objs:
            boxes.append(o['bbox'])
            cids.append(o['category_id'])

        H_rescale, W_rescale = H_ / H * rescale, W_ / W * rescale
        boxes = torch.tensor(boxes) * torch.tensor([[W_rescale, H_rescale, W_rescale, H_rescale]])
        boxes = box.xywh_to_xyxy(boxes)

        if (self.random_flip == 'random' and random.choice([True, False])) or self.random_flip == 'horizontal':
            img = torch.flip(img, [2])
            boxes = boxes * torch.tensor([[-1, 1, -1, 1]]) + torch.tensor([[W_, 0, W_, 0]])

        return img, boxes, cids, img_id

    def __len__(self):
        return self.sample_num


def collate_fn(batch):
    Ws = [b[0].shape[2] for b in batch]
    Hs = [b[0].shape[1] for b in batch]
    W_ = max(Ws)
    H_ = max(Hs)

    padded_imgs = []
    if len(batch) > 1:
        masks = []
    else:
        masks = None
    cids_batch = []
    boxes_batch = []
    img_ids_batch = []
    for img, boxes, cids, img_id in batch:
        C, H, W = img.shape
        img_ = torch.zeros(C, H_, W_)
        img_[:, :H, :W] = img
        padded_imgs.append(img_)
        if masks is not None:
            mask = torch.zeros(H_ // config.patch_size, W_ // config.patch_size)
            mask[:H // config.patch_size, :W // config.patch_size] = 1
            mask= mask.reshape(1, -1)
            masks.append(mask)
        cids_batch.append(cids)
        boxes_batch.append(boxes)
        img_ids_batch.append(img_id)

    padded_imgs = torch.stack(padded_imgs)
    if masks is not None:
        masks=torch.stack(masks)
    return padded_imgs, masks, boxes_batch, cids_batch, img_ids_batch






if __name__ == '__main__':
    from common.config import val_annotation_file, val_img_od_dict_file, val_img_dir

    dicts = anno.build_img_dict(val_annotation_file, val_img_od_dict_file, task='od')
    ds = DetrProDS(val_img_dir, dicts, sample_num=10, random_flip=True)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    for img, mask, boxes, cids, img_id in dl:
        print(img.shape)
        print(mask.shape)
        # print(boxes)
        print(cids)
        print(img_id)
        print('---------------')
