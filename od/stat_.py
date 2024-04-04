from od import anno, detr_dataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from common.config import train_annotation_file, train_img_od_dict_file
from od.config import cid_to_occurrence, n_cls
from od import anchor

n_query = len(anchor.generate_anchors())



def cal_cid_occurrence():
    cid_to_occurrence = {i: 0 for i in range(n_cls)}
    dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')
    ds = detr_dataset.OdDataset(dicts, n_query, cid_only=True)
    for img, bboxes_padded, indices_padded, num_box, img_id in tqdm(ds):
        cid_to_occurrence[0] = cid_to_occurrence[0] + n_query - num_box
        for i in indices_padded[:num_box]:
            ind = i.item()
            cid_to_occurrence[ind] = cid_to_occurrence[ind] + 1
    return cid_to_occurrence

# print(cal_cid_occurrence())
def cal_weights(cid_to_occurrence):

    min_occr = 5000
    # min_occr = min([occr for occr in cid_to_occurrence.values() if occr > 0])

    cid_weights = [0] * n_cls
    for cid, occur in cid_to_occurrence.items():
        if occur == 0:
            continue
        weight = min_occr / occur
        if weight > 1:
            weight = 1
        cid_weights[cid] = weight
    print(cid_weights)

# cal_weights(cal_cid_occurrence())
'''
imgs = set()
population = 1000
dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')
for i in range(500):
    print(f'{i} / 500')
    ds = detr_dataset.OdDataset(dicts, train=True, sample_num=population, random_shift=False, cid_only=True)
    dl = DataLoader(ds, batch_size=2, shuffle=False)
    print(len(ds))
    for img, boxes_gt_xyxy, cids_gt, _, img_id in dl:
        imgs.update(set(img_id))
    print(len(imgs)) # 107145
'''
# dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')
# ds = detr_dataset.OdDataset(dicts, train=True, random_shift=False, cid_only=True)
# print(len(ds))

# obj_cnt = 0
# for i, objs in dicts.items():
#     obj_cnt += len(objs['objs'])
# print(obj_cnt)

# obj_num = sum(cid_to_occurrence.values())
#
# img_num = len(dicts)
#
# total = img_num * n_query
# no_obj_num = total - obj_num
# cid_to_occurrence[0] = no_obj_num
#
# loss_weights = [(i, total / c) for i, c in cid_to_occurrence.items()]
# max_weight = max([w for i, w in loss_weights])
# loss_weights = [(i, (w / max_weight)) for i, w in loss_weights]
# cids = [i for i, w in loss_weights]
# for i in range(category_num):
#     if i not in cids:
#         loss_weights.append((i, 1))
# loss_weights.sort(key=lambda x: x[0])
# loss_weights = [w for i, w in loss_weights]
#
# occurrence = [(i, c) for i, c in cid_to_occurrence.items()]
# occurrence.sort(key=lambda x: x[0])
# occurrence = torch.tensor([c for i, c in occurrence])
#
#
#
# a = torch.arange(12).view(12, 1)
# b = torch.arange(24, 36).view(12, 1)
# c = torch.concat((a, b), dim=-1)
# print(c)

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')
ds = detr_dataset.OdDataset(dicts, n_query, cid_only=True)
max_num = 0
for img, bboxes_padded, indices_padded, num_box, img_id in ds:
    cid_to_count = {}
    for i in range(num_box):
        cid = indices_padded[i].item()
        cid_to_count.setdefault(cid, []).append(cid)
        count = len(cid_to_count[cid])
        if count > max_num:
            print(count)
            print(img_id)
            max_num = count
