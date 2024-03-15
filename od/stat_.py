from od import anno, detr_dataset
import torch
from tqdm import tqdm
from common.config import train_annotation_file, train_img_od_dict_file
from od.config import cid_to_occurrence, category_num, n_query


def cal_cid_occurrence():
    cid_to_occurrence = {i: 0 for i in range(category_num)}
    dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')
    ds = detr_dataset.OdDataset(dicts, cid_only=True)
    for img, bboxes_padded, indices_padded, num_box, img_id in tqdm(ds):
        cid_to_occurrence[0] = cid_to_occurrence[0] + n_query - num_box
        for i in indices_padded[:num_box]:
            ind = i.item()
            cid_to_occurrence[ind] = cid_to_occurrence[ind] + 1
    return cid_to_occurrence

min_occr = 5000
# min_occr = min([occr for occr in cid_to_occurrence.values() if occr > 0])

cid_weights = [0] * category_num
for cid, occur in cid_to_occurrence.items():
    if occur == 0:
        continue
    weight = min_occr / occur
    if weight > 1:
        weight = 1
    cid_weights[cid] = weight
print(cid_weights)


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
