from od import anno
import torch
from config import train_annotation_file, train_img_od_dict_file, n_query
from config_category import cid_to_occurrence, category_num

dicts = anno.build_img_dict(train_annotation_file, train_img_od_dict_file, task='od')

# obj_cnt = 0
# for i, objs in dicts.items():
#     obj_cnt += len(objs['objs'])
# print(obj_cnt)

obj_num = sum(cid_to_occurrence.values())

img_num = len(dicts)

total = img_num * n_query
no_obj_num = total - obj_num
cid_to_occurrence[0] = no_obj_num

loss_weights = [(i, total / c) for i, c in cid_to_occurrence.items()]
max_weight = max([w for i, w in loss_weights])
loss_weights = [(i, (w / max_weight)) for i, w in loss_weights]
cids = [i for i, w in loss_weights]
for i in range(category_num):
    if i not in cids:
        loss_weights.append((i, 1))
loss_weights.sort(key=lambda x: x[0])
loss_weights = [w for i, w in loss_weights]

occurrence = [(i, c) for i, c in cid_to_occurrence.items()]
occurrence.sort(key=lambda x: x[0])
occurrence = torch.tensor([c for i, c in occurrence])



a = torch.arange(12).view(12, 1)
b = torch.arange(24, 36).view(12, 1)
c = torch.concat((a, b), dim=-1)
print(c)
