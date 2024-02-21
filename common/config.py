# ----------FILES--------------------
coco_dir = '/Users/zx/Documents/ml/dataset/coco'
class_file = '/Users/zx/Documents/ml/dataset/coco/ms_coco_classnames.txt'

train_img_dir = f'{coco_dir}/train2017'
val_img_dir = f'{coco_dir}/val2017'

train_anno_dir = f'{coco_dir}/annotations/train_annotations'
val_anno_dir = f'{coco_dir}/annotations/val_annotations'

train_annotation_file = f'{coco_dir}/annotations/instances_train2017.json'
val_annotation_file = f'{coco_dir}/annotations/instances_val2017.json'

val_img_od_dict_file = f'{coco_dir}/annotations/val_img_od_dict.json'
train_img_od_dict_file = f'{coco_dir}/annotations/train_img_od_dict.json'

img_size = (320, 320)
patch_size = 8
grid_size_x = img_size[1] // patch_size
grid_size_y = img_size[0] // patch_size
