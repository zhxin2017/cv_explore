# ----------FILES--------------------
coco_dir = '/content/coco'
class_file = f'{coco_dir}/ms_coco_classnames.txt'

train_img_dir = f'{coco_dir}/train2017'
val_img_dir = f'{coco_dir}/val2017'

anno_dir = f'{coco_dir}/annotations'
train_anno_dir = f'{anno_dir}/train_annotations'
val_anno_dir = f'{anno_dir}/val_annotations'

train_annotation_file = f'{anno_dir}/instances_train2017.json'
val_annotation_file = f'{anno_dir}/instances_val2017.json'

val_img_od_dict_file = f'{anno_dir}/val_img_od_dict.json'
train_img_od_dict_file = f'{anno_dir}/train_img_od_dict.json'

img_size = (512, 512)
max_img_size = max(img_size)
patch_size = 16
max_grid_y = img_size[0] // patch_size
max_grid_x = img_size[1] // patch_size
grid_size_x = img_size[1] // patch_size
grid_size_y = img_size[0] // patch_size

num_grid = grid_size_y * grid_size_x


# train
model_save_dir = '/content/drive/MyDrive/od_model'
model_save_stride = 5
# device_type = 'cpu'
device_type = 'cuda'

train_base_bsz = 1
train_pro_bsz = 1
train_ssl_bsz = 1
train_anchor_bsz = 2
