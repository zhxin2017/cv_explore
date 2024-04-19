import os
import json


def build_img_dict(annotation_file, img_dict_file, task='od'):
    assert task in ['od', 'seg']

    if os.path.exists(img_dict_file):
        with open(img_dict_file, 'r') as f:
            img_dict = json.load(f)
        return img_dict
    with open(annotation_file, 'r') as f:
        instances = json.load(f)
    img_dict = {}
    for img in instances['images']:
        img_dict[img['id']] = {'shape': (img['height'], img['width']), 'objs': []}

    for obj in instances['annotations']:
        if task == 'detr':
            obj_dict = {'bbox': obj['bbox'], 'category_id': obj['category_id'], 'iscrowd': obj['iscrowd']}
        else:
            obj_dict = {'segmentation': obj['segmentation'], 'category_id': obj['category_id'],
                        'iscrowd': obj['iscrowd'], 'id': obj['id']}

        img_dict[obj['image_id']]['objs'].append(obj_dict)

    with open(img_dict_file, 'w') as f:
        f.write(json.dumps(img_dict))

    return img_dict


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from common.config import val_img_od_dict_file, train_img_od_dict_file, train_annotation_file, val_annotation_file

    build_img_dict(train_annotation_file, train_img_od_dict_file, task='detr')
    build_img_dict(val_annotation_file, val_img_od_dict_file, task='detr')
