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
        img_dict[img['id']] = {'file_name': img['file_name'], 'shape': (img['height'], img['width']), 'objs': []}

    for obj in instances['annotations']:
        if task == 'od':
            obj_dict = {'bbox': obj['bbox'], 'category_id': obj['category_id']}
        else:
            obj_dict = {'segmentation': obj['segmentation'], 'category_id': obj['category_id'], 'id': obj['id']}

        img_dict[obj['image_id']]['objs'].append(obj_dict)

    with open(img_dict_file, 'w') as f:
        f.write(json.dumps(img_dict))

    return img_dict
