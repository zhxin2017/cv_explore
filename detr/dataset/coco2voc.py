import os
import json

from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree

coco_dir = '/Users/zx/Documents/dataset/coco'

coco_classes_file = 'detr/dataset/coco-labels-91.txt'
with open(coco_classes_file, 'r') as f:
    coco_classes = [line.strip() for line in f.readlines()]

def image_id_to_name(img_id):
    img_id = str(img_id)
    digits = len(img_id)
    total_len = 12
    pad_len = total_len - digits
    padding = '0' * pad_len
    return f'{padding}{img_id}.jpg'

def to_xml(boxes, cids, img_shape, img_name):
    root = Element('annotation')
    size = SubElement(root, 'size')
    width = SubElement(size, 'width')
    width.text = str(img_shape[1])
    height = SubElement(size, 'height')
    height.text = str(img_shape[0])
    
    for box, cid in zip(boxes, cids):
        obj = SubElement(root, 'object')
        name = SubElement(obj, 'name')
        name.text = coco_classes[cid - 1]  # coco classes are 1-indexed
        bndbox = SubElement(obj, 'bndbox')
        x1, y1, x2, y2 = box
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(int(x1))
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(int(y1))
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(int(x2))
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(int(y2))
    
    xml_str = tostring(root, encoding='unicode')
    return xml_str


def load_anno(phase):
    if phase == 'train':
        annotation_file = f'{coco_dir}/annotations/instances_train2017.json'
    elif phase == 'val':
        annotation_file = f'{coco_dir}/annotations/instances_val2017.json'

    with open(annotation_file, 'r') as f:
        instances = json.load(f)
    annos = {}
    for img in instances['images']:
        annos[img['id']] = {'shape': (img['height'], img['width']), 'objs': []}

    for obj in instances['annotations']:
        if annos.get(obj['image_id']) is None:
            continue
        if obj['iscrowd']:
            del annos[obj['image_id']]
            continue
        annos[obj['image_id']]['objs'].append(obj['bbox'] + [obj['category_id']])
    
    xml_dir = 'coco_xml_train' if phase == 'train' else 'coco_xml_val'
    if not os.path.exists(xml_dir):
        os.makedirs(xml_dir)

    for img_id, d in annos.items():
        image_name = image_id_to_name(img_id)
        shape = (d['shape'][0], d['shape'][1])
        objs = d['objs']
        bboxes = [box[:4] for box in objs]
        cids = [box[4] for box in objs]
        xml_str = to_xml(bboxes, cids, shape, image_name)
        xml_file = os.path.join(xml_dir, f'{image_name[:-4]}.xml')
        with open(xml_file, 'w') as f:
            f.write(xml_str)

# load_anno('train')
# load_anno('val')

root_folder = '/Users/zx/Documents/dataset/VOC2012'
anno_folder = os.path.join(root_folder, 'Annotations')

# files = os.listdir(os.path.join(anno_folder, 'train2017'))

# filelist_file = os.path.join(root_folder, 'ImageSets', 'Main', 'train2017.txt')

# with open(filelist_file, 'w') as f:
#     for file in files:
#         if file.endswith('.xml'):
#             f.write('train2017/' + file[:-4] + '\n')

files = os.listdir(os.path.join(anno_folder, 'val2017'))

filelist_file = os.path.join(root_folder, 'ImageSets', 'Main', 'val2017.txt')

with open(filelist_file, 'w') as f:
    for file in files:
        if file.endswith('.xml'):
            f.write('val2017/' + file[:-4] + '\n')