from common import image
import xml.etree.ElementTree as ET


def img_id_to_name(img_id):
    img_id = str(img_id)
    digits = len(img_id)
    total_len = 12
    pad_len = total_len - digits
    padding = '0' * pad_len
    return f'{padding}{img_id}.jpg'


def read_img_by_id(img_id, img_dir, channel_first=True):
    img_name = img_id_to_name(img_id)
    img_fp = f'{img_dir}/{img_name}'
    return image.read_img(img_fp, channel_first)

def parse_xml(xml_path, category_to_idx):
    with open(xml_path, 'r') as xml:
        data = xml.read()
    root = ET.XML(data)
    objs = root.findall('object')
    boxes = []
    for obj in objs:
        name = obj.find('name').text
        idx = category_to_idx[name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, idx])
    return boxes



if __name__ == '__main__':
    xml_path = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/Annotations/2007_000032.xml'
    voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    category_to_idx = {cls: idx for idx, cls in enumerate(voc_classes)}
    boxes = parse_xml(xml_path, category_to_idx)
    print(boxes)