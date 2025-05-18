import os

# img_root_dir = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/JPEGImages'
img_root_dir = '/Users/zx/Documents/ml/dataset/coco/val2017'

exts = set()
for root, dirs, files in os.walk(img_root_dir):
    for file in files:
        ext = file.split('.')[-1]
        exts.add(ext)

print(exts)