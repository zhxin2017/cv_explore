
# dataset
img_root_dir = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/JPEGImages'
xml_root_dir = '/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/Annotations'
filelist_files = ['/Users/zx/Documents/ml/dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt']
img_h = 288
img_w = 512


# model
dmodel = 256
dhead = 32
num_enc_layer = 20
num_dec_layer = 8
num_query = 300


# train
batch_size = 2
epoch = 300