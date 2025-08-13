
# dataset
img_root_dir = '/Users/zx/Documents/dataset/VOCdevkit/VOC2012/JPEGImages'
xml_root_dir = '/Users/zx/Documents/dataset//VOCdevkit/VOC2012/Annotations'
filelist_files = ['/Users/zx/Documents/dataset/VOCdevkit/VOC2012/ImageSets/Main/train.txt']
img_h = 360
img_w = 480


# model
ds1 = 3
ds2 = 4
downsample_seg = 2
dmodel = 256
dhead = 32
num_enc_layer = 20
num_dec_layer = 8
num_query = 300


# train
batch_size = 2
epoch = 300