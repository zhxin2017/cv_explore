
# dataset
img_root_dir = '/Users/zx/Documents/dataset/VOC2012/JPEGImages'
xml_root_dir = '/Users/zx/Documents/dataset/VOC2012/Annotations'
filelist_files = ['/Users/zx/Documents/dataset/VOC2012/ImageSets/Main/train2017.txt']
img_h = 252
img_w = 336


# model
ds1 = 3
ds2 = 2
downsample_seg = 4
dmodel = 256
dhead = 32
num_enc_layer = 24
num_dec_layer = 8
num_query = 128

# train
batch_size = 1
epoch = 300