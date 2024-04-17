from common import image


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


if __name__ == '__main__':
    pass
