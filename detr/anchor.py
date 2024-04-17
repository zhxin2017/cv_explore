import math
import torch
import random

anchor_max_size = 8


def generate_anchors(random_shift=False, random_sample_ratio=1):
    ratios = [1, 2, 4]
    sizes = [1, 2, 4, 6]
    strides = [1, 1, 2, 3]
    anchor_max_w = anchor_max_h = sizes[-1]

    anchors = []
    for h, sy in zip(sizes, strides):
        for w, sx in zip(sizes, strides):
            ratio = max(w, h) // min(w, h)
            # print(ratio)
            if ratio not in ratios:
                continue
            for y1 in range(0, anchor_max_h, sy):
                y2 = y1 + h
                if y2 > anchor_max_h:
                    continue
                for x1 in range(0, anchor_max_w, sx):
                    x2 = x1 + w
                    if x2 > anchor_max_w:
                        continue
                    if random_shift:
                        x1_shift = (random.random() - .5) * .8 * sx
                        y1_shift = (random.random() - .5) * .8 * sy
                        x2_shift = (random.random() - .5) * .8 * sx
                        y2_shift = (random.random() - .5) * .8 * sy
                        x1, y1 = max(x1 + x1_shift, 0), max(y1 + y1_shift, 0)
                        x2, y2 = min(x2 + x2_shift, anchor_max_w), min(y2 + y2_shift, anchor_max_h)
                    anchors.append([x1 / anchor_max_w, y1 / anchor_max_h, x2 / anchor_max_w,
                                    y2 / anchor_max_h])
    if random_sample_ratio < 1:
        sample_num = int(len(anchors) * random_sample_ratio)
        random.shuffle(anchors)
        anchors = anchors[:sample_num]
    return anchors


def generate_anchors2(stride_ratio=-1.0, random_shift=True):
    h_iter = [1, 2, 3, 4, 5, 6, 7, 8]
    w_iter = [1, 2, 3, 4, 5, 6, 7, 8]
    anchor_max_size_W = w_iter[-1]
    anchor_max_size_H = h_iter[-1]

    anchors = []
    for h in h_iter:
        y_stride = 1 if stride_ratio == -1 else math.ceil(h * stride_ratio)
        for w in w_iter:
            x_stride = 1 if stride_ratio == -1 else math.ceil(w * stride_ratio)

            ratio = max(w, h) // min(w, h)
            if ratio > 5:
                continue
            for y1 in range(0, anchor_max_size_H, y_stride):
                y2 = y1 + h
                if y2 > anchor_max_size_H + .5 * y_stride:
                    continue
                for x1 in range(0, anchor_max_size_W, x_stride):
                    if random_shift:
                        x1_shift = random.random() / (anchor_max_size_W / w_iter[0] * 2)
                        y1_shift = random.random() / (anchor_max_size_H / h_iter[0] * 2)
                        w_shift = (random.random() - .5) / (anchor_max_size_W / w_iter[0] * 2)
                        h_shift = (random.random() - .5) / (anchor_max_size_H / h_iter[0] * 2)
                    else:
                        x1_shift = y1_shift = w_shift = h_shift = 0
                    x1 = x1 + x1_shift
                    y1 = y1 + y1_shift
                    x2 = x1 + w
                    if x2 > anchor_max_size_W + .5 * x_stride:
                        continue
                    x2 = x2 + w_shift
                    y2 = y2 + h_shift
                    anchors.append([x1 / anchor_max_size_W, y1 / anchor_max_size_H, x2 / anchor_max_size_W,
                                    y2 / anchor_max_size_H])
    return anchors


if __name__ == '__main__':
    anchors = generate_anchors(random_shift=False)
    print(len(anchors))
    # print(anchors)

    import torch
    import random
    from matplotlib import pyplot as plt

    colors = ['gray', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'blue', 'olive', 'cyan']

    img = torch.ones([128, 128, 3]) * 0.6
    fig, axes = plt.subplots(8, 8)
    random.shuffle(anchors)
    for i in range(8):
        for j in range(8):
            axes[i, j].axis('off')
            axes[i, j].imshow(img)

            x1, y1, x2, y2 = anchors[4 * i + j]
            x1 = x1 * 128
            y1 = y1 * 128
            x2 = x2 * 128
            y2 = y2 * 128
            axes[i, j].add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=random.choice(colors), lw=1))
    plt.pause(0)
