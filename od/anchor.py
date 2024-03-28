from od.config import anchor_min_size, anchor_max_size
import torch


def generate_anchors(variable_stride=False):
    anchor_max_size_W = anchor_max_size
    anchor_max_size_H = anchor_max_size
    ratio_thresh = 5

    anchors = []
    for h in range(anchor_min_size, anchor_max_size_H + 1, anchor_min_size):
        if variable_stride:
            y_stride = int((h / anchor_min_size) ** 0.5 * anchor_min_size)
        else:
            y_stride = anchor_min_size
        for w in range(anchor_min_size, anchor_max_size_W + 1, anchor_min_size):
            x_stride = int((w / anchor_min_size) ** 0.5 * anchor_min_size)
            ratio = max(w, h) / min(w, h)
            # print(ratio)
            if ratio > ratio_thresh:
                continue
            for y1 in range(0, anchor_max_size_H, y_stride):
                y2 = y1 + h
                if y2 > anchor_max_size_H - anchor_min_size / 2:
                    continue
                for x1 in range(0, anchor_max_size_W, x_stride):
                    x2 = x1 + w
                    if x2 > anchor_max_size_W - anchor_min_size / 2:
                        continue
                    anchors.append([x1 / anchor_max_size_W, y1 / anchor_max_size_H, x2 / anchor_max_size_W,
                                    y2 / anchor_max_size_H])
    return anchors


if __name__ == '__main__':
    anchors = generate_anchors(variable_stride=True)
    print(len(anchors))
    # print(anchors)

    import torch
    import matplotlib
    import random
    from matplotlib import pyplot as plt

    colors = ['gray', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'blue', 'olive', 'cyan']

    img = torch.ones([128, 128, 3]) * 0.6
    fig, axes = plt.subplots(4, 4)
    random.shuffle(anchors)
    for i in range(4):
        for j in range(4):
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
