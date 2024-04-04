import math

h_iter = [1, 2, 4, 6, 8]
w_iter = [1, 2, 4, 6, 8]
anchor_max_size_W = w_iter[-1]
anchor_max_size_H = h_iter[-1]
anchor_max_size = max(anchor_max_size_H, anchor_max_size_W)


def generate_anchors(stride_ratio=-1.0):
    ratios = [1, 2, 3, 4]
    anchors = []
    for h in h_iter:
        y_stride = 1 if stride_ratio == -1 else math.ceil(h * stride_ratio)
        for w in w_iter:
            x_stride = 1 if stride_ratio == -1 else math.ceil(w * stride_ratio)

            ratio = max(w, h) // min(w, h)
            # print(ratio)
            if ratio not in ratios:
                continue
            for y1 in range(0, anchor_max_size_H, y_stride):
                y2 = y1 + h
                if y2 > anchor_max_size_H - .5 * y_stride:
                    continue
                for x1 in range(0, anchor_max_size_W, x_stride):
                    x2 = x1 + w
                    if x2 > anchor_max_size_W - .5 * x_stride:
                        continue
                    anchors.append([x1 / anchor_max_size_W, y1 / anchor_max_size_H, x2 / anchor_max_size_W,
                                    y2 / anchor_max_size_H])
    return anchors


if __name__ == '__main__':
    anchors = generate_anchors(stride_ratio=1/3)
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
