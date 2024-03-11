
def generate_anchors():

    xy_stride = 16
    wh_stride = 16
    W = 128
    H = 128
    ratio_thresh = 5

    anchors = []
    for h in range(16, H, wh_stride):
        for w in range(16, W, wh_stride):
            ratio = max(w, h) / min(w, h)
            # print(ratio)
            if ratio > ratio_thresh:
                continue
            for y in range(8, H, xy_stride):
                for x in range(8, W, xy_stride):
                    x1 = x
                    y1 = y
                    x2 = x1 + w
                    y2 = y1 + h
                    if x2 > W - xy_stride / 2 or y2 > H - xy_stride / 2:
                        continue
                    anchors.append([x1 / W, y1 / H, x2 / W, y2 / H])
    return anchors


if __name__ == '__main__':
    anchors = generate_anchors()
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
            axes[i, j].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=random.choice(colors), lw=1))
    plt.pause(0)