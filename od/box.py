import torch


def pad_bbox(bboxes, indices, n_query):
    n_bboxes = len(bboxes)
    bboxes_padded = torch.zeros(n_query, 4)
    indices_padded = torch.zeros(n_query, dtype=torch.int)
    if len(bboxes) > 0:
        bboxes_padded[:n_bboxes] = torch.tensor(bboxes)
        indices_padded[:n_bboxes] = torch.tensor(indices, dtype=torch.int)
    return bboxes_padded, indices_padded


def resize_bbox(b_xyxy_l, ratio):
    return [[i * ratio for i in b] for b in b_xyxy_l]


def xywh_to_xyxy(bboxes, offset_h, offset_w):
    bboxes[:, :1] = bboxes[:, :1] + offset_w
    bboxes[:, 1:2] = bboxes[:, 1:2] + offset_h
    bboxes[:, 2:3] = bboxes[:, 2:3] + bboxes[:, :1]
    bboxes[:, 3:4] = bboxes[:, 3:4] + bboxes[:, 1:2]
    return bboxes
