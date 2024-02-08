import torch


def pad_bbox(bboxes, indices, n_query):
    n_bboxes = len(bboxes)
    bboxes_padded = torch.zeros(n_query, 4)
    indices_padded = torch.zeros(n_query, dtype=torch.int)
    if len(bboxes) > 0:
        bboxes_padded[:n_bboxes] = torch.tensor(bboxes)
        indices_padded[:n_bboxes] = torch.tensor(indices, dtype=torch.int)
    return bboxes_padded, indices_padded


def offset_box(boxes_xywh, offset_h, offset_w):
    boxes_xywh[:, :1] = boxes_xywh[:, :1] + offset_w
    boxes_xywh[:, 1:2] = boxes_xywh[:, 1:2] + offset_h
    return boxes_xywh


def xywh_to_xyxy(boxes):
    boxes[:, 2:3] = boxes[:, 2:3] + boxes[:, :1]
    boxes[:, 3:4] = boxes[:, 3:4] + boxes[:, 1:2]
    return boxes


def cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# def cxcywh_to_xyxy2(boxes):
#     boxes2 = boxes + 0
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)


def xyxy_to_cxcywh(boxes):
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)