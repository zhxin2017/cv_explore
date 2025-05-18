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


def xyxy_to_cxcy(boxes):
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2]
    return torch.stack(b, dim=-1)


def inters(box1, box2):
    box1x1, box1y1, box1x2, box1y2 = box1.unbind(-1)
    box2x1, box2y1, box2x2, box2y2 = box2.unbind(-1)

    inter_x1 = torch.max(box1x1, box2x1)
    inter_y1 = torch.max(box1y1, box2y1)
    inter_x2 = torch.min(box1x2, box2x2)
    inter_y2 = torch.min(box1y2, box2y2)

    inter_w = torch.max(torch.zeros_like(inter_x1, device=box1.device), inter_x2 - inter_x1)
    inter_h = torch.max(torch.zeros_like(inter_y1, device=box1.device), inter_y2 - inter_y1)
    return inter_w * inter_h
