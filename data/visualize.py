import cv2


def draw_bbox(img, bboxes):
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return img