import torch
from detr import box
from common.config import max_grid_h, max_grid_w, patch_size, max_img_len

grid_size = patch_size / max_img_len


def separate_bgd_grids(w, h, x_shift, y_shift, boxes):
    n_obj = len(boxes)
    n_grid = w * h

    grid_x1 = ((torch.arange(w, device=boxes.device) + x_shift) * grid_size).view(1, w).repeat(h, 1).reshape(n_grid)
    grid_y1 = ((torch.arange(h, device=boxes.device) + y_shift) * grid_size).view(h, 1).repeat(1, w).reshape(n_grid)
    grid_x2 = grid_x1 + grid_size
    grid_y2 = grid_y1 + grid_size
    grid_box = torch.stack([grid_x1, grid_y1, grid_x2, grid_y2], dim=-1)
    grid_box = grid_box.view(n_grid, 1, 4).repeat(1, n_obj, 1)

    grid_box_intersections = box.inters(grid_box, boxes.view(1, n_obj, 4).repeat(w * h, 1, 1))
    grid_intersection_sum = grid_box_intersections.sum(dim=-1)

    grid_indices = torch.arange(n_grid, device=boxes.device)
    grid_bgd_indices = grid_indices[grid_intersection_sum == 0]

    grid_obj_indices = grid_indices[grid_intersection_sum > 0]
    return grid_bgd_indices, grid_obj_indices, grid_box_intersections, grid_indices, grid_x1, grid_y1, grid_x2, grid_y2
