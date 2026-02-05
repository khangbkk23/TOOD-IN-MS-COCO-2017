import torch

def box_iou(boxes1, boxes2):
    """
    boxes1: [N, 4] (x1, y1, x2, y2)
    boxes2: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / union

def dist2bbox(distance, anchor_points, stride):
    x1 = anchor_points[..., 0] - distance[..., 0]
    y1 = anchor_points[..., 1] - distance[..., 1]
    x2 = anchor_points[..., 0] + distance[..., 2]
    y2 = anchor_points[..., 1] + distance[..., 3]
    
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return bboxes * stride