import torch

def make_anchors(feats, strides, grid_cell_offset=0.5):

    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[-2:]
        sx = torch.arange(w, device=feats[i].device) + grid_cell_offset
        sy = torch.arange(h, device=feats[i].device) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        
        anchor_points.append(torch.stack([sx, sy], dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feats[i].device))
        
    return torch.cat(anchor_points), torch.cat(stride_tensor)