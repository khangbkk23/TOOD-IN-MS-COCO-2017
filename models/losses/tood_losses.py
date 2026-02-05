import torch
import torch.nn as nn
import torch.nn.functional as F

def quality_focal_loss(pred, target_score, beta=2.0):
    pred_sigmoid = pred.sigmoid()
    scale_factor = (pred_sigmoid - target_score).abs().pow(beta)
    
    loss = F.binary_cross_entropy_with_logits(pred, target_score, reduction='none') * scale_factor
    return loss.sum()

def giou_loss(pred, target, weight=None):
    from torchvision.ops import generalized_box_iou_loss
    loss = generalized_box_iou_loss(pred, target, reduction='none')
    if weight is not None:
        loss *= weight
    return loss.sum()