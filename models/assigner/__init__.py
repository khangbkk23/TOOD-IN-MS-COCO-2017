import torch
import torch.nn as nn
from utils.box_ops import box_iou # Giả định bạn sẽ tạo file utils này

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, alpha=1.0, beta=6.0, num_classes=80):
        super().__init__()
        self.topk = topk
        self.alpha = alpha # Power for classification score
        self.beta = beta   # Power for IoU
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """
        pd_scores: [num_anchors, num_classes]
        pd_bboxes: [num_anchors, 4]
        gt_labels: [num_gt, 1]
        gt_bboxes: [num_gt, 4]
        """
        num_gt = gt_bboxes.size(0)
        num_anchors = pd_bboxes.size(0)

        if num_gt == 0:
            return torch.full((num_anchors,), self.num_classes, dtype=torch.long), \
                   torch.zeros_like(pd_bboxes)

        # 1. Tính IoU giữa mọi dự đoán và mọi Ground Truth
        # ious shape: [num_gt, num_anchors]
        ious = box_iou(gt_bboxes, pd_bboxes)
        
        # 2. Lấy điểm dự đoán tương ứng với nhãn của GT
        # scores shape: [num_gt, num_anchors]
        batch_ind = torch.arange(num_gt, device=gt_labels.device).view(-1, 1)
        scores = pd_scores[:, gt_labels.flatten()].t() 

        # 3. Tính Alignment Metric t = s^alpha * u^beta
        alignment_metrics = scores.pow(self.alpha) * ious.pow(self.beta)

        # 4. Chọn Top-K ứng viên cho mỗi GT dựa trên t
        is_in_topk = self.select_topk_candidates(alignment_metrics, topk=self.topk)
        
        # 5. Căn chỉnh nhãn và tọa độ mục tiêu (simplified logic)
        # Tìm metric lớn nhất cho mỗi anchor để tránh xung đột giữa các GT
        max_metrics, max_indices = alignment_metrics.max(dim=0)
        
        target_labels = gt_labels[max_indices].flatten()
        target_labels[max_metrics <= 0] = self.num_classes # Nhãn background
        
        target_bboxes = gt_bboxes[max_indices]

        return target_labels, target_bboxes, max_metrics

    def select_topk_candidates(self, metrics, topk):
        # metrics: [num_gt, num_anchors]
        topk_metrics, topk_indices = torch.topk(metrics, topk, dim=-1)
        is_in_topk = torch.zeros_like(metrics, dtype=torch.bool)
        for i in range(topk):
            is_in_topk.scatter_(1, topk_indices[:, i:i+1], True)
        return is_in_topk