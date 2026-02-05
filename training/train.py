import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import custom modules
from utils.config import load_config
from datasets.coco import COCODataset
from datasets.collate import collate_fn
from models.detectors.tood import TOOD
from models.assigners.task_aligned_assigner import TaskAlignedAssigner
from models.utils.anchors import make_anchors
from utils.box_ops import box_iou, dist2bbox

def quality_focal_loss(pred, target, beta=2.0):
    pred_sigmoid = pred.sigmoid()
    scale_factor = (pred_sigmoid - target).abs().pow(beta)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none') * scale_factor
    return loss.sum()

def get_giou_loss(pred, target, weight=None):
    """
    Generalized IoU Loss weighted by alignment metric.
    """
    from torchvision.ops import generalized_box_iou_loss
    loss = generalized_box_iou_loss(pred, target, reduction='none')
    if weight is not None:
        loss *= weight
    return loss.sum()


def train():
    cfg = load_config("./config/config.yaml")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    train_dataset = COCODataset(
        root=f"{cfg.data.root_dir}/{cfg.data.train_imgs}",
        ann_file=f"{cfg.data.root_dir}/{cfg.data.train_ann}",
        img_size=cfg.data.img_size,
        mosaic=cfg.data.use_mosaic
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True
    )
    
    model = TOOD(cfg).to(device)
    assigner = TaskAlignedAssigner(
        topk=cfg.tal.topk, 
        alpha=cfg.tal.alpha, 
        beta=cfg.tal.beta,
        num_classes=cfg.data.num_classes
    )

    optimizer = optim.SGD(
        model.parameters(), 
        lr=cfg.train.lr, 
        momentum=cfg.train.momentum, 
        weight_decay=cfg.train.weight_decay
    )
    
    print(f"Starting training on {device} for {cfg.train.epochs} epochs...")
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            cls_scores, reg_dist = model(images)
            
            # Generate anchors for current feature maps
            strides = [8, 16, 32, 64, 128] # P3 to P7 strides
            anchor_points, stride_tensor = make_anchors(cls_scores, strides)
            
            # Concatenate outputs from all levels
            # cls_scores: [B, num_anchors, num_classes]
            # reg_dist: [B, num_anchors, 4]
            cls_scores = torch.cat([x.flatten(2).transpose(1, 2) for x in cls_scores], dim=1)
            reg_dist = torch.cat([x.flatten(2).transpose(1, 2) for x in reg_dist], dim=1)
            
            # Decode bboxes: [B, num_anchors, 4]
            pd_bboxes = dist2bbox(reg_dist, anchor_points, stride_tensor)
            
            # Calculate Loss for each image in batch
            batch_cls_loss = 0
            batch_reg_loss = 0
            
            for j in range(len(targets)):
                gt_boxes = targets[j]['boxes'].to(device)
                gt_labels = targets[j]['labels'].to(device)
                
                # Task-aligned Assignment (TAL)
                target_labels, target_bboxes, target_scores = assigner(
                    cls_scores[j].detach().sigmoid(), 
                    pd_bboxes[j].detach(), 
                    gt_labels, 
                    gt_boxes
                )
                
                # Classification Loss (QFL)
                num_anchors = cls_scores.shape[1]
                t_cls = torch.zeros_like(cls_scores[j])
                pos_idx = target_labels < cfg.data.num_classes
                if pos_idx.any():
                    t_cls[pos_idx, target_labels[pos_idx]] = target_scores[pos_idx]
                
                batch_cls_loss += quality_focal_loss(cls_scores[j], t_cls)
                
                # Regression loss 
                if pos_idx.any():
                    batch_reg_loss += get_giou_loss(
                        pd_bboxes[j][pos_idx], 
                        target_bboxes[pos_idx], 
                        weight=target_scores[pos_idx]
                    )

            total_loss = (batch_cls_loss + batch_reg_loss) / len(targets)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
            if i % cfg.train.print_freq == 0:
                print(f"Epoch [{epoch+1}/{cfg.train.epochs}] Iter [{i}/{len(train_loader)}] "
                      f"Loss: {total_loss.item():.4f} (Cls: {batch_cls_loss.item()/len(targets):.4f}, "
                      f"Reg: {batch_reg_loss.item()/len(targets):.4f})")

        # Save Checkpoint
        if (epoch + 1) % cfg.train.val_interval == 0:
            save_path = os.path.join(cfg.train.save_dir, f"tood_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()