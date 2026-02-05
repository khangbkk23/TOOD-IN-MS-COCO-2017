import torch.nn as nn
from models.backbones.resnet import ResNet
from models.necks.fpn import FPN
from models.heads.tood_head import TOODHead

class TOOD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(
            pretrained=cfg.model.pretrained,
            frozen_bn=cfg.model.frozen_bn
        )
        
        self.neck = FPN(
            in_channels=[512, 1024, 2048],
            out_channels=cfg.model.neck.out_channels
        )

        self.head = TOODHead(
            num_classes=cfg.data.num_classes,
            in_channels=cfg.model.neck.out_channels,
            feat_channels=cfg.model.head.feat_channels
        )

    def forward(self, x):
        body_feats = self.backbone(x)
        pyramid_feats = self.neck(body_feats)
        cls_scores, bbox_preds = self.head(pyramid_feats)
        
        return cls_scores, bbox_preds