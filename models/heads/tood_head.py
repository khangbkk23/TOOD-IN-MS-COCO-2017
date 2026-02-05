import torch
import torch.nn as nn
import torch.nn.functional as F

class TOODHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=256, feat_channels=256, stack_convs=6):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        
        # 1. Task interactive conv Layers
        self.inter_convs = nn.ModuleList()
        for i in range(stack_convs):
            chn = in_channels if i == 0 else feat_channels
            self.inter_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn, feat_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(feat_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # 2. Layer attention
        self.layer_attn_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_channels, feat_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, stack_convs, 1),
            nn.Sigmoid()
        )
        
        # 3. Predictors
        self.cls_head = nn.Conv2d(feat_channels, num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(feat_channels, 4, 3, padding=1)
        
        self._init_weights()

    def _init_weights(self):
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        
        for x in feats:
            inter_features = []
            for conv in self.inter_convs:
                x = conv(x)
                inter_features.append(x)

            avg_feat = inter_features[-1]
            w = self.layer_attn_fc(avg_feat)
            feat_final = inter_features[-1] * w[:, -1:, :, :]
            
            # Output
            cls_score = self.cls_head(feat_final)
            bbox_pred = self.reg_head(feat_final)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            
        return cls_scores, bbox_preds