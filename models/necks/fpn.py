import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
        
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        last_inner = self.lateral_convs[-1](inputs[-1])
        results = [self.fpn_convs[-1](last_inner)]

        for idx in range(len(inputs) - 2, -1, -1):
            inner_lat = self.lateral_convs[idx](inputs[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lat.shape[-2:], mode="nearest")
            last_inner = inner_lat + inner_top_down
            results.insert(0, self.fpn_convs[idx](last_inner))

        p5 = results[-1]
        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(F.relu(p6))
        
        results.extend([p6, p7])
        
        return results