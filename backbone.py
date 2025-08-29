
import torch
import torch.nn as nn
from .ghost import GhostBottleneck, GhostConv

class GhostMobileBackbone(nn.Module):
    """Backbone that produces multi-scale features (P3:152, P4:76, P5:38 for 608 input pre-packed to 304x304).
    Input expected: (B, 12, 304, 304) from PatchPack
    """
    def __init__(self, width_mult=1.0, exp_ratio=6):
        super().__init__()
        def c(ch): 
            return max(8, int(ch * width_mult))

        self.stem = GhostConv(12, c(16), 3, 1)   # (304)->(304)
        # Downsample to 152
        self.stage1 = nn.Sequential(
            GhostBottleneck(c(16), c(24), stride=2, exp_ratio=exp_ratio),  # 304->152
            GhostBottleneck(c(24), c(24), stride=1, exp_ratio=exp_ratio),
        )
        # Downsample to 76
        self.stage2 = nn.Sequential(
            GhostBottleneck(c(24), c(32), stride=2, exp_ratio=exp_ratio),  # 152->76
            GhostBottleneck(c(32), c(32), stride=1, exp_ratio=exp_ratio),
        )
        # Downsample to 38
        self.stage3 = nn.Sequential(
            GhostBottleneck(c(32), c(64), stride=2, exp_ratio=exp_ratio),  # 76->38
            GhostBottleneck(c(64), c(64), stride=1, exp_ratio=exp_ratio),
        )
        # A rich head conv (optional)
        self.head = GhostConv(c(64), c(128), 1, 1)

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage1(x)  # 152x152
        p4 = self.stage2(p3) # 76x76
        p5 = self.stage3(p4) # 38x38
        p5 = self.head(p5)
        return p3, p4, p5
