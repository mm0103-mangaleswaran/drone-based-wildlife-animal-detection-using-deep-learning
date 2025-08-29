
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ghost import GhostConv

class WeightedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))
        self.eps = 1e-4
    def forward(self, a, b):
        w1 = F.relu(self.w1)
        w2 = F.relu(self.w2)
        return (w1 * a + w2 * b) / (w1 + w2 + self.eps)

class LiteBiDFPN(nn.Module):
    """Three-scale bi-directional weighted fusion: (P3:152, P4:76, P5:38) -> refined P3,P4,P5"""
    def __init__(self, c3, c4, c5):
        super().__init__()
        self.p3_in = GhostConv(c3, c3, 1, 1)
        self.p4_in = GhostConv(c4, c4, 1, 1)
        self.p5_in = GhostConv(c5, c5, 1, 1)
        self.merge34 = WeightedAdd()
        self.merge45 = WeightedAdd()
        self.out3 = GhostConv(c3, c3, 1, 1)
        self.out4 = GhostConv(c4, c4, 1, 1)
        self.out5 = GhostConv(c5, c5, 1, 1)

    def forward(self, p3, p4, p5):
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)
        # Top-down
        p4_td = self.merge45(p4, F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3_td = self.merge34(p3, F.interpolate(p4_td, scale_factor=2, mode="nearest"))
        # Bottom-up
        p4_out = self.merge34(p4_td, F.max_pool2d(p3_td, kernel_size=2, stride=2))
        p5_out = self.merge45(p5, F.max_pool2d(p4_out, kernel_size=2, stride=2))
        return self.out3(p3_td), self.out4(p4_out), self.out5(p5_out)
