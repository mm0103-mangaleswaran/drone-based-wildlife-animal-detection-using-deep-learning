
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, kernel_size=3, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=bias)
    def forward(self, x): return self.conv(x)

class GhostConv(nn.Module):
    """Ghost convolution: primary conv -> cheap depthwise ops to generate ghost maps."""
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, ratio=2, dw_kernel_size=3, act=True):
        super().__init__()
        init_ch = math.ceil(out_ch / ratio)
        new_ch = init_ch * (ratio - 1)
        padding = kernel_size // 2
        self.primary = nn.Conv2d(in_ch, init_ch, kernel_size, stride, padding, bias=False)
        self.primary_bn = nn.BatchNorm2d(init_ch)
        self.cheap = DepthwiseConv(init_ch, kernel_size=dw_kernel_size, stride=1, padding=dw_kernel_size//2, bias=False)
        self.cheap_pw = nn.Conv2d(init_ch, new_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.cheap_bn = nn.BatchNorm2d(new_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.out_ch = out_ch

    def forward(self, x):
        x1 = self.act(self.primary_bn(self.primary(x)))
        x2 = self.act(self.cheap_bn(self.cheap_pw(self.cheap(x1))))
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_ch, :, :]

class GhostBottleneck(nn.Module):
    """MobileNetV2-like bottleneck built with GhostConv."""
    def __init__(self, in_ch, out_ch, stride=1, exp_ratio=6):
        super().__init__()
        hidden_ch = _make_divisible(in_ch * exp_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.expand = GhostConv(in_ch, hidden_ch, kernel_size=1, stride=1, act=True)
        self.dw = DepthwiseConv(hidden_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_ch)
        self.project = GhostConv(hidden_ch, out_ch, kernel_size=1, stride=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.expand(x)
        out = self.act(self.dw_bn(self.dw(out)))
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out
