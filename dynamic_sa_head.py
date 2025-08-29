
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttentionGate(nn.Module):
    """Binary spatial gate computed from a 1x1 conv + sigmoid and threshold."""
    def __init__(self, in_ch, thresh: float = 0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1, 1, 0)
        self.thresh = thresh
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        score = self.sigmoid(self.conv(x))
        gate = (score > self.thresh).float()
        # Apply gating by masking activations (approximation of conditional compute)
        gated = x * gate
        gate_mean = gate.mean().detach()
        return gated, gate, score, gate_mean

class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_anchors, num_classes):
        super().__init__()
        hidden = max(32, in_ch // 2)
        self.conv = nn.Conv2d(in_ch, hidden, 3, 1, 1)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.pred = nn.Conv2d(hidden, num_anchors * (5 + num_classes), 1, 1, 0)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.pred(x)

class TinyHeadDynamicSA(nn.Module):
    """Dual-scale YOLO head with sparse attention gates for P3 (152) and P4 (76)."""
    def __init__(self, c3, c4, num_classes, anchors_p3, anchors_p4):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_p3 = torch.tensor(anchors_p3, dtype=torch.float32)
        self.anchors_p4 = torch.tensor(anchors_p4, dtype=torch.float32)
        self.gate3 = SparseAttentionGate(c3, thresh=0.5)
        self.gate4 = SparseAttentionGate(c4, thresh=0.5)
        self.head3 = DetectionHead(c3, num_anchors=len(anchors_p3), num_classes=num_classes)
        self.head4 = DetectionHead(c4, num_anchors=len(anchors_p4), num_classes=num_classes)

    def forward(self, p3, p4):
        p3_gated, g3, s3, g3_mean = self.gate3(p3)
        p4_gated, g4, s4, g4_mean = self.gate4(p4)
        out3 = self.head3(p3_gated)  # (B, A*(5+nc), 152, 152)
        out4 = self.head4(p4_gated)  # (B, A*(5+nc), 76, 76)
        # For optimizer gradient filtering, expose mean gate factor
        gate_mean = (g3_mean + g4_mean) / 2.0
        return {
            "p3": out3, "p4": out4,
            "gate_mean": gate_mean,
            "gates": {"p3": g3, "p4": g4},
            "scores": {"p3": s3, "p4": s4},
        }
