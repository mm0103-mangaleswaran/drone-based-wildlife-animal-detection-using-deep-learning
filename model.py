
import torch
import torch.nn as nn
from .patchpack import PatchPack
from .backbone import GhostMobileBackbone
from .lite_bidfpn import LiteBiDFPN
from .dynamic_sa_head import TinyHeadDynamicSA
from .config import ModelConfig
from .utils import decode_yolo
from .nms import batched_nms

class MobileYOLOLite(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.pack = PatchPack(downscale=2)
        self.backbone = GhostMobileBackbone(width_mult=1.0, exp_ratio=6)
        # channel guesses (match backbone head outputs)
        c3, c4, c5 = 24, 32, 128
        self.neck = LiteBiDFPN(c3=c3, c4=c4, c5=c5)
        self.head = TinyHeadDynamicSA(c3=c3, c4=c4, num_classes=cfg.num_classes,
                                      anchors_p3=cfg.anchors_p3, anchors_p4=cfg.anchors_p4)

    def forward(self, x, decode=False, conf_thresh=None, nms_iou=None):
        if conf_thresh is None: conf_thresh = self.cfg.conf_thresh
        if nms_iou is None: nms_iou = self.cfg.nms_iou
        x = self.pack(x)
        p3, p4, p5 = self.backbone(x)
        f3, f4, f5 = self.neck(p3, p4, p5)
        out = self.head(f3, f4)
        if not decode:
            return out  # raw logits and gate stats for loss computation/optimization
        # Decode to boxes per scale then NMS
        B = x.shape[0]
        preds = []
        for b in range(B):
            boxes_all = []
            scores_all = []
            labels_all = []
            # P3
            boxes, obj, cls = decode_yolo(out['p3'][b:b+1], self.head.anchors_p3.to(x.device), self.cfg.num_classes, stride=4)  # 608->152 => stride 4
            scores, labels = (obj.unsqueeze(-1) * cls).max(-1)
            boxes_all.append(boxes.squeeze(0))
            scores_all.append(scores.squeeze(0))
            labels_all.append(labels.squeeze(0))
            # P4
            boxes2, obj2, cls2 = decode_yolo(out['p4'][b:b+1], self.head.anchors_p4.to(x.device), self.cfg.num_classes, stride=8)  # 608->76 => stride 8
            scores2, labels2 = (obj2.unsqueeze(-1) * cls2).max(-1)
            boxes_all.append(boxes2.squeeze(0))
            scores_all.append(scores2.squeeze(0))
            labels_all.append(labels2.squeeze(0))
            boxes_cat = torch.cat(boxes_all, 0)
            scores_cat = torch.cat(scores_all, 0)
            labels_cat = torch.cat(labels_all, 0)
            sel = scores_cat > conf_thresh
            boxes_cat, scores_cat, labels_cat = boxes_cat[sel], scores_cat[sel], labels_cat[sel]
            keep = batched_nms(boxes_cat, scores_cat, labels_cat, iou_threshold=nms_iou)
            preds.append({
                "boxes": boxes_cat[keep],
                "scores": scores_cat[keep],
                "labels": labels_cat[keep],
            })
        return preds, out.get("gate_mean", None)
