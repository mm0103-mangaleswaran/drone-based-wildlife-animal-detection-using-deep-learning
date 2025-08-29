
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    num_classes: int = 6  # WAID default classes
    img_size: int = 608
    # Two heads: 152x152 (P3) and 76x76 (P4)
    # Anchors: (w,h) in pixels for each scale (3 per scale)
    anchors_p3: Tuple[Tuple[float, float], ...] = ((7,7), (10,12), (14,18))
    anchors_p4: Tuple[Tuple[float, float], ...] = ((20,24), (28,32), (36,40))
    conf_thresh: float = 0.25
    nms_iou: float = 0.5
