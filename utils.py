
import torch
import torch.nn.functional as F

def sigmoid(x): return 1 / (1 + torch.exp(-x))

def decode_yolo(pred, anchors, num_classes, stride):
    """Decode raw conv output to boxes/scores per scale.
    pred: (B, A*(5+nc), H, W)
    anchors: (A, 2) in pixels at input scale
    stride: input_size / feature_map_size
    Returns boxes (B, N, 4), obj (B, N), cls (B, N, nc)
    """
    B, _, H, W = pred.shape
    A = anchors.shape[0]
    pred = pred.view(B, A, 5 + num_classes, H, W).permute(0,1,3,4,2).contiguous()
    # grid
    yv, xv = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W, device=pred.device), indexing='ij')
    grid = torch.stack((xv, yv), 2).view(1,1,H,W,2).float()
    # components
    xy = (sigmoid(pred[..., 0:2]) + grid) * stride
    wh = (torch.exp(pred[..., 2:4]) * anchors.view(1, A, 1, 1, 2))
    obj = sigmoid(pred[..., 4])
    cls = sigmoid(pred[..., 5:])
    # convert to x1y1x2y2
    x1y1 = xy - wh / 2
    x2y2 = xy + wh / 2
    boxes = torch.cat([x1y1, x2y2], dim=-1).view(B, -1, 4)
    obj = obj.view(B, -1)
    cls = cls.view(B, -1, num_classes)
    return boxes, obj, cls
