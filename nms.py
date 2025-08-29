
import torch

def box_iou(box1, box2):
    # box: (N, 4) / (M, 4) format: x1,y1,x2,y2
    inter = (torch.min(box1[:, None, 2:], box2[None, :, 2:]) - torch.max(box1[:, None, :2], box2[None, :, :2])).clamp(0).prod(2)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union

def nms(boxes, scores, iou_threshold=0.5):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze(0)
        idxs = idxs[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def batched_nms(boxes, scores, labels, iou_threshold=0.5):
    # class-wise NMS by offsetting boxes per class
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_offset = boxes.max()
    offsets = labels.to(boxes) * (max_offset + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)
