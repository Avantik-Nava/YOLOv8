"""Box utilities: format converts, IoU/CIoU, NMS, DFL helpers."""

import torch
import torch.nn.functional as F


def xywh2xyxy(boxes):
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def xyxy2xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def dist2bbox(points, dist, xywh=False):
    """points [N,2] centers, dist [N,4] ltrb -> boxes."""
    x1 = points[..., 0] - dist[..., 0]
    y1 = points[..., 1] - dist[..., 1]
    x2 = points[..., 0] + dist[..., 2]
    y2 = points[..., 1] + dist[..., 3]
    if xywh:
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2dist(points, boxes, reg_max=16):
    """points [N,2], boxes [N,4] xyxy -> ltrb distances (clamped to reg_max-1e-3)."""
    l = (points[..., 0] - boxes[..., 0]).clamp(0, reg_max - 1e-3)
    t = (points[..., 1] - boxes[..., 1]).clamp(0, reg_max - 1e-3)
    r = (boxes[..., 2] - points[..., 0]).clamp(0, reg_max - 1e-3)
    b = (boxes[..., 3] - points[..., 1]).clamp(0, reg_max - 1e-3)
    return torch.stack([l, t, r, b], -1)


def box_iou(a, b, ciou=False, eps=1e-7):
    """a [N,4], b [M,4] xyxy -> [N,M] IoU (or CIoU if requested)."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.max(a[:, None, :2], b[:, :2])  # [N,M,2]
    rb = torch.min(a[:, None, 2:], b[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    # CIoU
    cw = torch.max(a[:, None, 2], b[:, 2]) - torch.min(a[:, None, 0], b[:, 0])
    ch = torch.max(a[:, None, 3], b[:, 3]) - torch.min(a[:, None, 1], b[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b[:, 0] + b[:, 2] - a[:, None, 0] - a[:, None, 2]) ** 2 +
            (b[:, 0 + 1] + b[:, 3] - a[:, None, 1] - a[:, None, 3]) ** 2) / 4
    v = (4 / (3.14159265 ** 2)) * torch.pow(
        torch.atan((b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1] + eps)) -
        torch.atan((a[:, 2] - a[:, 0]) / (a[:, 3] - a[:, 1] + eps))[:, None], 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + v * alpha)


def nms(boxes, scores, iou_thres=0.7):
    """Pure-torch NMS. boxes [N,4], scores [N] -> keep indices."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_thres)
    except Exception:
        pass
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_thres]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.7, max_det=300):
    """pred [B,N,4+nc] (boxes xyxy + cls scores) -> list per image [M,6] (x1,y1,x2,y2,conf,cls)."""
    out = []
    nc = pred.shape[-1] - 4
    for b in range(pred.shape[0]):
        boxes = pred[b, :, :4]
        scores = pred[b, :, 4:]
        conf, cls = scores.max(-1)
        mask = conf > conf_thres
        boxes, conf, cls = boxes[mask], conf[mask], cls[mask].float()
        if boxes.shape[0] == 0:
            out.append(torch.zeros((0, 6), device=pred.device))
            continue
        # class-aware NMS: offset boxes by class
        order = conf.argsort(descending=True)[:max_det * 5]
        boxes, conf, cls = boxes[order], conf[order], cls[order]
        keep = []
        # NMS per class
        for c in cls.unique():
            m = cls == c
            kb = boxes[m]
            kc = conf[m]
            idx = nms(kb, kc, iou_thres)
            for i in idx:
                gi = torch.where(m)[0][i].item()
                keep.append(gi)
        keep = sorted(keep, key=lambda i: conf[i].item(), reverse=True)[:max_det]
        det = torch.cat([boxes[keep], conf[keep, None], cls[keep, None]], 1)
        out.append(det)
    return out
