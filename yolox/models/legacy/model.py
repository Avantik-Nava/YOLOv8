#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Legacy YOLOX wrapper (inference): same nesting/names as YOLOX YOLOX class
so checkpoints trained in YOLOX repos load directly.
predict() returns per-image [M,6] (x1,y1,x2,y2,conf,cls) in pixels.
"""

import torch
import torch.nn as nn

from .backbone import YOLOPAFPN
from .head import YOLOXHead


class YOLOX(nn.Module):
    yolox_legacy = True  # marker: BGR top-left-pad preprocessing (YOLOX ValTransform)
    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict(self, images, conf_threshold=0.25, nms_threshold=0.45,
                iou_threshold=None):
        """images: Tensor [B,3,H,W] letterboxed. Returns list of [M,6] tensors."""
        if iou_threshold is not None:
            nms_threshold = iou_threshold
        from yolox.utils.boxes import nms

        self.eval()
        outputs = self.forward(images)  # [B,N,4+1+nc] cxcywh+obj+cls
        out = []
        for b in range(outputs.shape[0]):
            pred = outputs[b]
            boxes = pred[:, :4]
            obj = pred[:, 4:5]
            cls = pred[:, 5:]
            scores = obj * cls
            conf, cls_id = scores.max(-1)
            keep = conf > conf_threshold
            boxes, conf, cls_id = boxes[keep], conf[keep], cls_id[keep].float()
            if boxes.shape[0] > 2000:  # cap before NMS (low-conf floods on CPU)
                top = conf.argsort(descending=True)[:2000]
                boxes, conf, cls_id = boxes[top], conf[top], cls_id[top]
            if boxes.shape[0] == 0:
                out.append(torch.zeros((0, 6), device=images.device))
                continue
            # xyxy
            xyxy = torch.stack([boxes[:, 0] - boxes[:, 2] / 2,
                                boxes[:, 1] - boxes[:, 3] / 2,
                                boxes[:, 0] + boxes[:, 2] / 2,
                                boxes[:, 1] + boxes[:, 3] / 2], -1)
            rows = []
            for c in cls_id.unique():
                m = cls_id == c
                idx = nms(xyxy[m], conf[m], nms_threshold)
                gi = torch.where(m)[0][idx]
                for i in gi.tolist():
                    rows.append(torch.cat([xyxy[i], conf[i].unsqueeze(0),
                                           cls_id[i].unsqueeze(0)]))
            if rows:
                det = torch.stack(rows)
                out.append(det[det[:, 4].argsort(descending=True)])
            else:
                out.append(torch.zeros((0, 6), device=images.device))
        return out
