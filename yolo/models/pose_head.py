"""YOLOv8-pose head: detect + keypoints (yolov8n-pose.pt, 17 COCO kpts default).

Adds per-scale kpt branch: Conv1x1(c -> nkpt*3) with (x, y, vis) in cells.
Decode: kpt xy = (sigmoid*2-0.5 + anchor_grid) * stride, vis = sigmoid.
"""

import torch

from .head import YOLOv8Head
from .common import Conv
import torch.nn as nn


class YOLOv8PoseHead(YOLOv8Head):
    def __init__(self, num_classes=1, in_channels=(256, 256, 512), reg_max=16,
                 strides=(8, 16, 32), nkpt=17):
        super().__init__(num_classes, in_channels, reg_max, strides)
        self.nkpt = nkpt
        self.kpt_convs = nn.ModuleList()
        self.kpt_preds = nn.ModuleList()
        for c in in_channels:
            self.kpt_convs.append(nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)))
            self.kpt_preds.append(nn.Conv2d(c, nkpt * 3, 1))

    def forward(self, feats):
        cls_l, reg_l = super().forward(feats)
        kpt_l = [kp(kc(f)) for kc, kp, f in zip(self.kpt_convs, self.kpt_preds, feats)]
        return cls_l, reg_l, kpt_l

    def decode(self, feats):
        cls_list, reg_list, kpt_list = self.forward(feats)
        boxes, scores = super().decode(feats)
        kpts = []
        for kpt, stride, cls in zip(kpt_list, self.strides, cls_list):
            b, _, h, w = kpt.shape
            n = h * w
            k = kpt.view(b, self.nkpt, 3, n).permute(0, 3, 1, 2).contiguous()
            yv, xv = torch.meshgrid(
                torch.arange(h, device=kpt.device), torch.arange(w, device=kpt.device), indexing="ij")
            grid = torch.stack((xv, yv), -1).float().view(1, -1, 1, 2)
            xy = ((k[..., :2].sigmoid() * 2 - 0.5) + grid) * stride  # [B,N,K,2]
            vis = k[..., 2:].sigmoid()
            kpts.append(torch.cat([xy, vis], -1))
        return boxes, scores, torch.cat(kpts, 1)  # kpts [B,N,K,3]

    def predict(self, feats):
        boxes, scores, kpts = self.decode(feats)
        return torch.cat([boxes, scores], -1), kpts
