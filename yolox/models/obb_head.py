"""YOLOv8-obb head: detect + angle branch (yolov8n-obb.pt, DOTA).

Angle encoded as single value in [-pi/2, pi/2) via tanh; loss = L1 on angle
for foreground + standard detect loss. Rotated IoU approximated axis-aligned
for assignment (documented simplification).
"""

import torch
import torch.nn as nn

from .common import Conv
from .head import YOLOv8Head


class YOLOv8OBBHead(YOLOv8Head):
    def __init__(self, num_classes=15, in_channels=(256, 256, 512), reg_max=16,
                 strides=(8, 16, 32), angle_range=3.14159265 / 2):
        super().__init__(num_classes, in_channels, reg_max, strides)
        self.angle_range = angle_range
        self.ang_convs = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        for c in in_channels:
            self.ang_convs.append(nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)))
            self.ang_preds.append(nn.Conv2d(c, 1, 1))

    def forward(self, feats):
        cls_l, reg_l = super().forward(feats)
        ang_l = [ap(ac(f)) for ac, ap, f in zip(self.ang_convs, self.ang_preds, feats)]
        return cls_l, reg_l, ang_l

    def decode(self, feats):
        cls_list, reg_list, ang_list = self.forward(feats)
        boxes, scores = super().decode(feats)
        angs = []
        for a in ang_list:
            b, _, h, w = a.shape
            angs.append((a.view(b, -1, 1).permute(0, 2, 1) if False else
                         a.view(b, 1, -1).permute(0, 2, 1)).tanh() * self.angle_range)
        return boxes, scores, torch.cat(angs, 1)  # [B,N,1] radians

    def predict(self, feats):
        boxes, scores, ang = self.decode(feats)
        return torch.cat([boxes, scores], -1), ang
