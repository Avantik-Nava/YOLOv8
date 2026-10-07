"""YOLOv8-seg head: detect head + mask coefficients + Proto (YOLACT-style).

Official: yolov8n-seg.pt etc. Masks = sigmoid(Proto(P3) @ coeff.T) cropped by boxes.
Simplified functional version: nm=32 coefficients, proto P3 -> 32 x H/4 x W/4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv
from .head import YOLOv8Head


class Proto(nn.Module):
    def __init__(self, c1, nm=32):
        super().__init__()
        self.cv1 = Conv(c1, c1, 3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c1, c1, 3)
        self.cv3 = Conv(c1, nm, 1, act=False)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))  # [B,nm,H*2,W*2]


class YOLOv8SegHead(YOLOv8Head):
    def __init__(self, num_classes=80, in_channels=(256, 256, 512), reg_max=16,
                 strides=(8, 16, 32), nm=32):
        super().__init__(num_classes, in_channels, reg_max, strides)
        self.nm = nm
        self.proto = Proto(in_channels[0], nm)
        self.mask_coeff = nn.ModuleList(
            nn.Conv2d(c, nm, 1) for c in in_channels
        )

    def forward(self, feats):
        cls_l, reg_l = super().forward(feats)
        mc_l = [mc(f) for mc, f in zip(self.mask_coeff, feats)]
        proto = self.proto(feats[0])
        return cls_l, reg_l, mc_l, proto

    def decode(self, feats):
        cls_list, reg_list, mc_list, proto = self.forward(feats)
        # reuse detect decode for boxes/scores
        boxes, scores = super().decode(feats)
        # flatten mask coeffs aligned with anchors
        mc_all = []
        for mc in mc_list:
            b, _, h, w = mc.shape
            mc_all.append(mc.view(b, self.nm, -1).permute(0, 2, 1))
        mc_all = torch.cat(mc_all, 1)  # [B,N,nm]
        return boxes, scores, mc_all, proto

    def predict(self, feats):
        boxes, scores, mc, proto = self.decode(feats)
        return torch.cat([boxes, scores], -1), mc, proto
