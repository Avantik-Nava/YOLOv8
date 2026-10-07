"""YOLOv8 anchor-free decoupled head with DFL.

Differences vs YOLOX/YOLOv26 head:
  - YOLOX head: cls + reg + obj (objectness) branches
  - YOLOv8 head: cls + reg(DFL) only, NO objectness branch, anchor-free

Each scale: 2x 3x3 Conv for cls, 2x 3x3 Conv for box,
  cls -> Conv1x1(num_classes), box -> Conv1x1(4 * reg_max).
Reference: https://docs.ultralytics.com/models/yolov8
"""

import math

import torch
import torch.nn as nn

from .common import Conv


class DFL(nn.Module):
    """Distribution Focal Loss integral: probs over reg_max bins -> distance."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        proj = torch.arange(reg_max, dtype=torch.float)
        self.register_buffer("proj", proj)

    def forward(self, x):
        # x: [B, 4, reg_max, N] -> [B, 4, N]
        b, _, _, n = x.shape
        x = x.softmax(2)
        proj = self.proj.to(x.dtype).to(x.device)
        return (x * proj.view(1, 1, -1, 1)).sum(2)


class YOLOv8Head(nn.Module):
    def __init__(self, num_classes=80, in_channels=(256, 256, 512), reg_max=16,
                 strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = list(strides)
        self.nc = num_classes
        self.no = num_classes + 4 * reg_max  # no obj branch (unlike YOLOX)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.dfl = DFL(reg_max) if reg_max > 1 else None

        for c in in_channels:
            self.cls_convs.append(nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)))
            self.reg_convs.append(nn.Sequential(Conv(c, c, 3), Conv(c, c, 3)))
            self.cls_preds.append(nn.Conv2d(c, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(c, 4 * reg_max, 1))

        self._init_bias()

    def _init_bias(self):
        # cls bias ~ low prior (like Ultralytics: -log((1-p)/p), p=0.01 -> -4.6)
        for cls_pred, reg_pred, s in zip(self.cls_preds, self.reg_preds, self.strides):
            nn.init.constant_(cls_pred.bias, -4.6)
            nn.init.constant_(reg_pred.bias, 1.0)

    def forward(self, feats):
        """Train mode: return raw per-scale (cls_logits, box_dist).

        Returns:
            cls_logits: list of [B, nc, H, W]
            box_dists:  list of [B, 4*reg_max, H, W]
        """
        cls_out, reg_out = [], []
        for x, cls_conv, reg_conv, cls_pred, reg_pred in zip(
            feats, self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds
        ):
            cls_out.append(cls_pred(cls_conv(x)))
            reg_out.append(reg_pred(reg_conv(x)))
        return cls_out, reg_out

    # ---------------- inference helpers ----------------
    def decode(self, feats):
        """Decode to [B, N, 4 + nc]: boxes (xyxy, pixels) + cls scores (sigmoid)."""
        # NOTE: call base forward explicitly so Seg/Pose/OBB subclasses
        # (which override forward to also return masks/kpts/angles) still decode boxes.
        cls_list, reg_list = YOLOv8Head.forward(self, feats)
        batch = cls_list[0].shape[0]
        device = cls_list[0].device

        boxes_all, scores_all = [], []
        for cls, reg, stride in zip(cls_list, reg_list, self.strides):
            b, _, h, w = cls.shape
            # anchors
            yv, xv = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            grid = torch.stack((xv, yv), 2).float()  # [H, W, 2]
            anchor = (grid.view(1, h, w, 2) + 0.5) * stride  # centers, pixels
            anchor = anchor.view(1, -1, 2)  # [1, H*W, 2]

            # DFL -> ltrb distances (in cells) -> * stride -> pixels
            n = h * w
            dist = reg.view(b, 4, self.reg_max, n)  # [B,4,reg_max,N]
            if self.dfl is not None:
                dist = self.dfl(dist)  # [B,4,N]
            else:
                dist = dist.mean(2)
            dist = dist.permute(0, 2, 1) * stride  # [B,N,4] ltrb pixels

            # dist2bbox center form
            cx, cy = anchor[..., 0], anchor[..., 1]
            l, t, r, bo = dist[..., 0], dist[..., 1], dist[..., 2], dist[..., 3]
            x1 = cx - l
            y1 = cy - t
            x2 = cx + r
            y2 = cy + bo
            boxes = torch.stack([x1, y1, x2, y2], -1)  # [B,N,4]

            scores = cls.view(b, self.num_classes, n).permute(0, 2, 1).sigmoid()
            boxes_all.append(boxes)
            scores_all.append(scores)

        return torch.cat(boxes_all, 1), torch.cat(scores_all, 1)

    def predict(self, feats):
        boxes, scores = self.decode(feats)
        return torch.cat([boxes, scores], -1)  # [B, N, 4+nc]
