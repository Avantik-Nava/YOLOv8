"""YOLOv8-cls head: global-pooled classifier (yolov8n-cls.pt etc)."""

import torch.nn as nn

from .common import Conv


class YOLOv8ClsHead(nn.Module):
    def __init__(self, num_classes=1000, in_channels=512, hidden=1280, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, feats):
        # feats: [P3,P4,P5] -> use P5
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats
        x = self.pool(x).flatten(1)
        return self.fc2(self.act(self.fc1(self.drop(x))))  # logits [B,nc]
