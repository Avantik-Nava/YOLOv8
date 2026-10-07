"""YOLOv8 PAN-FPN neck with C2f (Ultralytics style).

Top-down FPN + bottom-up PAN, all fusions followed by C2f.
Inputs : (P3 /8, P4 /16, P5 /32) from backbone
Outputs: (P3, P4, P5) same resolutions, same channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import C2f, Conv


class YOLOv8Neck(nn.Module):
    def __init__(self, in_channels, version="s", depth=None):
        """in_channels: (c3, c4, c5)."""
        super().__init__()
        from .backbone import _neck_blocks
        c3, c4, c5 = in_channels
        n = _neck_blocks(version, depth)

        # Top-down
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse4 = C2f(c5 + c4, c4, n, shortcut=False)
        self.fuse3 = C2f(c4 + c3, c3, n, shortcut=False)

        # Bottom-up
        self.down1 = Conv(c3, c3, 3, 2)
        self.fuse_p4 = C2f(c3 + c4, c4, n, shortcut=False)
        self.down2 = Conv(c4, c4, 3, 2)
        self.fuse_p5 = C2f(c4 + c5, c5, n, shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, feats):
        p3, p4, p5 = feats

        # FPN top-down: P5 -> P4 -> P3
        up_p5 = self.upsample(p5)
        if up_p5.shape[-2:] != p4.shape[-2:]:
            up_p5 = F.interpolate(up_p5, size=p4.shape[-2:], mode="nearest")
        f4 = self.fuse4(torch.cat([up_p5, p4], dim=1))

        up_f4 = self.upsample(f4)
        if up_f4.shape[-2:] != p3.shape[-2:]:
            up_f4 = F.interpolate(up_f4, size=p3.shape[-2:], mode="nearest")
        f3 = self.fuse3(torch.cat([up_f4, p3], dim=1))  # P3 out

        # PAN bottom-up: P3 -> P4 -> P5
        d3 = self.down1(f3)
        if d3.shape[-2:] != f4.shape[-2:]:
            d3 = F.adaptive_max_pool2d(d3, f4.shape[-2:])
        p4_out = self.fuse_p4(torch.cat([d3, f4], dim=1))

        d4 = self.down2(p4_out)
        if d4.shape[-2:] != p5.shape[-2:]:
            d4 = F.adaptive_max_pool2d(d4, p5.shape[-2:])
        p5_out = self.fuse_p5(torch.cat([d4, p5], dim=1))

        return [f3, p4_out, p5_out]
