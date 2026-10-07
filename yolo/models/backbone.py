"""YOLOv8 CSPDarknet-C2f backbone (Ultralytics YOLOv8 style).

Outputs P3 (stride 8), P4 (stride 16), P5 (stride 32).
Reference: https://docs.ultralytics.com/models/yolov8

Scales (depth=block repeats, width=channels) — same convention as Ultralytics:
  n: d=0.33 w=0.25 | s: d=0.33 w=0.50 | m: d=0.67 w=0.75
  l: d=1.00 w=1.00 | x: d=1.00 w=1.25
"""

import torch.nn as nn

from .common import C2f, Conv, SPPF


# version -> (base_channels [stem0, stem1, s1, s2(P3), s3(P4), s4(P5)],
#             block repeats [n1, n2, n3, n4])
VERSIONS = {
    # [3, 16, 32, 64, 128, 256] matches YOLOv26-n channel plan for familiarity
    "n": ([3, 16, 32, 64, 128, 256], [1, 2, 2, 1]),
    "s": ([3, 32, 64, 128, 256, 512], [1, 2, 2, 1]),
    "m": ([3, 48, 96, 192, 384, 768], [2, 4, 4, 2]),
    "l": ([3, 64, 128, 256, 512, 1024], [3, 6, 6, 3]),
    "x": ([3, 80, 160, 320, 640, 1280], [3, 6, 6, 3]),
}

# Ultralytics depth/width factors (for exps/ + docs)
DEPTH_WIDTH = {
    "n": (0.33, 0.25),
    "s": (0.33, 0.50),
    "m": (0.67, 0.75),
    "l": (1.00, 1.00),
    "x": (1.00, 1.25),
}


class CSPDarknetC2f(nn.Module):
    """YOLOv8 backbone. Returns (P3 /8, P4 /16, P5 /32)."""

    def __init__(self, version="s"):
        super().__init__()
        if version not in VERSIONS:
            raise ValueError(f"Unknown version '{version}', choose from {list(VERSIONS)}")
        ch, n = VERSIONS[version]
        self.version = version

        self.stem = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2),  # /2
            Conv(ch[1], ch[2], 3, 2),  # /4
        )
        self.stage1 = C2f(ch[2], ch[3], n[0], shortcut=True)

        self.down1 = Conv(ch[3], ch[3], 3, 2)  # /8
        self.stage2 = C2f(ch[3], ch[4], n[1], shortcut=True)  # P3

        self.down2 = Conv(ch[4], ch[4], 3, 2)  # /16
        # P4 keeps ch[4] width; P5 expands to ch[5] after /32 downsample
        c_p4 = ch[4]
        c_p5 = ch[5]
        self.stage3 = C2f(ch[4], c_p4, n[2], shortcut=True)  # P4
        # For P5 we first downsample P4 then expand to c_p5:
        self.down3 = Conv(c_p4, c_p4, 3, 2)  # /32
        self.stage4 = C2f(c_p4, c_p5, n[3], shortcut=True)
        self.sppf = SPPF(c_p5, c_p5)

        self.out_channels = (ch[4], c_p4, c_p5)

    def forward(self, x):
        x = self.stem(x)        # /4
        x = self.stage1(x)      # /4
        x = self.down1(x)       # /8
        p3 = self.stage2(x)     # /8  -> P3
        x = self.down2(p3)      # /16
        p4 = self.stage3(x)     # /16 -> P4
        x = self.down3(p4)      # /32
        x = self.stage4(x)
        p5 = self.sppf(x)       # /32 -> P5
        return p3, p4, p5
