"""Build a legacy YOLOX model matching a checkpoint (arch auto-detect)."""

import torch

from .backbone import YOLOPAFPN
from .head import YOLOXHead
from .model import YOLOX

CANDIDATES = [(0.33, 0.25), (0.33, 0.50), (0.67, 0.75), (1.00, 1.00), (1.00, 1.25)]


def _norm_ckpt(path):
    from yolox.utils import load_checkpoint
    sd = load_checkpoint(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def _has_dfl(sd):
    return any(k.startswith("head.dfl_proj") for k in sd)


def _score(sd, depth, width, nc, reg_max, act):
    try:
        m = YOLOX(YOLOPAFPN(depth, width, act=act),
                  YOLOXHead(nc, width, reg_max=reg_max, act=act))
    except Exception:
        return -1, None
    own = m.state_dict()
    hit = sum(1 for k, v in sd.items() if k in own and own[k].shape == v.shape)
    total = sum(1 for k in sd if k in own)
    return (hit / max(total, 1)), m


def build_legacy(weights_path, num_classes, depth_hint=0.33, width_hint=0.50, act="silu"):
    """Pick (depth,width) with best weight-shape coverage. Returns (model, info)."""
    sd = _norm_ckpt(weights_path)
    reg_max = 16 if _has_dfl(sd) else 0
    order = [(depth_hint, width_hint)] + [c for c in CANDIDATES if c != (depth_hint, width_hint)]
    best, best_m, best_dw = -1, None, order[0]
    for dw in order:
        s, m = _score(sd, dw[0], dw[1], num_classes, reg_max, act)
        if s > best:
            best, best_m, best_dw = s, m, dw
    own = best_m.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in sd if k not in keep and not k.startswith(("optimizer", "scheduler"))]
    best_m.load_state_dict(keep, strict=False)
    best_m.eval()
    info = {"depth": best_dw[0], "width": best_dw[1], "reg_max": reg_max,
            "coverage": round(best, 4), "skipped": len(skipped)}
    return best_m, info, skipped
