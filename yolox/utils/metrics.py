"""Detection metrics: precision/recall/AP helpers."""

import torch

from .boxes import box_iou


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-7):
    """Simple mAP helper (all-class). Returns dict with mAP50 estimate."""
    if len(tp) == 0:
        return {"map50": 0.0, "precision": 0.0, "recall": 0.0}
    order = torch.argsort(conf, descending=True)
    tp = tp[order]
    fp = 1 - tp
    tp_c = torch.cumsum(tp, 0)
    fp_c = torch.cumsum(fp, 0)
    n_gt = max(int((target_cls >= 0).sum()), 1)
    recall = tp_c / n_gt
    precision = tp_c / (tp_c + fp_c + eps)
    # 11-point-ish area
    ap = torch.trapz(precision, recall).item() if len(recall) > 1 else 0.0
    return {"map50": ap, "precision": float(precision[-1]), "recall": float(recall[-1])}
