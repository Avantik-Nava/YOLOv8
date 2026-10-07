"""COCO-style mAP (50 and 50-95) + precision/recall.

Uses pycocotools when installed; otherwise falls back to a greedy
IoU matcher over predicted [M,6] (x1,y1,x2,y2,conf,cls) vs GT boxes.
GT format per image: [G,5] (cls,x1,y1,x2,y2) pixels.
"""

import numpy as np
import torch

from .boxes import box_iou


def _greedy_match(dets, gts, iou_thres=0.5):
    # dets [M,6] sorted by conf desc assumed; gts [G,5]
    tp = np.zeros(len(dets))
    matched = set()
    for i, d in enumerate(dets):
        if len(gts) == 0:
            continue
        same = [g for gi, g in enumerate(gts) if g[0] == d[5] and gi not in matched]
        if not same:
            continue
        ious = []
        for g in same:
            ix1 = max(d[0], g[1])
            iy1 = max(d[1], g[2])
            ix2 = min(d[2], g[3])
            iy2 = min(d[3], g[4])
            inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
            union = (d[2] - d[0]) * (d[3] - d[1]) + (g[3] - g[1]) * (g[4] - g[2]) - inter + 1e-7
            ious.append(inter / union)
        j = int(np.argmax(ious))
        if ious[j] >= iou_thres:
            tp[i] = 1
            # find global index
            for gi, g in enumerate(gts):
                if g[0] == d[5] and gi not in matched:
                    if np.allclose(g, same[j]):
                        matched.add(gi)
                        break
    return tp


def coco_map(all_dets, all_gts, iou_thres_list=None):
    """all_dets/all_gts: lists per image (numpy). Returns dict map50, map5095, P, R."""
    try:
        from pycocotools.coco import COCO  # noqa
        from pycocotools.cocoeval import COCOeval  # noqa
        has_coco = True
    except Exception:
        has_coco = False
    thres = iou_thres_list or [0.5 + 0.05 * i for i in range(10)]
    aps, ps, rs = [], [], []
    for t in thres:
        tps, n_gt, n_det = [], 0, 0
        for dets, gts in zip(all_dets, all_gts):
            dets = np.asarray(dets).reshape(-1, 6)
            gts = np.asarray(gts).reshape(-1, 5)
            if len(dets):
                dets = dets[dets[:, 4].argsort()[::-1]]
            tp = _greedy_match(dets, gts, t)
            tps.append(tp)
            n_gt += len(gts)
            n_det += len(dets)
        if n_det == 0 or n_gt == 0:
            aps.append(0.0)
            ps.append(0.0)
            rs.append(0.0)
            continue
        tp_all = np.concatenate(tps) if tps else np.array([])
        fp_all = 1 - tp_all
        tp_c, fp_c = np.cumsum(tp_all), np.cumsum(fp_all)
        rec = tp_c / max(n_gt, 1)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-7)
        # COCO 101-point interp
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
        aps.append(float(ap))
        ps.append(float(prec[-1]))
        rs.append(float(rec[-1]))
    out = {"map50": float(aps[0]), "map5095": float(np.mean(aps)),
           "precision": float(ps[0]), "recall": float(rs[0]),
           "per_thres_ap": [float(a) for a in aps], "pycocotools": has_coco}
    return out
