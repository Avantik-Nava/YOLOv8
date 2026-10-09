#!/usr/bin/env python3
"""Visualize compat (YOLOX yolox/utils/visualize.py port)."""
import cv2


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    out = img.copy()
    for b, s, c in zip(boxes, scores, cls_ids):
        if s < conf:
            continue
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[int(c)] if class_names else int(c)}:{s:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out
