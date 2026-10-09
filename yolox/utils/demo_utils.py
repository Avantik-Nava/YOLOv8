#!/usr/bin/env python3
"""Demo utils compat (YOLOX yolox/utils/demo_utils.py port)."""
import cv2
import numpy as np


def letterbox_demo(img, size=640):
    from yolox.data import letterbox
    return letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)


def postprocess_demo(pred, conf=0.25, iou=0.7):
    from yolox.utils.boxes import non_max_suppression
    return non_max_suppression(pred, conf, iou)
