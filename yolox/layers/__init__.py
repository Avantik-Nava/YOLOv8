#!/usr/bin/env python3
"""Layers compat (YOLOX yolox/layers/ port).

YOLOX layers (fast_coco_eval_api, jit_ops) are YOLOX-arch helpers.
YOLOv8 uses C2f/DFL/TAL instead — this module exposes the same names
so `from yolox.layers import ...` keeps working.
"""
from yolox.utils.boxes import *  # noqa

__all__ = []
