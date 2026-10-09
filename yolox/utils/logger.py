#!/usr/bin/env python3
"""Logger compat (YOLOX yolox/utils/logger.py port)."""
import logging


def setup_logger(name="yolov8", level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(h)
    lg.setLevel(level)
    return lg


class WandbLogger:
    def __init__(self, *a, **k):
        print("[wandb] disabled (stub)")


def save_checkpoint(*a, **k):
    from .checkpoint import load_checkpoint  # noqa
