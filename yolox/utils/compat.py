#!/usr/bin/env python3
"""compat (YOLOX port)."""
import torch


def meshgrid(*args, **kwargs):
    return torch.meshgrid(*args, **kwargs)
