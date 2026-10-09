#!/usr/bin/env python3
"""Model utils compat (YOLOX yolox/utils/model_utils.py port)."""
import torch
import torch.nn as nn


def fuse_model(model):
    return model


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False
