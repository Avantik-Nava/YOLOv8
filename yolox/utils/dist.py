#!/usr/bin/env python3
"""Dist compat (YOLOX yolox/utils/dist.py port). Single-process fallback."""
import torch


def get_world_size():
    return 1


def get_rank():
    return 0


def is_main_process():
    return True


def synchronize():
    pass


def all_reduce(tensor):
    return tensor
