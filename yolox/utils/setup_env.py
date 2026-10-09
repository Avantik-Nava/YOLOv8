#!/usr/bin/env python3
"""Setup-env compat (YOLOX yolox/utils/setup_env.py port)."""
import os
import random

import numpy as np
import torch


def configure_omp(num_threads=None):
    n = num_threads or os.cpu_count()
    torch.set_num_threads(n)


def configure_nccl():
    pass


def configure_module():
    pass


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
