#!/usr/bin/env python3
"""DataLoader + prefetcher (YOLOX yolox/data/dataloading.py + data_prefetcher.py port)."""
import os

import torch
from torch.utils.data import DataLoader as torchDataLoader


def get_yolox_datadir():
    return os.environ.get("YOLOX_DATADIR", "./datasets")


class DataLoader(torchDataLoader):
    pass


class DataPrefetcher:
    """CUDA prefetcher; CPU fallback returns batch as-is (like YOLOX API)."""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def next(self):
        try:
            batch = next(self.loader)
        except StopIteration:
            return None, None
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device, non_blocking=True) if torch.is_tensor(b) else b
                    for b in batch]
        return batch.to(self.device), None
