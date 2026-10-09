#!/usr/bin/env python3
"""Samplers (YOLOX yolox/data/samplers.py port)."""
import itertools
from typing import Optional

import torch
from torch.utils.data.sampler import BatchSampler as torchBatchSampler
from torch.utils.data.sampler import Sampler


class YoloBatchSampler(torchBatchSampler):
    """Yields [(mosaic_flag, idx)] batches like YOLOX (mosaic on/off)."""

    def __init__(self, *args, mosaic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = 0):
        self.size = size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()
            else:
                yield from itertools.repeat(range(self.size))
