#!/usr/bin/env python3
# Exact YOLOX BaseExp API (yolox/exp/base_exp.py) for the YOLOv8 port.

import ast
import pprint
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

import torch
from torch.nn import Module

from yolox.utils import LRScheduler


class BaseExp(metaclass=ABCMeta):
    """Basic class for any experiment."""

    def __init__(self):
        self.seed = None
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 100
        self.eval_interval = 10
        self.dataset = None

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        pass

    @abstractmethod
    def get_data_loader(
        self, batch_size: int, is_distributed: bool
    ) -> Dict[str, torch.utils.data.DataLoader]:
        pass

    @abstractmethod
    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(
        self, lr: float, iters_per_epoch: int, **kwargs
    ) -> LRScheduler:
        pass

    @abstractmethod
    def get_evaluator(self):
        pass

    @abstractmethod
    def eval(self, model, evaluator, weights):
        pass

    def __repr__(self):
        try:
            from tabulate import tabulate
            table_header = ["keys", "values"]
            exp_table = [
                (str(k), pprint.pformat(v))
                for k, v in vars(self).items()
                if not k.startswith("_")
            ]
            return tabulate(exp_table, headers=table_header, tablefmt="grid")
        except ImportError:
            return pprint.pformat(vars(self))

    def merge(self, cfg_list):
        if not cfg_list:
            return
        # Accept both YOLOX style [k, v, k, v] and Ultralytics style [k=v, ...]
        flat = []
        for item in cfg_list:
            if isinstance(item, str) and "=" in item and item.startswith("-") is False:
                # split k=v -> k, v (only on first =)
                k, v = item.split("=", 1)
                flat += [k.strip(), v.strip()]
            else:
                flat.append(item)
        assert len(flat) % 2 == 0, f"length must be even, check value here: {cfg_list}"
        for k, v in zip(flat[0::2], flat[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)

                # pre-process input if source type is list or tuple
                if isinstance(src_value, (List, Tuple)):
                    v = v.strip("[]()")
                    v = [t.strip() for t in v.split(",")]

                    # find type of tuple
                    if len(src_value) > 0:
                        src_item_type = type(src_value[0])
                        v = [src_item_type(t) for t in v]

                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
