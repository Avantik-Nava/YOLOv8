#!/usr/bin/env python3
# YOLOX-style LRScheduler (warmup + cosine), YOLOv8 defaults.
# Mirrors yolox.utils.lr_scheduler.LRScheduler API: update_lr(num_iter).

import math


class LRScheduler:
    def __init__(self, scheduler_name, lr, iters_per_epoch, total_epochs,
                 warmup_epochs=5, warmup_lr_start=0.0, no_aug_epochs=15,
                 min_lr_ratio=0.05):
        self.scheduler_name = scheduler_name
        self.lr = lr
        self.iters_per_epoch = max(iters_per_epoch, 1)
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.no_aug_epochs = no_aug_epochs
        self.min_lr_ratio = min_lr_ratio
        self.warmup_iters = warmup_epochs * self.iters_per_epoch
        self.no_aug_iters = no_aug_epochs * self.iters_per_epoch
        self.total_iters = total_epochs * self.iters_per_epoch

    def _warm_cos_lr(self, num_iter):
        if num_iter <= self.warmup_iters:
            alpha = num_iter / max(self.warmup_iters, 1)
            return self.warmup_lr_start + (self.lr - self.warmup_lr_start) * alpha
        total = self.total_iters - self.no_aug_iters
        num_iter = min(num_iter, self.total_iters)
        alpha = (num_iter - self.warmup_iters) / max(total - self.warmup_iters, 1)
        alpha = min(max(alpha, 0.0), 1.0)
        return self.lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * alpha)))

    def _cos_lr(self, num_iter):
        alpha = min(max(num_iter / max(self.total_iters, 1), 0.0), 1.0)
        return self.lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * alpha)))

    def update_lr(self, num_iter, optimizer=None):
        if self.scheduler_name == "yoloxwarmcos":
            lr = self._warm_cos_lr(num_iter)
        else:
            lr = self._cos_lr(num_iter)
        if optimizer is not None:
            for g in optimizer.param_groups:
                g["lr"] = lr
        return lr
