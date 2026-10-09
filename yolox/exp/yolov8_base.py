#!/usr/bin/env python3
# YOLOv8 base experiment — exact YOLOX Exp API (get_model/get_dataset/
# get_data_loader/get_optimizer/get_lr_scheduler/get_evaluator/eval),
# YOLOv8 hyper-parameters (anchor-free DFL head, TAL, close-mosaic recipe).

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp

__all__ = ["Exp"]


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.backend = "native"   # native (dependency-free reimpl) | ultra (real ultralytics pkg)
        self.task = "detect"      # detect | segment | classify | pose | obb
        self.num_classes = 80
        # factor of model depth / width (YOLOX-style; replaces version string)
        self.depth = 0.33         # s: n/s=0.33, m=0.67, l/x=1.0
        self.width = 0.50         # s: n=0.25, s=0.50, m=0.75, l=1.0, x=1.25
        self.version = "s"        # shorthand; depth/width win when both differ
        self.act = "silu"
        self.reg_max = 16         # YOLOv8 DFL bins
        self.strides = (8, 16, 32)
        self.nkpt = 17            # pose keypoints
        self.nm = 32              # seg mask coefficients

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        self.multiscale_range = 5
        self.data_dir = None          # VOCdevkit / COCO root for json datasets
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.data_yaml = "data/custom.yaml"  # YOLO-format fallback
        self.train_sets = (("2007", "trainval"), ("2012", "trainval"))
        self.val_sets = (("2007", "test"),)

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        self.shear = 0.0

        # --------------  training config --------------------- #
        self.warmup_epochs = 3
        self.max_epoch = 100          # YOLOv8 docs example: 100 epochs
        self.epochs = 100             # alias kept for older tools
        self.warmup_lr = 0
        self.min_lr_ratio = 0.0001
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 10       # close mosaic/mixup in last N epochs
        self.ema = True
        self.amp = True
        self.patience = 50            # early stopping
        self.weight_decay = 5e-4
        self.momentum = 0.937
        # loss gains (Ultralytics defaults)
        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5
        self.tal_topk = 10
        self.tal_alpha = 0.5
        self.tal_beta = 6.0
        self.print_interval = 10
        self.eval_interval = 1
        self.save_interval = 10
        self.save_history_ckpt = True
        self.save_dir = "YOLOX_outputs"  # alias of output_dir
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.01
        self.nmsthre = 0.7

    def merge(self, cfg_list):
        if not cfg_list:
            return
        super().merge(cfg_list)
        flat = " ".join(cfg_list) if isinstance(cfg_list, list) else str(cfg_list)
        if "max_epoch" in flat:
            self.epochs = self.max_epoch
        if "epochs" in flat:
            self.max_epoch = self.epochs

    # ---------------- model ---------------- #
    def get_model(self):
        from yolox.models.tasks import create_model

        kw = {}
        if self.task == "pose":
            kw["nkpt"] = self.nkpt
        if self.task == "segment":
            kw["nm"] = self.nm
        model = create_model(
            self.task, self.version, self.num_classes,
            reg_max=self.reg_max, depth=self.depth, width=self.width, **kw)
        model.train()
        return model

    # ---------------- dataset ---------------- #
    def _yolo_dirs(self):
        import yaml
        from pathlib import Path
        with open(self.data_yaml) as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        return root, d

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """COCO json when data_dir+annotations exist, else YOLO-format yaml."""
        from pathlib import Path
        task = getattr(self, "task", "detect")
        if self.data_dir is not None:
            ann = Path(self.data_dir) / "annotations" / self.train_ann
            if ann.exists():
                from yolox.data.datasets import COCODataset
                return COCODataset(
                    data_dir=self.data_dir, json_file=self.train_ann,
                    img_size=self.input_size[0], augment=True,
                    mosaic_prob=self.mosaic_prob, flip_prob=self.flip_prob,
                    degrees=self.degrees, translate=self.translate, shear=self.shear)
        from yolox.data import YOLODataset
        root, d = self._yolo_dirs()
        imgsz = self.input_size[0]
        ti = root / d.get("train_images", "images/train")
        tl = root / d.get("train_labels", "labels/train")
        if task == "segment":
            from yolox.data import SegDataset
            return SegDataset(ti, tl, imgsz, augment=True, flip_prob=self.flip_prob)
        if task == "pose":
            from yolox.data import PoseDataset
            return PoseDataset(ti, tl, imgsz, augment=True, flip_prob=self.flip_prob)
        if task == "classify":
            from yolox.data import ClsDataset
            return ClsDataset(root, imgsz)
        return YOLODataset(
            ti, tl, imgsz, augment=True, flip_prob=self.flip_prob,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob,
            degrees=self.degrees, translate=self.translate, shear=self.shear)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        from yolox.data import yolo_collate

        if self.dataset is None:
            self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        if hasattr(self.dataset, "augment"):
            self.dataset.augment = not no_aug
        if hasattr(self.dataset, "mosaic_prob") and no_aug:
            self.dataset.mosaic_prob = 0.0
            self.dataset.mixup_prob = 0.0

        collate = yolo_collate
        if getattr(self, "task", "detect") == "segment":
            from yolox.data import seg_collate as collate
        elif getattr(self, "task", "detect") == "pose":
            from yolox.data import pose_collate as collate
        elif getattr(self, "task", "detect") == "classify":
            collate = None

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=(sampler is None),
            sampler=sampler, num_workers=self.data_num_workers,
            pin_memory=True, collate_fn=collate)

    def get_data(self):
        """Compat wrapper (older tools): returns train/val loaders + meta."""
        train_loader = self.get_data_loader(self.batch_size if hasattr(self, "batch_size") else 16,
                                            False)
        return train_loader, train_loader, {"nc": self.num_classes}

    # ---------------- optimizer / scheduler ---------------- #
    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.warmup_lr if self.warmup_epochs > 0 else self.basic_lr_per_img * batch_size
            pg0, pg1, pg2 = [], [], []
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)
            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay})
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer
        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler
        return LRScheduler(
            self.scheduler, lr, iters_per_epoch, self.max_epoch,
            warmup_epochs=self.warmup_epochs, warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs, min_lr_ratio=self.min_lr_ratio)

    # ---------------- eval ---------------- #
    def get_eval_dataset(self, **kwargs):
        from yolox.data import YOLODataset
        root, d = self._yolo_dirs()
        return YOLODataset(
            root / d.get("val_images", "images/val"),
            root / d.get("val_labels", "labels/val"),
            self.test_size[0], augment=False)

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        from yolox.data import yolo_collate
        valdataset = self.get_eval_dataset(**kwargs)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(
            valdataset, batch_size=batch_size, sampler=sampler,
            num_workers=self.data_num_workers, pin_memory=True,
            collate_fn=yolo_collate)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size, confthre=self.test_conf,
            nmsthre=self.nmsthre, num_classes=self.num_classes)

    def get_trainer(self, args):
        from yolox.core import Trainer
        return Trainer(self, args)

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, is_distributed, half, return_outputs=return_outputs)
