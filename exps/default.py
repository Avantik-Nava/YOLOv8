"""Base experiment (YOLOX exps/ style, YOLOv8 hyper-params).

Mirror of YOLOX `exps/default/yolox_s.py` idea:
  python tools/train.py -e exps/yolov8_s.py
plus top-level config.yaml for simple `python tools/train.py --config config.yaml`.
"""

import os


class YOLOv8Exp:
    def __init__(self):
        # task: detect | segment | classify | pose | obb
        self.task = "detect"
        # model
        self.version = "s"          # n, s, m, l, x
        self.num_classes = 80       # COCO=80; your dataset: set to len(names)
        self.nkpt = 17              # pose keypoints
        self.nm = 32                # seg mask coefficients
        self.reg_max = 16           # YOLOv8 DFL bins
        self.strides = (8, 16, 32)
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # data (Ultralytics-style yaml, see data/custom.yaml)
        self.data_yaml = "data/custom.yaml"

        # training (Ultralytics YOLOv8 defaults blended with YOLOX schedule)
        self.epochs = 100           # docs example: 100 epochs on coco8
        self.batch_size = 16
        self.lr = 0.01
        self.min_lr_ratio = 0.0001  # cosine floor
        self.momentum = 0.937       # YOLOv8 SGD momentum
        self.weight_decay = 5e-4
        self.warmup_epochs = 3
        self.close_mosaic = 10      # YOLOv8: disable mosaic/mixup last N epochs
        self.ema = True

        # augmentation (YOLOv8)
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0       # YOLOv8 default mixup off for many sets
        self.hsv_h = 0.015
        self.hsv_s = 0.7
        self.hsv_v = 0.4
        self.flip_prob = 0.5
        self.degrees = 0.0
        self.translate = 0.1
        self.shear = 0.0

        # loss gains (Ultralytics defaults)
        self.box_gain = 7.5
        self.cls_gain = 0.5
        self.dfl_gain = 1.5
        self.tal_topk = 10

        # val / ckpt
        self.eval_interval = 1
        self.save_interval = 10
        self.save_dir = "YOLOv8_outputs"
        self.test_conf = 0.01
        self.nmsthre = 0.7
        self.print_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # device / train infra
        self.device = "cuda"  # tools fall back to cpu automatically
        self.amp = True             # mixed precision (cuda only)
        self.ddp = False            # torchrun --nproc_per_node=N (stub: single node)
        self.patience = 50          # early stopping epochs
        self.warmup_bias_lr = 0.1

    def get_model(self):
        from yolo.models.tasks import create_model
        kw = {}
        if self.task == "pose":
            kw["nkpt"] = getattr(self, "nkpt", 17)
        if self.task == "segment":
            kw["nm"] = getattr(self, "nm", 32)
        return create_model(self.task, self.version, self.num_classes,
                            reg_max=self.reg_max, **kw)

    def get_data(self):
        import yaml
        from pathlib import Path
        from torch.utils.data import DataLoader
        from yolo.data import YOLODataset, yolo_collate

        with open(self.data_yaml) as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        imgsz = self.input_size[0]
        train_ds = YOLODataset(root / d["train_images"] if "train_images" in d else root / "images/train",
                               root / d["train_labels"] if "train_labels" in d else root / "labels/train",
                               imgsz, augment=True, flip_prob=self.flip_prob)
        val_ds = YOLODataset(root / d["val_images"] if "val_images" in d else root / "images/val",
                             root / d["val_labels"] if "val_labels" in d else root / "labels/val",
                             imgsz, augment=False)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=yolo_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                collate_fn=yolo_collate, num_workers=0)
        return train_loader, val_loader, d
