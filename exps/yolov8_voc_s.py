# encoding: utf-8
# YOLOv8 VOC experiment — CHANGE ONLY THIS FILE (num_classes / data_dir / version).
# Mirrors YOLOX exps/example/yolox_voc/yolox_voc_s.py
# Train: python tools/train.py -e exps/yolov8_voc_s.py
import os

from exps.default import YOLOv8Exp


class Exp(YOLOv8Exp):
    def __init__(self):
        super().__init__()
        # ===== EDIT HERE =====
        self.num_classes = 20          # len of VOC_CLASSES in yolo/data/datasets/voc_classes.py
        self.version = "s"             # n, s, m, l, x
        self.data_dir = "../VOCdevkit"  # path to VOCdevkit folder
        self.train_sets = (("2007", "trainval"), ("2012", "trainval"))
        self.val_sets = (("2007", "test"),)
        # =====================
        self.task = "detect"
        self.warmup_epochs = 1
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data(self):
        from torch.utils.data import DataLoader
        from yolo.data.datasets import VOCDetection
        from yolo.data import yolo_collate

        imgsz = self.input_size[0]
        train_ds = VOCDetection(
            data_dir=self.data_dir, image_sets=self.train_sets,
            img_size=imgsz, augment=True, flip_prob=self.flip_prob,
            hsv_h=self.hsv_h, hsv_s=self.hsv_s, hsv_v=self.hsv_v)
        val_ds = VOCDetection(
            data_dir=self.data_dir, image_sets=self.val_sets,
            img_size=self.test_size[0], augment=False)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  collate_fn=yolo_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                collate_fn=yolo_collate, num_workers=0)
        return train_loader, val_loader, {"nc": self.num_classes}
