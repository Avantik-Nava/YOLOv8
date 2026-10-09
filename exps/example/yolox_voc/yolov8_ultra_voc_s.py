#!/usr/bin/env python3
# PRIMARY exp: real ultralytics backend trained ONLY from datasets/VOCdevkit.
# Train: python tools/train.py -f exps/example/yolox_voc/yolov8_ultra_voc_s.py -b 4
# Backup (not used here): datasets/COCO + dataset/ YOLO-txt stay as-is.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ===== PRIMARY SOURCE (only this is used) =====
        self.backend = "ultra"  # real `ultralytics` pkg
        self.data_dir = "datasets/VOCdevkit"  # VOC2007/{JPEGImages,Annotations,ImageSets}
        self.train_sets = (("2007", "trainval"),)  # 180 imgs
        self.val_sets = (("2007", "test"),)        # 20 imgs
        # ==============================================
        self.task = "detect"
        self.version = "s"
        self.num_classes = 3  # YOUR CLASSES: len(VOC_CLASSES) — change here, not CLI
        self.depth = 0.33
        self.width = 0.50
        self.max_epoch = 100  # YOUR EPOCHS: change here, not CLI
        self.epochs = 100     # alias kept in sync
        self.batch_size = 4   # YOUR BATCH: used when -b not passed
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data.datasets import VOCDetection
        return VOCDetection(
            data_dir=self.data_dir, image_sets=self.train_sets,
            img_size=self.input_size[0], augment=True,
            flip_prob=self.flip_prob, mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob, degrees=self.degrees,
            translate=self.translate, shear=self.shear)

    def get_eval_dataset(self, **kwargs):
        from yolox.data.datasets import VOCDetection
        return VOCDetection(
            data_dir=self.data_dir, image_sets=self.val_sets,
            img_size=self.test_size[0], augment=False)

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator
        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size, confthre=self.test_conf,
            nmsthre=self.nmsthre, num_classes=self.num_classes)
