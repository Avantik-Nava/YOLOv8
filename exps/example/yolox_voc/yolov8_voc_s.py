# encoding: utf-8
# YOLOv8 VOC experiment — CHANGE ONLY THIS FILE (num_classes / data_dir / depth / width).
# Mirrors YOLOX exps/example/yolox_voc/yolox_voc_s.py
# Train: python tools/train.py -f exps/example/yolox_voc/yolov8_voc_s.py
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ===== EDIT HERE =====
        self.num_classes = 3           # len of VOC_CLASSES in yolox/data/datasets/voc_classes.py
        self.depth = 0.33
        self.width = 0.50
        self.data_dir = "datasets/VOCdevkit"  # YOLOX-style: VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets}
        self.train_sets = (("2007", "trainval"),)
        self.val_sets = (("2007", "test"),)
        # =====================
        self.warmup_epochs = 1
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data.datasets import VOCDetection

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.train_sets,
            img_size=self.input_size[0],
            augment=True,
            flip_prob=self.flip_prob,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data.datasets import VOCDetection

        return VOCDetection(
            data_dir=self.data_dir,
            image_sets=self.val_sets,
            img_size=self.test_size[0],
            augment=False,
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
