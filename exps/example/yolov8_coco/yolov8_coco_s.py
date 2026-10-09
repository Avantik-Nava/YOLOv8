# encoding: utf-8
# YOLOv8 COCO experiment — exact YOLOX default flow (COCO json).
# Mirrors YOLOX exps/default/yolox_s.py (COCODataset via data_dir).
# Train: python tools/train.py -f exps/example/yolov8_coco/yolov8_coco_s.py
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ===== EDIT HERE =====
        self.num_classes = 3           # 3: curtain_open, feed_machine, bulb_off
        self.depth = 0.33
        self.width = 0.50
        self.data_dir = "datasets/COCO"  # YOLOX-style: annotations/instances_{train,val}2017.json
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # =====================
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data.datasets import COCODataset

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size[0],
            augment=True,
            mosaic_prob=self.mosaic_prob,
            flip_prob=self.flip_prob,
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data.datasets import COCODataset

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size[0],
            augment=False,
            name="val2017",
        )
