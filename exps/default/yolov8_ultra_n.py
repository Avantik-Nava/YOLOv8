#!/usr/bin/env python3
# YOLOv8-n via REAL ultralytics backend, YOLOX flow.
# Train: python tools/train.py -f exps/default/yolov8_ultra_n.py -b 4
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.backend = "ultra"  # real `ultralytics` pkg, no YOLOX features
        self.task = "detect"
        self.version = "n"
        self.num_classes = 3  # overridden by data/custom.yaml nc
        self.depth = 0.33
        self.width = 0.25
        self.data_yaml = "data/custom.yaml"
        self.max_epoch = 50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
