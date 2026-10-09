#!/usr/bin/env python3
# YOLOv8-s via REAL ultralytics backend, YOLOX flow.
# Train: python tools/train.py -f exps/default/yolov8_ultra_s.py -b 16
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.backend = "ultra"  # real `ultralytics` pkg, no YOLOX features
        self.task = "detect"
        self.version = "s"
        self.num_classes = 3
        self.depth = 0.33
        self.width = 0.50
        self.data_yaml = "data/custom.yaml"
        self.max_epoch = 100
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
