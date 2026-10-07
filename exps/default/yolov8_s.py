#!/usr/bin/env python3
# YOLOv8-s experiment. Train: python tools/train.py -f exps/default/yolov8_s.py
#                 or: python tools/train.py -n yolov8-s
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
