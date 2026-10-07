#!/usr/bin/env python3
# YOLOv8-m experiment. Train: python tools/train.py -f exps/default/yolov8_m.py
#                 or: python tools/train.py -n yolov8-m
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
