#!/usr/bin/env python3
# YOLOv8-l experiment. Train: python tools/train.py -f exps/default/yolov8_l.py
#                 or: python tools/train.py -n yolov8-l
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.00
        self.width = 1.00
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
