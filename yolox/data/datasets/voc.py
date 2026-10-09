#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOv8 VOC dataset — CHANGE ONLY data_dir/image_sets here if needed.
# Mirrors YOLOX yolox/data/datasets/voc.py but returns YOLOv8 tensors:
#   img Tensor [3,H,W] 0..1 (letterboxed), labels Tensor [M,5] (cls,cx,cy,w,h) normalized.
# VOCdevkit layout:
#   VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets/Main/*.txt}

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .voc_classes import VOC_CLASSES
from ..dataset import letterbox


class AnnotationTransform:
    """XML -> [xmin,ymin,xmax,ymax,label] pixels + (h,w)."""

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            diff = obj.find("difficult")
            difficult = int(diff.text) == 1 if diff is not None else False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            if name not in self.class_to_ind:
                continue
            bbox = obj.find("bndbox")
            bndbox = [int(float(bbox.find(pt).text)) - 1 for pt in ("xmin", "ymin", "xmax", "ymax")]
            bndbox.append(self.class_to_ind[name])
            res = np.vstack((res, bndbox))
        w = int(target.find("size").find("width").text)
        h = int(target.find("size").find("height").text)
        return res, (h, w)


class VOCDetection(Dataset):
    def __init__(self, data_dir, image_sets=(("2007", "trainval"),), img_size=640,
                 augment=False, target_transform=None, flip_prob=0.5,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                 mosaic_prob=0.0, mixup_prob=0.0,
                 degrees=0.0, translate=0.1, shear=0.0):
        self.root = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.flip_prob = flip_prob
        self.hsv = (hsv_h, hsv_s, hsv_v)
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.target_transform = target_transform or AnnotationTransform()
        self._annopath = str(self.root / "VOC%s" / "Annotations" / "%s.xml")
        self._imgpath = str(self.root / "VOC%s" / "JPEGImages" / "%s.jpg")
        self.ids = []
        for year, name in image_sets:
            txt = self.root / f"VOC{year}" / "ImageSets" / "Main" / f"{name}.txt"
            with open(txt) as f:
                for line in f:
                    self.ids.append((year, line.strip()))
        print(f"VOCDetection: {len(self.ids)} images from {data_dir} sets={image_sets}")

    def __len__(self):
        return len(self.ids)

    def _load(self, index):
        year, img_id = self.ids[index % len(self.ids)]
        target = ET.parse(self._annopath % (year, img_id)).getroot()
        res, (h, w) = self.target_transform(target)
        img = cv2.imread(self._imgpath % (year, img_id), cv2.IMREAD_COLOR)
        assert img is not None, f"missing {self._imgpath % (year, img_id)}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # to normalized cxcywh
        labels = np.zeros((0, 5), np.float32)
        if len(res):
            boxes = res[:, :4]
            cls = res[:, 4:5]
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2 / w)
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2 / h)
            bw = (boxes[:, 2] - boxes[:, 0]) / w
            bh = (boxes[:, 3] - boxes[:, 1]) / h
            labels = np.concatenate([cls, cx[:, None], cy[:, None], bw[:, None], bh[:, None]], 1).astype(np.float32)
        return img, labels

    def __getitem__(self, index):
        from ..augment import augment_hsv, mosaic4, mixup, random_affine

        if self.augment and len(self.ids) >= 4 and np.random.rand() < self.mosaic_prob:
            samples = [self._load(index)] + \
                      [self._load(np.random.randint(0, len(self.ids))) for _ in range(3)]
            img, labels = mosaic4(samples, self.img_size)
            if np.random.rand() < self.mixup_prob and len(self.ids) >= 2:
                img2, lb2 = self._load(np.random.randint(0, len(self.ids)))
                img2, _, _ = letterbox(img2, self.img_size)
                img, labels = mixup(img, labels, img2, lb2)
        else:
            img, labels = self._load(index)
            img, _, _ = letterbox(img, self.img_size)
        if self.augment:
            img, labels = random_affine(img, labels, degrees=self.degrees,
                                        translate=self.translate, shear=self.shear)
            if np.random.rand() < self.flip_prob:
                img = np.fliplr(img).copy()
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
            img = augment_hsv(img, *self.hsv)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        lb = torch.from_numpy(labels).float() if len(labels) else torch.zeros((0, 5))
        return img, lb
