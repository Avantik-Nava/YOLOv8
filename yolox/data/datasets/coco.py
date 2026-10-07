#!/usr/bin/env python3
# COCO json dataset — mirrors YOLOX yolox/data/datasets/coco.py API
# (COCODataset(data_dir, json_file, ...)), returns YOLOv8 tensors:
#   img Tensor [3,H,W] 0..1 (letterboxed), labels Tensor [M,5] (cls,cx,cy,w,h) norm.

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco_classes import COCO_CLASSES
from ..dataset import letterbox


class COCODataset(Dataset):
    def __init__(self, data_dir, json_file, img_size=640, augment=False,
                 name="train2017", mosaic_prob=0.0, flip_prob=0.5,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.flip_prob = flip_prob
        self.hsv = (hsv_h, hsv_s, hsv_v)
        with open(os.path.join(data_dir, "annotations", json_file)) as f:
            coco = json.load(f)
        # map original 90-class ids -> 0..79
        cat_ids = [c["id"] for c in coco["categories"]]
        self.id2cls = {cid: i for i, cid in enumerate(sorted(cat_ids)[:80])}
        self.imgs = {im["id"]: im for im in coco["images"]}
        anns = {}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0):
                continue
            anns.setdefault(a["image_id"], []).append(a)
        self.ids = [i for i in self.imgs if i in anns]
        self.anns = anns
        print(f"COCODataset ({name}): {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def _load_raw(self, idx):
        img_id = self.ids[idx % len(self.ids)]
        info = self.imgs[img_id]
        img = cv2.imread(str(self.data_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if img is None:  # try images/ subfolder layout
            img = cv2.imread(str(self.data_dir / "images" / info["file_name"]), cv2.IMREAD_COLOR)
        assert img is not None, f"missing {info['file_name']}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        rows = []
        for a in self.anns[img_id]:
            if a["category_id"] not in self.id2cls:
                continue
            x, y, bw, bh = a["bbox"]
            rows.append([self.id2cls[a["category_id"]],
                         (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h])
        labels = np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)
        return img, labels

    def __getitem__(self, idx):
        from ..augment import augment_hsv, mosaic4
        if self.augment and len(self.ids) >= 4 and np.random.rand() < self.mosaic_prob:
            samples = [self._load_raw(idx)] + \
                      [self._load_raw(np.random.randint(0, len(self.ids))) for _ in range(3)]
            img, labels = mosaic4(samples, self.img_size)
        else:
            img, labels = self._load_raw(idx)
            img, _, _ = letterbox(img, self.img_size)
        if self.augment:
            if np.random.rand() < self.flip_prob:
                img = np.fliplr(img).copy()
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
            img = augment_hsv(img, *self.hsv)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        lb = torch.from_numpy(labels).float() if len(labels) else torch.zeros((0, 5))
        return img, lb
