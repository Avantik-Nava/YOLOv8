"""YOLO-format dataset (Ultralytics layout) with letterbox + basic augments.

Layout:
  dataset/
    images/train/*.jpg  images/val/*.jpg
    labels/train/*.txt  labels/val/*.txt   (cls cx cy w h normalized)

Returns: img Tensor [3,H,W] 0..1, labels Tensor [M,5] (cls,cxcywh norm, padded -1 in collate).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    s = min(new_size / w, new_size / h)
    nw, nh = int(w * s), int(h * s)
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    dw, dh = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = img
    return canvas, s, (dw, dh)


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, augment=False,
                 hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flip_prob=0.5,
                 mosaic_prob=0.0, mixup_prob=0.0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.img_size = img_size
        self.augment = augment
        self.hsv_h, self.hsv_s, self.hsv_v = hsv_h, hsv_s, hsv_v
        self.flip_prob = flip_prob
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for e in exts:
            files += sorted(self.images_dir.glob(e))
            files += sorted(self.images_dir.glob(e.upper()))
        self.files = files
        print(f"Found {len(self.files)} images in {images_dir}")

    def __len__(self):
        return len(self.files)

    def _load_label(self, stem):
        p = self.labels_dir / f"{stem}.txt"
        if not p.exists():
            return np.zeros((0, 5), dtype=np.float32)
        rows = []
        with open(p) as f:
            for line in f:
                d = line.strip().split()
                if len(d) == 5:
                    rows.append([int(float(d[0])), float(d[1]), float(d[2]), float(d[3]), float(d[4])])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)

    def _load_raw(self, idx):
        img_path = self.files[idx % len(self.files)]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, self._load_label(img_path.stem)

    def __getitem__(self, idx):
        from .augment import augment_hsv, mosaic4, mixup

        if self.augment and len(self.files) >= 4 and np.random.rand() < self.mosaic_prob:
            samples = [self._load_raw(idx)] + \
                      [self._load_raw(np.random.randint(0, len(self.files))) for _ in range(3)]
            img, labels = mosaic4(samples, self.img_size)
            if self.augment and np.random.rand() < self.mixup_prob and len(self.files) >= 2:
                img2, lb2 = self._load_raw(np.random.randint(0, len(self.files)))
                img2, _, _ = letterbox(img2, self.img_size)
                img, labels = mixup(img, labels, img2, lb2)
        else:
            img, labels = self._load_raw(idx)
            img, _, _ = letterbox(img, self.img_size)

        if self.augment:
            if np.random.rand() < self.flip_prob:  # hflip
                img = np.fliplr(img).copy()
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
            img = augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)

        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        lb = torch.from_numpy(labels).float() if len(labels) else torch.zeros((0, 5))
        return img, lb


def yolo_collate(batch):
    """Pad labels to [B, max_M, 5] with cls=-1 for empty slots."""
    imgs, lbs = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_m = max([l.shape[0] for l in lbs] + [1])
    out = torch.full((len(lbs), max_m, 5), -1.0)
    for i, l in enumerate(lbs):
        if l.shape[0]:
            out[i, :l.shape[0]] = l
    return imgs, out
