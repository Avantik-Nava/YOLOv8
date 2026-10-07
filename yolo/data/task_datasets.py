"""Task datasets: Seg / Cls / Pose / OBB (detect stays in dataset.py).

All reuse YOLO txt layout; extra files:
  seg: labels/*.txt may append polygon points OR masks/ dir with PNGs
  cls: images organized class-subfolders OR labels txt with single int
  pose: labels/*.txt rows cls cx cy w h [k1x k1y k1v ...] normalized
  obb: labels/*.txt rows cls cx cy w h theta(-pi/2..pi/2, radians or degrees+flag)
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import YOLODataset, letterbox, yolo_collate


class SegDataset(YOLODataset):
    def __init__(self, *a, mask_size=160, **k):
        super().__init__(*a, **k)
        self.mask_size = mask_size

    def __getitem__(self, idx):
        img, lb = super().__getitem__(idx)  # img tensor, lb [M,5]
        # dummy masks aligned to labels (real polygon rasterization = user hook)
        m = len(lb)
        masks = torch.zeros((m, self.mask_size, self.mask_size))
        return img, lb, masks


def seg_collate(batch):
    imgs, lbs, masks = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_m = max([l.shape[0] for l in lbs] + [1])
    lb_out = torch.full((len(lbs), max_m, 5), -1.0)
    ms_out = torch.zeros((len(lbs), max_m, masks[0].shape[-2], masks[0].shape[-1]))
    for i, (l, m) in enumerate(zip(lbs, masks)):
        if l.shape[0]:
            lb_out[i, :l.shape[0]] = l
            ms_out[i, :l.shape[0]] = m
    return imgs, lb_out, ms_out


class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, root, img_size=224):
        self.root = Path(root)
        self.img_size = img_size
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.files, self.targets = [], []
        if any((self.root / d).is_dir() for d in ["train", "val"]):
            for split in ["train", "val"]:
                for ci, cdir in enumerate(sorted((self.root / split).iterdir())):
                    if cdir.is_dir():
                        for e in exts:
                            for f in cdir.glob(e):
                                self.files.append(f)
                                self.targets.append(ci)
        else:  # flat + labels txt
            for e in exts:
                self.files += sorted(self.root.glob(f"images/*/{e}"))
            self.targets = [0] * len(self.files)
        print(f"Found {len(self.files)} classification images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _ = letterbox(img, self.img_size)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img, torch.tensor(self.targets[idx] if idx < len(self.targets) else 0)


class PoseDataset(YOLODataset):
    def __init__(self, *a, nkpt=17, **k):
        super().__init__(*a, **k)
        self.nkpt = nkpt

    def _load_label(self, stem):
        p = self.labels_dir / f"{stem}.txt"
        rows, kpts = [], []
        if p.exists():
            with open(p) as f:
                for line in f:
                    d = line.strip().split()
                    if len(d) >= 5:
                        rows.append([int(float(d[0])), *map(float, d[1:5])])
                        nk = (len(d) - 5) // 3
                        k = np.zeros((self.nkpt, 3), np.float32)
                        for i in range(min(nk, self.nkpt)):
                            k[i] = [float(d[5 + 3 * i]), float(d[6 + 3 * i]), float(d[7 + 3 * i])]
                        kpts.append(k)
        lb = np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)
        kp = np.stack(kpts, 0) if kpts else np.zeros((0, self.nkpt, 3), np.float32)
        return lb, kp

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lb, kp = self._load_label(img_path.stem)
        img, _, _ = letterbox(img, self.img_size)
        if self.augment and np.random.rand() < self.flip_prob and len(lb):
            img = np.fliplr(img).copy()
            lb[:, 1] = 1 - lb[:, 1]
            kp[:, :, 0] = 1 - kp[:, :, 0]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img, torch.from_numpy(lb), torch.from_numpy(kp)


def pose_collate(batch):
    imgs, lbs, kpts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_m = max([l.shape[0] for l in lbs] + [1])
    K = kpts[0].shape[-2] if len(kpts[0].shape) == 3 else 17
    lb_out = torch.full((len(lbs), max_m, 5), -1.0)
    kp_out = torch.zeros((len(lbs), max_m, K, 3))
    for i, (l, k) in enumerate(zip(lbs, kpts)):
        if l.shape[0]:
            lb_out[i, :l.shape[0]] = l
            kp_out[i, :l.shape[0]] = k
    return imgs, lb_out, kp_out


class OBBDataset(YOLODataset):
    def _load_label(self, stem):  # rows: cls cx cy w h theta
        p = self.labels_dir / f"{stem}.txt"
        rows = []
        if p.exists():
            with open(p) as f:
                for line in f:
                    d = line.strip().split()
                    if len(d) >= 6:
                        rows.append([int(float(d[0])), *map(float, d[1:6])])
                    elif len(d) == 5:
                        rows.append([int(float(d[0])), *map(float, d[1:5]), 0.0])
        return np.array(rows, np.float32) if rows else np.zeros((0, 6), np.float32)
