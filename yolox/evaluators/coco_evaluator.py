#!/usr/bin/env python3
# Evaluators — YOLOX yolox/evaluators API (COCOEvaluator/VOCEvaluator.evaluate),
# metrics via yolox.utils.coco_map (mAP50 + mAP50-95).

import numpy as np
import torch

from yolox.utils import coco_map


class COCOEvaluator:
    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False):
        self.dataloader = dataloader
        self.img_size = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self, model, is_distributed=False, half=False, return_outputs=False):
        model.eval()
        all_dets, all_gts = [], []
        for imgs, labels in self.dataloader:
            dets = model.predict(imgs, conf_threshold=self.confthre, iou_threshold=self.nmsthre)
            for b in range(imgs.shape[0]):
                all_dets.append(np.asarray(dets[b].cpu()))
                gt = []
                for m in range(labels.shape[1]):
                    c = float(labels[b, m, 0])
                    if c < 0:
                        continue
                    _, cx, cy, w, h = labels[b, m].tolist()
                    W = H = self.img_size
                    gt.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                               (cx + w / 2) * W, (cy + h / 2) * H])
                all_gts.append(np.array(gt, np.float32).reshape(-1, 5))
        res = coco_map(all_dets, all_gts)
        print(f"mAP50={res['map50']:.4f} mAP50-95={res['map5095']:.4f} "
              f"P={res['precision']:.4f} R={res['recall']:.4f}")
        return res["map5095"], res["map50"]


class VOCEvaluator(COCOEvaluator):
    pass
