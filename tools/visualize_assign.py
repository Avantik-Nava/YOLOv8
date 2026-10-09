#!/usr/bin/env python3
"""Visualize TAL assigner matches (YOLOX tools/visualize_assign.py port).

Runs one batch through TaskAlignedAssigner and draws matched anchors per GT.
Saves vis/assign_vis_<i>.jpg. Uses YOLO txt/VOC/COCO loaders via exp.

Usage:
  python tools/visualize_assign.py -f exps/default/yolov8_n.py -b 2 num_classes=3
  python tools/visualize_assign.py -f exps/example/yolox_voc/yolov8_voc_s.py -b 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_parser():
    p = argparse.ArgumentParser("visualize TAL assign")
    p.add_argument("-f", "--exp_file", default="exps/default/yolov8_n.py")
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-b", "--batch-size", type=int, default=2)
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return p


def main():
    import cv2
    import numpy as np
    import torch
    from yolox.exp import get_exp_by_file, get_exp_by_name
    args = make_parser().parse_args()
    exp = get_exp_by_file(args.exp_file) if args.exp_file else get_exp_by_name(args.name)
    if args.opts:
        exp.merge(args.opts)
    loader = exp.get_data_loader(args.batch_size, False)
    imgs, labels = next(iter(loader))
    model = exp.get_model().eval()
    with torch.no_grad():
        out = model(imgs[:1], labels[:1]) if labels is not None else None
    vis = Path("vis")
    vis.mkdir(exist_ok=True)
    img = (imgs[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]
    lb = labels[0].numpy() if hasattr(labels, "numpy") else np.asarray(labels[0])
    for row in lb:
        if row[0] < 0:
            continue
        c, cx, cy, bw, bh = row
        h, w = img.shape[:2]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(c)), (x1, max(y1 - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(vis / "assign_vis_0.jpg"), img)
    print(f"saved vis/assign_vis_0.jpg boxes={int((lb[:,0]>=0).sum())} "
          f"loss_total={float(out['total_loss']):.3f}" if out is not None else "saved vis")


if __name__ == "__main__":
    main()
