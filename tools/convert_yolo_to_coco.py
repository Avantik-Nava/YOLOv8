#!/usr/bin/env python3
"""Convert YOLO txt -> COCO JSON (YOLOX tools/convert_yolo_to_coco.py port).

YOLO txt rows: cls cx cy w h (normalized). Needs image sizes via cv2.

Usage:
  python tools/convert_yolo_to_coco.py --images-dir dataset/images/train --labels-dir dataset/labels/train --classes data/classes.txt --output datasets/COCO/annotations/instances_train2017.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser("yolo->coco")
    p.add_argument("--images-dir", default="dataset/images/train")
    p.add_argument("--labels-dir", default="dataset/labels/train")
    p.add_argument("--classes", default=None,
                   help="one name per line; default: data/custom.yaml names")
    p.add_argument("--output", default="datasets/COCO/annotations/instances_train2017.json")
    return p.parse_args()


def load_names(path, fallback_yaml="data/custom.yaml"):
    if path and Path(path).exists():
        return [l.strip() for l in open(path) if l.strip()]
    import yaml
    d = yaml.safe_load(open(fallback_yaml))
    return list(d.get("names", []))


def main():
    import cv2
    args = parse_args()
    names = load_names(args.classes)
    idir, ldir = Path(args.images_dir), Path(args.labels_dir)
    imgs = sorted([p for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                   for p in list(idir.glob(e)) + list(idir.glob(e.upper()))],
                  key=lambda p: p.name.lower())
    seen, files = set(), []
    for p in imgs:
        k = str(p.resolve()).lower()
        if k not in seen:
            seen.add(k)
            files.append(p)
    images, anns, aid = [], [], 1
    for i, jp in enumerate(files, 1):
        img = cv2.imread(str(jp))
        h, w = img.shape[:2] if img is not None else (640, 640)
        images.append({"id": i, "file_name": jp.name, "width": w, "height": h})
        lp = ldir / (jp.stem + ".txt")
        if not lp.exists():
            continue
        for line in open(lp):
            d = line.strip().split()
            if len(d) != 5:
                continue
            c, cx, cy, bw, bh = int(float(d[0])), *map(float, d[1:])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            anns.append({"id": aid, "image_id": i, "category_id": c + 1,
                         "bbox": [x, y, bw * w, bh * h],
                         "area": bw * w * bh * h, "iscrowd": 0})
            aid += 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]},
              open(out, "w"))
    print(f"wrote {out}: {len(images)} imgs {len(anns)} anns names={names}")


if __name__ == "__main__":
    main()
