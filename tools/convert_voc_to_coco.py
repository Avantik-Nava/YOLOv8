#!/usr/bin/env python3
"""Convert VOCdevkit XML -> COCO JSON (YOLOX tools/convert_voc_to_coco.py port).

Reads datasets/VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets/Main/*.txt},
writes datasets/COCO-style json. Class order = yolox/data/datasets/voc_classes.py.

Usage:
  python tools/convert_voc_to_coco.py --voc datasets/VOCdevkit --out datasets/COCO/annotations/instances_train2017.json --split trainval
  python tools/convert_voc_to_coco.py --voc datasets/VOCdevkit --year 2007 --split test --out datasets/COCO/annotations/instances_val2017.json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser("voc->coco")
    p.add_argument("--voc", default="datasets/VOCdevkit")
    p.add_argument("--year", default="2007")
    p.add_argument("--split", default="trainval")
    p.add_argument("--out", default="datasets/COCO/annotations/instances_train2017.json")
    return p.parse_args()


def main():
    args = parse_args()
    from yolox.data.datasets import VOC_CLASSES
    cat_id = {n: i + 1 for i, n in enumerate(VOC_CLASSES)}
    voc = Path(args.voc) / ("VOC" + str(args.year))
    ids = (voc / "ImageSets" / "Main" / (args.split + ".txt")).read_text().split()
    images, anns, aid = [], [], 1
    for i, stem in enumerate(ids, 1):
        t = ET.parse(voc / "Annotations" / (stem + ".xml")).getroot()
        w = int(t.find("size").find("width").text)
        h = int(t.find("size").find("height").text)
        images.append({"id": i, "file_name": stem + ".jpg", "width": w, "height": h})
        for o in t.iter("object"):
            name = o.find("name").text.strip()
            if name not in cat_id:
                continue
            b = o.find("bndbox")
            xmin = float(b.find("xmin").text); ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text); ymax = float(b.find("ymax").text)
            bw, bh = xmax - xmin, ymax - ymin
            anns.append({"id": aid, "image_id": i, "category_id": cat_id[name],
                         "bbox": [xmin, ymin, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": v, "name": k} for k, v in cat_id.items()]},
              open(out, "w"))
    print(f"wrote {out}: {len(images)} imgs {len(anns)} anns classes={list(cat_id)}")


if __name__ == "__main__":
    main()
