#!/usr/bin/env python3
"""Unified CLI:  python yolo_cli.py task=detect mode=train model=yolov8s.pt data=data/custom.yaml epochs=10
Tasks: detect/segment/classify/pose/obb | Modes: train/val/predict/export/track/benchmark
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_kv(argv):
    kv = {}
    for a in argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
    return kv


def main():
    kv = parse_kv(sys.argv)
    task = kv.get("task", "detect")
    mode = kv.get("mode", "train")
    model = kv.get("model", "yolov8n.pt")
    data = kv.get("data", "data/custom.yaml")
    epochs = int(kv.get("epochs", 10))
    imgsz = int(kv.get("imgsz", 640))
    nc = int(kv.get("nc", kv.get("num_classes", 80)))

    from yolox.engine import YOLO
    m = YOLO(model, task=task, num_classes=nc)
    if mode == "train":
        m.train(data=data, epochs=epochs, imgsz=imgsz)
    elif mode == "val":
        m.val(data=data, imgsz=imgsz)
    elif mode == "predict":
        print(m.predict(kv.get("source", "demo.jpg"), imgsz=imgsz))
    elif mode == "export":
        m.export(format=kv.get("format", "onnx"), imgsz=imgsz)
    elif mode == "track":
        m.track(source=kv.get("source", 0), imgsz=imgsz)
    elif mode == "benchmark":
        m.benchmark(imgsz=imgsz)
    else:
        raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
