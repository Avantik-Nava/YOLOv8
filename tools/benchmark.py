"""tools/benchmark.py — latency/FPS/params (mode=benchmark)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--task", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=50)
    args = p.parse_args()

    from yolox.engine import YOLO
    kw = {}
    if args.task:
        kw["task"] = args.task
    if args.version:
        kw["version"] = args.version
    m = YOLO(args.weights, num_classes=args.num_classes, **kw)
    print(m.benchmark(imgsz=args.imgsz, runs=args.runs))


if __name__ == "__main__":
    main()
