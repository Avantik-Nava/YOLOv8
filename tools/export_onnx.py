"""tools/export_onnx.py — export YOLOv8 to ONNX (YOLOX export_onnx style)."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="s")
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("-o", "--output", default="yolov8.onnx")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    from yolox.models import YOLOv8
    model = YOLOv8(version=args.version, num_classes=args.num_classes).eval()
    if args.ckpt and Path(args.ckpt).exists():
        from yolox.utils import load_checkpoint
        sd = load_checkpoint(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(model, dummy, args.output, opset_version=args.opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    print(f"exported {args.output}")


if __name__ == "__main__":
    main()
