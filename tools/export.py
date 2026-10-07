"""tools/export.py — multi-format export: onnx | torchscript.

YOLOv8 path (Ultralytics-style weights):
  python tools/export.py --weights yolov8n.pt --format onnx -o yolov8n.onnx

Legacy YOLOX path (exp file + checkpoint, e.g. for GPU-server inference
without copying the code):
  python tools/export.py --exp exps/example/yolox_voc/yolov8_voc_s.py \
      --ckpt best.pth --classes classes.txt -o yolox_s.onnx --imgsz 640
  Only model.onnx + tools/infer_onnx.py + video need to go to the server.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--exp", default=None, help="exp .py file (legacy YOLOX exps supported)")
    p.add_argument("--ckpt", default=None, help="checkpoint for --exp path")
    p.add_argument("--classes", default=None, help="class names .txt (defaults: ckpt nc)")
    p.add_argument("--task", default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--format", default="onnx", choices=["onnx", "torchscript"])
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    if args.exp:
        from yolox.utils.exp_parse import is_legacy_exp, parse_classes, parse_exp_meta
        names = parse_classes(Path(args.classes)) if args.classes else None
        if is_legacy_exp(Path(args.exp)):
            from yolox.models.legacy.loader import build_legacy
            meta = parse_exp_meta(Path(args.exp))
            nc = len(names) if names else int(meta.get("num_classes", 80))
            model, info, _ = build_legacy(
                args.ckpt or args.weights, nc,
                float(meta.get("depth", 0.33)), float(meta.get("width", 0.50)),
                meta.get("act", "silu") or "silu")
            print(f"legacy arch d={info['depth']} w={info['width']} "
                  f"coverage={info['coverage']*100:.1f}%")
        else:
            raise ValueError("--exp is only supported for legacy YOLOX exps in this tool")
        out = args.output or "model.onnx"
        model.eval()
        dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # torch.onnx prints ✅ (crashes cp1252)
        except Exception:
            pass
        if args.format == "onnx":
            # dynamo=False -> single self-contained .onnx (weights inlined,
            # easy to ship to the GPU box); dynamo splits out a .data file.
            torch.onnx.export(model, dummy, out, opset_version=args.opset,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                              dynamo=False)
        else:
            torch.jit.save(torch.jit.trace(model, dummy), out)
        print(f"exported {out}")
        return

    from yolox.engine import YOLO
    kw = {}
    if args.task:
        kw["task"] = args.task
    if args.version:
        kw["version"] = args.version
    m = YOLO(args.weights, num_classes=args.num_classes, **kw)
    m.export(format=args.format, imgsz=args.imgsz, path=args.output)


if __name__ == "__main__":
    main()
