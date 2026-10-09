#!/usr/bin/env python3
"""Export TorchScript with YOLOX-style CLI (tools/export_torchscript.py port).

Uses YOLOv8 model (C2f + DFL head) built from exp file, not YOLOX arch.

Usage:
  python tools/export_torchscript.py -f exps/default/yolov8_n.py -c YOLOX_outputs/yolov8_n/best.pth --output-name yolov8n.torchscript.pt
  python tools/export_torchscript.py -n yolov8-n --output-name yolov8n.torchscript.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_parser():
    p = argparse.ArgumentParser("YOLOv8 torchscript deploy")
    p.add_argument("--output-name", default="yolov8n.torchscript.pt")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("-f", "--exp_file", default=None)
    p.add_argument("-expn", "--experiment-name", default=None)
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-c", "--ckpt", default=None)
    p.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return p


def main():
    import torch
    from yolox.exp import get_exp_by_file, get_exp_by_name
    args = make_parser().parse_args()
    exp = get_exp_by_file(args.exp_file) if args.exp_file else (
        get_exp_by_name(args.name) if args.name else get_exp_by_file("exps/default/yolov8_n.py"))
    if args.opts:
        exp.merge(args.opts)
    model = exp.get_model().eval()
    if args.ckpt and Path(args.ckpt).exists():
        from yolox.utils import load_checkpoint
        sd = load_checkpoint(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        print(f"loaded {args.ckpt}")
    dummy = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    torch.jit.save(torch.jit.trace(model, dummy), args.output_name)
    print(f"exported {args.output_name} nc={exp.num_classes}")


if __name__ == "__main__":
    main()
