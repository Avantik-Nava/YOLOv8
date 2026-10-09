#!/usr/bin/env python3
"""TensorRT export with YOLOX-style CLI (tools/trt.py port).

Prefers torch2trt when installed (GPU); otherwise exports ONNX (for trtexec)
so the command never hard-fails on CPU boxes.

Usage:
  python tools/trt.py -f exps/default/yolov8_n.py -c YOLOX_outputs/yolov8_n/best.pth -b 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_parser():
    p = argparse.ArgumentParser("YOLOv8 trt deploy")
    p.add_argument("-expn", "--experiment-name", default=None)
    p.add_argument("-n", "--name", default=None)
    p.add_argument("-f", "--exp_file", default=None)
    p.add_argument("-c", "--ckpt", default=None)
    p.add_argument("-w", "--workspace", type=int, default=32)
    p.add_argument("-b", "--batch", type=int, default=1)
    return p


def main():
    import torch
    from yolox.exp import get_exp_by_file, get_exp_by_name
    args = make_parser().parse_args()
    exp = get_exp_by_file(args.exp_file) if args.exp_file else (
        get_exp_by_name(args.name) if args.name else get_exp_by_file("exps/default/yolov8_n.py"))
    model = exp.get_model().eval()
    if args.ckpt and Path(args.ckpt).exists():
        from yolox.utils import load_checkpoint
        sd = load_checkpoint(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    dummy = torch.randn(args.batch, 3, exp.test_size[0], exp.test_size[1])
    try:
        from torch2trt import torch2trt
        model.cuda().eval()
        out = torch2trt(model.cuda(), [dummy.cuda()], max_workspace_size=(1 << 30) * args.workspace)
        torch.save(out.state_dict(), "yolov8_trt.pth")
        print("exported yolov8_trt.pth via torch2trt")
    except Exception as e:
        onnx_path = "yolov8_trt.onnx"
        torch.onnx.export(model, dummy, onnx_path, opset_version=18,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"torch2trt unavailable ({e}); exported {onnx_path} — build engine with: trtexec --onnx={onnx_path} --saveEngine=yolov8.engine")


if __name__ == "__main__":
    main()
