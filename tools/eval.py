#!/usr/bin/env python3
# YOLOv8 evaluation — exact YOLOX tools/eval.py CLI.
#
#   python tools/eval.py -f exps/default/yolov8_s.py -c YOLOX_outputs/yolov8_s/best.pth -b 16
#   python tools/eval.py -n yolov8-s -c <ckpt> --test-conf 0.01

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolox.exp import get_exp_by_file, get_exp_by_name


def make_parser():
    parser = argparse.ArgumentParser("YOLOv8 eval parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--dist-url", default=None, type=str)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-d", "--devices", default=None, type=int)
    parser.add_argument("-f", "--exp_file", default=None, type=str)
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument("--test", default=False, action="store_true", help="eval on testdev")
    parser.add_argument("--test-conf", type=float, default=None)
    parser.add_argument("--nmsthre", type=float, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser


def main():
    args = make_parser().parse_args()

    if args.exp_file is not None:
        exp = get_exp_by_file(args.exp_file)
    elif args.name is not None:
        exp = get_exp_by_name(args.name)
    else:
        exp = get_exp_by_file("exps/default/yolov8_s.py")

    exp.merge(args.opts)
    if args.test_conf is not None:
        exp.test_conf = args.test_conf
    if args.nmsthre is not None:
        exp.nmsthre = args.nmsthre
    if args.test_size is not None:
        exp.test_size = (args.test_size, args.test_size)
    print(exp)

    if getattr(exp, "backend", "native") == "ultra":
        from yolox.engine.ultra import val_ultra
        val_ultra(exp, args)
        return

    if args.devices is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.devices)
        device = torch.device(f"cuda:{args.devices}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.get_model().to(device).eval()
    if args.ckpt:
        from yolox.utils import load_checkpoint
        ckpt = load_checkpoint(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"loaded {args.ckpt}")

    evaluator = exp.get_evaluator(args.batch_size, False, testdev=args.test)
    map5095, map50 = exp.eval(model, evaluator, False)
    print(f"mAP50-95={map5095:.4f} mAP50={map50:.4f}")


if __name__ == "__main__":
    main()
