#!/usr/bin/env python3
# YOLOv8 training — exact YOLOX tools/train.py CLI.
#
#   python tools/train.py -f exps/default/yolov8_s.py -d 0 -b 16 --fp16 -o
#   python tools/train.py -n yolov8-s -b 16
#   python tools/train.py -f exps/example/yolox_voc/yolov8_voc_s.py -o num_classes=2
#   # initial-stage pretrained (fine-tune, head auto-skipped if nc differs):
#   python tools/train.py -f exps/default/yolov8_n.py -c pretrained/yolov8n_coco.pth -b 4 num_classes=3
#   # resume interrupted run (weights + optimizer + epoch):
#   python tools/train.py -f <exp> -c YOLOX_outputs/<exp>/last.pth --resume

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolox.exp import get_exp_by_file, get_exp_by_name


def make_parser():
    parser = argparse.ArgumentParser("YOLOv8 train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="batch size (default: exp.batch_size)")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")
    parser.add_argument("-f", "--exp_file", default=None, type=str,
                        help="plz input your experiment description file")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training (restore optimizer + epoch from -c)")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="pretrained weights for initial stage (fine-tune) or checkpoint with --resume")
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    parser.add_argument("--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision training.")
    parser.add_argument("--cache", type=str, nargs="?", const="ram",
                        help="Caching imgs to ram/disk for fast training.")
    parser.add_argument("-o", "--occupy", dest="occupy", default=False, action="store_true",
                        help="occupy GPU memory first for training.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
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
    if args.batch_size is None:
        args.batch_size = int(getattr(exp, "batch_size", 16) or 16)
    if args.name is not None:
        exp.exp_name = args.name.replace("-", "_")
    if args.experiment_name is not None:
        exp.exp_name = args.experiment_name
    print(exp)

    if getattr(exp, "backend", "native") == "ultra":
        from yolox.engine.ultra import train_ultra
        train_ultra(exp, args)
        return

    from yolox.core import Trainer
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    main()
