#!/usr/bin/env python3
"""Single-node launch (YOLOX yolox/core/launch.py port).

YOLOX supports DDP multi-GPU via torch.multiprocessing. This port keeps the
same launch() signature for compat but runs single-process training (YOLOv8
here is single-GPU/CPU; ddp:false in config.yaml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0,
           backend="nccl", dist_url=None, args=(), timeout=None):
    if num_gpus_per_machine > 1:
        print(f"[launch] DDP requested ({num_gpus_per_machine} gpus) — "
              "running single-process fallback (see tools/train.py -d)")
    main_func(*args)
