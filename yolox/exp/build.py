#!/usr/bin/env python3
# Exp factory — mirrors yolox/exp/build.py: get_exp_by_file / get_exp_by_name.

import importlib
import os
import sys

__all__ = ["get_exp_by_file", "get_exp_by_name"]


def get_exp_by_file(exp_file):
    assert os.path.isfile(exp_file), f"exp file {exp_file} not found"
    sys.path.insert(0, os.path.dirname(exp_file))
    current_dir = os.path.dirname(exp_file)
    exp_name = os.path.splitext(os.path.basename(exp_file))[0]
    # support exps/default/*.py loaded as top-level module (absolute imports inside)
    loader = importlib.machinery.SourceFileLoader(exp_name, exp_file)
    spec = importlib.util.spec_from_loader(exp_name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.Exp()


def get_exp_by_name(exp_name):
    # e.g. -n yolov8-s -> exps/default/yolov8_s.py
    candidates = [
        os.path.join("exps", "default", exp_name.replace("-", "_") + ".py"),
        os.path.join("exps", exp_name.replace("-", "_") + ".py"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return get_exp_by_file(c)
    raise ValueError(f"unknown exp name '{exp_name}', tried {candidates}")
