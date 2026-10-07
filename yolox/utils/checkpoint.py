"""Checkpoint loading that works with PyTorch >= 2.6 (weights_only=True default).

Training checkpoints (e.g. YOLOX best_ckpt) often store numpy scalars /
extra metadata that weights_only=True rejects. We first try the safe
restricted load, allowlist common numpy globals, and only then fall back
to weights_only=False (user's own file, local validation use).
"""

import torch


def _allow_numpy_scalars():
    try:
        import numpy as np
        globs = []
        for mod_path in ("numpy._core.multiarray", "numpy.core.multiarray"):
            try:
                mod = __import__(mod_path, fromlist=["scalar"])
                globs.append(mod.scalar)
            except Exception:
                pass
        if globs:
            torch.serialization.add_safe_globals(globs)
    except Exception:
        pass


def load_checkpoint(path, map_location="cpu"):
    """torch.load with graceful handling of weights_only restrictions."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception:
        pass
    _allow_numpy_scalars()
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(str(path), map_location=map_location, weights_only=False)
