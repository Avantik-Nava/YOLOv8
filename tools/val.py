#!/usr/bin/env python3
# Back-compat shim: `python tools/val.py ...` -> tools/eval.py (YOLOX CLI).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval import main  # noqa: E402

if __name__ == "__main__":
    main()
