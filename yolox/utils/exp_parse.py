"""Lightweight exp/class-file parsing (no torch/streamlit needed).

Shared by tools/app.py (validation UI) and tools/export.py (ONNX export).
"""

import re
from pathlib import Path

LEGACY_MARKERS = ("YOLOPAFPN", "YOLOXHead", "get_yolox_datadir", "SimOTA")


def parse_classes(path):
    """Class names from .txt (one/line), .yaml (names:), or .py (VOC_CLASSES/NAMES)."""
    path = Path(path)
    suf = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suf == ".txt":
        return [l.strip() for l in text.splitlines() if l.strip()]
    if suf in (".yaml", ".yml"):
        import yaml
        return [str(n) for n in yaml.safe_load(text).get("names", [])]
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    m = re.search(r"(?:VOC_CLASSES|NAMES|CLASSES)\s*=\s*\((.*?)\)", code, re.S)
    if not m:
        m = re.search(r"(?:VOC_CLASSES|NAMES|CLASSES)\s*=\s*\[(.*?)\]", code, re.S)
    if not m:
        raise ValueError("No VOC_CLASSES/NAMES found in .py file")
    names = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    if not names:
        raise ValueError("Could not parse class names from .py file")
    return names


def is_legacy_exp(path):
    """True if the exp file is YOLOX-style (needs the legacy inference path)."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return any(m in text for m in LEGACY_MARKERS)


def parse_exp_meta(path):
    """Read exp hyper-params WITHOUT executing the file (AST only).

    YOLOX-style exps can import things that don't exist here
    (get_yolox_datadir, YOLOX models, loguru...). Parsing values with AST
    means those imports can never break model loading.
    """
    import ast
    path = Path(path)
    meta = {"num_classes": 80, "depth": 0.33, "width": 0.50,
            "act": "silu", "exp_name": path.stem}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return meta
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Exp":
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if (isinstance(t, ast.Attribute)
                                        and isinstance(t.value, ast.Name)
                                        and t.value.id == "self"
                                        and t.attr in meta):
                                    try:
                                        meta[t.attr] = ast.literal_eval(sub.value)
                                    except Exception:
                                        pass
    return meta
