"""Local validation UI (Streamlit).

Upload: class file + exp file + weights + image/video  ->  annotated result.

Run:  streamlit run tools/app.py
Then open the printed Local URL (http://localhost:8501).

Class file formats accepted:
  .txt  -> one class name per line
  .py   -> VOC_CLASSES = (...) or NAMES = [...] (e.g. yolo/data/datasets/voc_classes.py)
  .yaml -> nc: N + names: [...] (e.g. data/custom.yaml)
"""

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yolo.data import letterbox  # noqa: E402


# ---------------- parsing ----------------
def parse_classes(path: Path):
    suf = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suf == ".txt":
        names = [l.strip() for l in text.splitlines() if l.strip()]
        return names
    if suf in (".yaml", ".yml"):
        import yaml
        d = yaml.safe_load(text)
        names = d.get("names", [])
        return [str(n) for n in names]
    # .py — find VOC_CLASSES / NAMES / CLASSES tuple or list (ignore # comments)
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


def load_exp_file(path: Path):
    spec = importlib.util.spec_from_file_location("ui_exp_module", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


# ---------------- model ----------------
_VERSION_BY_STEM = {16: "n", 32: "s", 48: "m", 64: "l", 80: "x"}


def _load_matching(model, sd):
    """Load only keys with matching shapes; return (loaded, skipped)."""
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in sd if k not in keep]
    model.load_state_dict(keep, strict=False)
    return keep, skipped


def build_model(exp_path: Path, weights_path: Path, names):
    exp = load_exp_file(exp_path)
    exp.num_classes = len(names)
    model = exp.get_model()
    sd = torch.load(str(weights_path), map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    try:
        _, skipped = _load_matching(model, sd)
        if skipped:
            # maybe wrong version in exp file? detect from checkpoint stem width
            stem = sd.get("backbone.stem.0.conv.weight")
            if stem is not None and stem.shape[1] == 3:
                guess = _VERSION_BY_STEM.get(int(stem.shape[0]))
                if guess and guess != exp.version:
                    exp.version = guess  # auto-fix version, rebuild
                    model = exp.get_model()
                    _, skipped = _load_matching(model, sd)
        if skipped:
            pass  # caller reports it (see main)
    except Exception as e:
        raise RuntimeError(f"weights do not match exp (version={exp.version}, "
                           f"nc={len(names)}): {e}")
    model.eval()
    return model, exp, skipped


@torch.no_grad()
def infer_image(model, img_bgr, names, conf=0.25, iou=0.7, imgsz=640):
    h0, w0 = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb, s, (dw, dh) = letterbox(rgb, imgsz)
    t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    dets = model.predict(t, conf_threshold=conf, iou_threshold=iou)[0]
    dets = dets.cpu().numpy()
    if len(dets):
        dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s
        dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w0)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h0)
    out = img_bgr.copy()
    for x1, y1, x2, y2, cf, cls in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)] if int(cls) < len(names) else int(cls)} {cf:.2f}"
        cv2.putText(out, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out, dets


def main():
    import streamlit as st

    st.set_page_config(page_title="YOLOv8 local validation", layout="wide")
    st.title("YOLOv8 — local validation UI")

    with st.sidebar:
        st.header("1. Upload files")
        class_file = st.file_uploader("Class file (.py / .txt / .yaml)", type=["py", "txt", "yaml", "yml"])
        exp_file = st.file_uploader("Exp file (.py)", type=["py"])
        weights_file = st.file_uploader("Weights (.pth)", type=["pth", "pt"])
        st.header("2. Upload media")
        media_file = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi"])
        st.header("3. Settings")
        conf = st.slider("conf threshold", 0.05, 0.9, 0.25, 0.05)
        iou = st.slider("nms iou", 0.2, 0.95, 0.7, 0.05)
        imgsz = st.selectbox("imgsz", [320, 480, 640], index=2)
        run = st.button("Run inference", type="primary")

    if not run:
        st.info("Upload class file + exp file + weights + an image/video, then press **Run inference**.")
        return

    if not (class_file and exp_file and weights_file and media_file):
        st.error("All four uploads are required (class file, exp file, weights, media).")
        return

    tmp = Path(tempfile.mkdtemp())
    cp = tmp / class_file.name
    ep = tmp / exp_file.name
    wp = tmp / weights_file.name
    mp = tmp / media_file.name
    cp.write_bytes(class_file.getvalue())
    ep.write_bytes(exp_file.getvalue())
    wp.write_bytes(weights_file.getvalue())
    mp.write_bytes(media_file.getvalue())

    try:
        names = parse_classes(cp)
    except Exception as e:
        st.error(f"Class file parse failed: {e}")
        return
    st.write(f"Classes ({len(names)}): {', '.join(names[:20])}" + (" ..." if len(names) > 20 else ""))

    try:
        model, exp, skipped = build_model(ep, wp, names)
    except Exception as e:
        st.error(f"Model build failed: {e}")
        return
    if skipped:
        st.warning(f"Weights partially matched ({len(skipped)} layers skipped — "
                   "usually a class-count difference vs training. "
                   "If labels look wrong, retrain for your class count.")
    st.success(f"Loaded exp `{exp.exp_name}` (yolov8-{exp.version}, nc={len(names)}) + weights `{weights_file.name}`")

    is_video = mp.suffix.lower() in (".mp4", ".avi")
    if not is_video:
        img = cv2.imread(str(mp), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Could not read image.")
            return
        out, dets = infer_image(model, img, names, conf, iou, imgsz)
        st.write(f"Detections: {len(dets)}")
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="input", use_column_width=True)
        c2.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="result", use_column_width=True)
        _, buf = cv2.imencode(".jpg", out)
        st.download_button("Download result image", buf.tobytes(), "result.jpg", "image/jpeg")
        if len(dets):
            import pandas as pd
            df = pd.DataFrame([{"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3],
                                          "conf": d[4], "class": names[int(d[5])] if int(d[5]) < len(names) else int(d[5])}
                                         for d in dets])
            st.dataframe(df)
    else:
        cap = cv2.VideoCapture(str(mp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_path = tmp / "result.mp4"
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        prog = st.progress(0, text="Processing video...")
        total_det, done = 0, 0
        preview = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out, dets = infer_image(model, frame, names, conf, iou, imgsz)
            total_det += len(dets)
            vw.write(out)
            done += 1
            if preview is None:
                preview = out
            if n_frames > 0:
                prog.progress(min(done / n_frames, 1.0), text=f"Frame {done}/{n_frames}")
        cap.release()
        vw.release()
        st.write(f"Frames: {done}, total detections: {total_det}")
        if preview is not None:
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="first frame", use_column_width=True)
        st.video(str(out_path))
        st.download_button("Download result video", out_path.read_bytes(), "result.mp4", "video/mp4")


if __name__ == "__main__":
    main()
