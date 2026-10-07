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

import streamlit as st  # noqa: E402 (needed for the player fragment)

from yolox.data import letterbox  # noqa: E402


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


def parse_exp_meta(path: Path):
    """Read exp hyper-params WITHOUT executing the file (AST only).

    YOLOX-style exps can import things that don't exist here
    (get_yolox_datadir, YOLOX models, loguru...). Parsing values with AST
    means those imports can never break model loading.
    """
    import ast
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


LEGACY_MARKERS = ("YOLOPAFPN", "YOLOXHead", "get_yolox_datadir", "SimOTA")


def is_legacy_exp(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return any(m in text for m in LEGACY_MARKERS)


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
    if is_legacy_exp(exp_path):
        # YOLOX-style exp + weights: values are AST-parsed (file is NEVER
        # executed, so its imports cannot fail), arch is auto-matched.
        from types import SimpleNamespace
        from yolox.models.legacy.loader import build_legacy
        meta = parse_exp_meta(exp_path)
        model, info, skipped = build_legacy(
            weights_path, len(names),
            depth_hint=float(meta.get("depth", 0.33)),
            width_hint=float(meta.get("width", 0.50)),
            act=meta.get("act", "silu") or "silu")
        exp = SimpleNamespace(
            exp_name=str(meta.get("exp_name", exp_path.stem)),
            version=f"yolox-legacy d={info['depth']} w={info['width']}",
            depth=info["depth"], width=info["width"],
            num_classes=len(names))
        return model, exp, skipped, info
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
    return model, exp, skipped, None


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

    st.set_page_config(page_title="Validation UI", layout="wide")
    st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; }
    .step-card { border: 1px solid #e0e0e0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    .step-title { font-size: 1.15rem; font-weight: 700; margin-bottom: .4rem; }
    div.stButton > button { border-radius: 10px; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)
    st.title("🎯 Model Validation Studio")

    for k, v in {"model": None, "exp": None, "names": None, "model_info": "",
                 "legacy": None, "media_kind": None, "img_result": None,
                 "img_dets": None, "frames": None, "fps": 25.0,
                 "video_bytes": None, "stats": "", "playing": False,
                 "frame": 0}.items():
        st.session_state.setdefault(k, v)

    # status bar
    s1, s2 = st.columns(2)
    s1.success("✅ Model loaded" if st.session_state.model is not None else "⚪ Model: not loaded")
    media_ok = st.session_state.img_result is not None or st.session_state.frames is not None
    s2.success("✅ Media processed" if media_ok else "⚪ Media: not processed")

    # ---------- STEP 1 ----------
    st.markdown('<div class="step-card"><div class="step-title">Step 1 — Model: upload weights + classes + exp, then Load Model</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    class_file = c1.file_uploader("📄 Class file (.py / .txt / .yaml)", type=["py", "txt", "yaml", "yml"])
    exp_file = c2.file_uploader("📜 Exp file (.py)", type=["py"])
    weights_file = c3.file_uploader("⚙️ Weights (.pth / .pt)", type=["pth", "pt"])
    load_model = st.button("📦 Load Model", type="primary", use_container_width=True)

    if load_model:
        if not (class_file and exp_file and weights_file):
            st.error("Upload all three files first (class file, exp file, weights).")
        else:
            tmp = Path(tempfile.mkdtemp())
            cp, ep, wp = tmp / class_file.name, tmp / exp_file.name, tmp / weights_file.name
            cp.write_bytes(class_file.getvalue())
            ep.write_bytes(exp_file.getvalue())
            wp.write_bytes(weights_file.getvalue())
            try:
                names = parse_classes(cp)
            except Exception as e:
                st.error(f"Class file parse failed: {e}")
                names = None
            if names is not None:
                try:
                    with st.spinner("Loading model..."):
                        model, exp, skipped, legacy_info = build_model(ep, wp, names)
                except Exception as e:
                    st.error(f"Model build failed: {e}")
                    model = None
                if model is not None:
                    st.session_state.update(model=model, exp=exp, names=names,
                                            legacy=legacy_info, frame=0, playing=False,
                                            img_result=None, img_dets=None,
                                            frames=None, video_bytes=None)
                    ok = True
                    if legacy_info is not None and legacy_info["coverage"] < 0.95:
                        st.error("Weights barely match any YOLOX size — check the weights file.")
                        ok = False
                    if ok:
                        st.session_state.model_info = (
                            f"exp `{exp.exp_name}` · arch `{exp.version}` · "
                            f"{len(names)} classes" +
                            (f" · YOLOX legacy coverage {legacy_info['coverage']*100:.1f}%"
                             if legacy_info else ""))
                        st.rerun()
    if st.session_state.model is not None:
        st.success("Loaded: " + st.session_state.model_info)
        st.caption("Classes: " + ", ".join(st.session_state.names[:25]) +
                   (" ..." if len(st.session_state.names) > 25 else ""))
        if st.session_state.legacy is not None:
            st.info(f"YOLOX legacy mode · depth={st.session_state.legacy['depth']} "
                    f"width={st.session_state.legacy['width']} "
                    f"reg_max={st.session_state.legacy['reg_max']}")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.model is None:
        st.info("⬆️ Load a model to unlock Step 2.")
        return

    # ---------- STEP 2 ----------
    st.markdown('<div class="step-card"><div class="step-title">Step 2 — Media: upload image/video, then Load Video</div>',
                unsafe_allow_html=True)
    media_file = st.file_uploader("🖼️ Image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi"])
    o1, o2, o3 = st.columns(3)
    conf = o1.slider("conf threshold", 0.05, 0.90, 0.25, 0.05)
    iou = o2.slider("nms iou", 0.20, 0.95, 0.70, 0.05)
    imgsz = o3.selectbox("imgsz (lower = faster)", [320, 480, 640], index=1)
    load_media = st.button("🎬 Load Video / Run Image", type="primary", use_container_width=True)

    if load_media:
        if not media_file:
            st.error("Upload an image or video first.")
        else:
            tmp = Path(tempfile.mkdtemp())
            mp = tmp / media_file.name
            mp.write_bytes(media_file.getvalue())
            model, names = st.session_state.model, st.session_state.names
            if mp.suffix.lower() in (".mp4", ".avi"):
                frames, fps, total_det, done = _process_video(
                    model, names, mp, conf, iou, imgsz)
                if done:
                    vw_path = tmp / "result.mp4"
                    _write_mp4(frames, fps, vw_path)
                    st.session_state.update(
                        frames=frames, fps=fps, frame=0, playing=False,
                        media_kind="video", img_result=None,
                        video_bytes=vw_path.read_bytes(),
                        stats=f"{done} frames · {total_det} detections · {fps:.0f} fps")
                    st.session_state.scrub = 0
            else:
                img = cv2.imread(str(mp), cv2.IMREAD_COLOR)
                if img is None:
                    st.error("Could not read image.")
                else:
                    with st.spinner("Running inference..."):
                        out, dets = infer_image(model, img, names, conf, iou, imgsz)
                    _, buf = cv2.imencode(".jpg", out)
                    _, buf0 = cv2.imencode(".jpg", img)
                    st.session_state.update(
                        img_result=(buf0.tobytes(), buf.tobytes()),
                        img_dets=[tuple(float(x) for x in d) for d in dets],
                        media_kind="image", frames=None, video_bytes=None,
                        stats=f"{len(dets)} detections")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RESULTS ----------
    if st.session_state.media_kind == "image" and st.session_state.img_result is not None:
        st.subheader(f"Result — {st.session_state.stats}")
        c1, c2 = st.columns(2)
        c1.image(st.session_state.img_result[0], caption="input", use_column_width=True)
        c2.image(st.session_state.img_result[1], caption="result", use_column_width=True)
        st.download_button("⬇️ Download result image", st.session_state.img_result[1],
                           "result.jpg", "image/jpeg")
        if st.session_state.img_dets:
            import pandas as pd
            names = st.session_state.names
            st.dataframe(pd.DataFrame(
                [{"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "conf": d[4],
                  "class": names[int(d[5])] if int(d[5]) < len(names) else int(d[5])}
                 for d in st.session_state.img_dets]), use_container_width=True)

    if st.session_state.media_kind == "video" and st.session_state.frames is not None:
        st.subheader(f"Result — {st.session_state.stats}")
        tab1, tab2 = st.tabs(["▶ Interactive player", "🎞 Full video"])
        with tab1:
            _player(st.session_state.frames, st.session_state.fps)
        with tab2:
            st.video(st.session_state.video_bytes)
        st.download_button("⬇️ Download result video", st.session_state.video_bytes,
                           "result.mp4", "video/mp4")


def _process_video(model, names, mp, conf, iou, imgsz, max_frames=1200):
    cap = cv2.VideoCapture(str(mp))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, total_det, done = [], 0, 0
    prog = st.progress(0, text="Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret or done >= max_frames:
            break
        out, dets = infer_image(model, frame, names, conf, iou, imgsz)
        total_det += len(dets)
        small = out if out.shape[1] <= 640 else cv2.resize(
            out, (640, int(out.shape[0] * 640 / out.shape[1])))
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frames.append(buf.tobytes())
        done += 1
        if n_frames > 0:
            prog.progress(min(done / n_frames, 1.0), text=f"Frame {done}/{n_frames}")
    cap.release()
    if done >= max_frames:
        st.warning(f"Trimmed to first {max_frames} frames.")
    prog.empty()
    return frames, fps, total_det, done


def _write_mp4(frames_jpg, fps, out_path):
    import numpy as np
    first = cv2.imdecode(np.frombuffer(frames_jpg[0], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for b in frames_jpg:
        vw.write(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
    vw.release()


@st.fragment(run_every=0.15)
def _player(frames, fps):
    """Interactive player: Start/Stop + frame scrubber (auto-advances while playing)."""
    n = len(frames)
    b1, b2, _sp = st.columns([1, 1, 6])
    if b1.button("▶ Start", use_container_width=True):
        st.session_state.playing = True
    if b2.button("⏹ Stop", use_container_width=True):
        st.session_state.playing = False
    if st.session_state.playing:
        st.session_state.frame = (st.session_state.frame + max(1, round(fps * 0.15))) % n
    else:
        st.session_state.frame = st.slider(
            "Scrub frames", 0, n - 1, min(st.session_state.frame, n - 1), key="scrub")
    st.image(frames[st.session_state.frame],
             caption=f"frame {st.session_state.frame + 1}/{n}")
    st.caption("▶ playing — press ⏹ Stop to pause" if st.session_state.playing else "⏸ paused")


if __name__ == "__main__":
    main()
