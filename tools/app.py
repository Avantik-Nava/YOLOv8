"""Local validation UI (Streamlit) — direct video inference.

Upload: class file + exp file + weights, Load Model,
then upload a video, Load Video -> annotated result with player.

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


# ---------------- parsing (shared helpers live in yolox.utils.exp_parse) ----------------
from yolox.utils.exp_parse import (  # noqa: E402
    is_legacy_exp, parse_classes, parse_exp_meta,
)


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


def build_model(exp_path: Path, weights_path: Path, names, device_choice="auto"):
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
        return _speedup(model, _pick_device(device_choice)), exp, skipped, info
    exp = load_exp_file(exp_path)
    exp.num_classes = len(names)
    model = exp.get_model()
    from yolox.utils import load_checkpoint
    sd = load_checkpoint(weights_path, map_location="cpu")
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
    return _speedup(model, _pick_device(device_choice)), exp, skipped, None


def _draw(img_bgr, dets, names):
    out = img_bgr.copy()
    for x1, y1, x2, y2, cf, cls in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)] if int(cls) < len(names) else int(cls)} {cf:.2f}"
        cv2.putText(out, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


@torch.no_grad()
def _infer_batch(model, frames_bgr, names, conf=0.25, iou=0.7, imgsz=640):
    """Batched inference (much faster than per-frame). Returns [(out, dets)].

    Legacy YOLOX models get YOLOX-faithful input: BGR, top-left pad (no centering).
    """
    import numpy as np
    legacy = bool(getattr(model, "yolox_legacy", False))
    pre, meta = [], []
    for f in frames_bgr:
        h0, w0 = f.shape[:2]
        s = min(imgsz / h0, imgsz / w0)
        if legacy:
            img = cv2.resize(f, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
            canvas[:img.shape[0], :img.shape[1]] = img  # BGR, top-left (YOLOX preproc)
            dw = dh = 0
        else:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            canvas, s, (dw, dh) = letterbox(rgb, imgsz)
        pre.append(torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0)
        meta.append((w0, h0, s, dw, dh))
    t = torch.stack(pre).contiguous(memory_format=torch.channels_last)
    dev = _model_device(model)
    with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
        dets_list = model.predict(t.to(dev, non_blocking=True),
                                  conf_threshold=conf, iou_threshold=iou)
    outs = []
    for f, dets, (w0, h0, s, dw, dh) in zip(frames_bgr, dets_list, meta):
        dets = dets.cpu().numpy()
        if len(dets):
            dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s
            dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
            dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w0)
            dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h0)
        outs.append((_draw(f, dets, names), dets))
    return outs


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
                 "legacy": None, "frames": None, "fps": 25.0,
                 "video_bytes": None, "stats": "", "playing": False,
                 "frame": 0, "mid_raw": None, "ssh": None, "ssh_info": "",
                 "ssh_cred": None, "remote_ready": False, "remote_log": "",
                 "cls_txt": None, "onnx_path": None}.items():
        st.session_state.setdefault(k, v)

    # status bar
    s1, s2, s3 = st.columns(3)
    s1.success("✅ Model loaded" if st.session_state.model is not None else "⚪ Model: not loaded")
    media_ok = st.session_state.frames is not None
    s2.success("✅ Video processed" if media_ok else "⚪ Video: not processed")
    s3.success("✅ GPU server" if _ssh_alive() else "⚪ GPU server: local mode")

    # ---------- STEP 1 ----------
    st.markdown('<div class="step-card"><div class="step-title">Step 1 — Model: upload weights + classes + exp, then Load Model</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    class_file = c1.file_uploader("📄 Class file (.py / .txt / .yaml)", type=["py", "txt", "yaml", "yml"])
    exp_file = c2.file_uploader("📜 Exp file (.py)", type=["py"])
    weights_file = c3.file_uploader("⚙️ Weights (.pth / .pt)", type=["pth", "pt"])
    d1, d2 = st.columns([1, 3])
    device_choice = d1.selectbox("Device", ["auto", "cuda", "cpu"], index=0,
                                 help="auto = GPU if available. Reload model after changing.")
    d2.caption("GPU = ~20-50x faster. Model loads onto the selected device.")
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
                        model, exp, skipped, legacy_info = build_model(
                            ep, wp, names, device_choice=device_choice)
                except Exception as e:
                    st.error(f"Model build failed: {e}")
                    model = None
                if model is not None:
                    st.session_state.update(model=model, exp=exp, names=names,
                                            legacy=legacy_info, frame=0, playing=False,
                                            frames=None, video_bytes=None)
                    ok = True
                    if legacy_info is not None and legacy_info["coverage"] < 0.95:
                        st.error("Weights barely match any YOLOX size — check the weights file.")
                        ok = False
                    if ok:
                        dev = _model_device(model)
                        amp = " · AMP fp16" if dev.type == "cuda" else ""
                        st.session_state.model_info = (
                            f"exp `{exp.exp_name}` · arch `{exp.version}` · "
                            f"{len(names)} classes · device `{dev}`{amp}" +
                            (f" · YOLOX legacy coverage {legacy_info['coverage']*100:.1f}%"
                             if legacy_info else ""))
                        st.rerun()
    if st.session_state.model is not None:
        st.success("✅ Model loaded successfully: " + st.session_state.model_info)
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

    # ---------- STEP 1.5 : GPU server ----------
    st.markdown('<div class="step-card"><div class="step-title">Step 1.5 — GPU server over SSH (optional, much faster)</div>',
                unsafe_allow_html=True)
    st.caption("Code stays on this machine. Only model.onnx + script + video are sent; "
               "inference runs on the server, the result video comes back here. "
               "Credentials live only in memory, never on disk.")
    g1, g2, g3, g4 = st.columns([2, 1, 1, 1])
    ssh_host = g1.text_input("Host", value=(st.session_state.ssh_cred or {}).get("host", "172.32.32.21"))
    ssh_user = g2.text_input("User", value=(st.session_state.ssh_cred or {}).get("user", "navavisionai"))
    ssh_port = g3.number_input("Port", 1, 65535, 22)
    ssh_pass = g4.text_input("Password", type="password")
    ssh_key = st.file_uploader("Key file instead of password (optional)", type=["pem", "key", "ppk", "openssh"])
    b1, b2 = st.columns(2)
    if b1.button("🔌 Connect to GPU", use_container_width=True):
        try:
            with st.spinner("Connecting..."):
                from tools.remote_gpu import connect, gpu_info, ensure_env
                client, ver = connect(
                    ssh_host, ssh_user,
                    password=ssh_pass or None,
                    key_bytes=ssh_key.getvalue() if ssh_key else None, port=int(ssh_port))
                st.session_state.ssh = client
                st.session_state.ssh_cred = {"host": ssh_host, "user": ssh_user, "port": int(ssh_port)}
                st.session_state.ssh_info = f"{ver} · {gpu_info(client)}"
                env_log = ensure_env(client)
                st.session_state.remote_log = "\n".join(env_log)
                st.session_state.remote_ready = False  # re-ship model below
        except Exception as e:
            st.session_state.ssh = None
            st.error(f"SSH failed: {e}")
    if b2.button("🔌 Disconnect", use_container_width=True):
        try:
            if st.session_state.ssh is not None:
                st.session_state.ssh.close()
        except Exception:
            pass
        st.session_state.update(ssh=None, ssh_info="", ssh_cred=None, remote_ready=False)
    if st.session_state.ssh_info:
        st.success("Connected: " + st.session_state.ssh_info)
        if st.session_state.remote_log:
            with st.expander("Server environment"):
                st.code(st.session_state.remote_log)
        if st.session_state.remote_ready:
            st.success("✅ Model shipped to GPU — videos will run remotely.")
        elif st.session_state.model is not None:
            if st.button("📤 Ship current model to GPU", use_container_width=True):
                _ship_model_to_gpu()
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- STEP 2 ----------
    st.markdown('<div class="step-card"><div class="step-title">Step 2 — Video: file upload or RTSP, then Load Video</div>',
                unsafe_allow_html=True)
    src = st.radio("Source", ["📁 Upload video file", "📡 RTSP stream (records N sec)"], horizontal=True)
    media_file, rtsp_url, rtsp_sec = None, "", 15
    if src.startswith("📁"):
        media_file = st.file_uploader("🎬 Video file (.mp4 / .avi)", type=["mp4", "avi"])
    else:
        r1, r2 = st.columns([3, 1])
        rtsp_url = r1.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:554/stream")
        rtsp_sec = r2.number_input("seconds", 5, 120, 15)
    o1, o2, o3 = st.columns(3)
    conf = o1.slider("conf threshold", 0.01, 0.90, 0.25, 0.01)
    iou = o2.slider("nms iou", 0.20, 0.95, 0.70, 0.05)
    imgsz = o3.selectbox("imgsz (lower = faster)", [320, 480, 640], index=0)
    with st.expander("⚡ Speed options", expanded=False):
        s1, s2 = st.columns(2)
        batch = s1.selectbox("batch size (frames per forward)", [1, 2, 4, 8], index=2)
        stride = s2.selectbox("process every Nth frame (rest reuse boxes)", [1, 2, 3, 5], index=0)
        st.caption("Batch 4 + stride 2 ≈ 4–6× faster than before on CPU.")
    load_label = ("⚡ Run on GPU server" if _ssh_alive() and
                    (st.session_state.remote_ready or st.session_state.model is not None)
                    else "🎬 Load Video")
    load_media = st.button(load_label, type="primary", use_container_width=True)

    if load_media:
        tmp = Path(tempfile.mkdtemp())
        if src.startswith("📡"):
            if not rtsp_url:
                st.error("Enter an RTSP URL first.")
                mp = None
            else:
                mp = _record_rtsp(st, rtsp_url, rtsp_sec, tmp)
        else:
            if not media_file:
                st.error("Upload a video first.")
                mp = None
            else:
                mp = tmp / media_file.name
                mp.write_bytes(media_file.getvalue())
        if mp is not None:
            use_gpu = False
            if _ssh_alive():
                if not st.session_state.remote_ready and st.session_state.model is not None:
                    with st.spinner("GPU connected but model not shipped — shipping now..."):
                        _ship_model_to_gpu()
                use_gpu = bool(st.session_state.remote_ready)
                if not use_gpu:
                    st.warning("⚠️ GPU ship failed — running on LOCAL CPU instead. "
                               "See the error above; fix it and reload the video.")
            if use_gpu:
                _load_video_remote(mp, conf, iou, imgsz, batch, stride)
            else:
                model, names = st.session_state.model, st.session_state.names
                frames, fps, total_det, done, calls, speed, mid_raw = _process_video(
                    model, names, mp, conf, iou, imgsz, batch=batch, stride=stride)
                if done:
                    vw_path = tmp / "result.mp4"
                    _write_mp4(frames, fps, vw_path)
                    _, mid_buf = cv2.imencode(".jpg", mid_raw if mid_raw is not None else
                                              cv2.imdecode(np.frombuffer(frames[0], np.uint8),
                                                           cv2.IMREAD_COLOR))
                    st.session_state.update(
                        frames=frames, fps=fps, frame=0, playing=False,
                        video_bytes=vw_path.read_bytes(),
                        mid_raw=mid_buf.tobytes(),
                        stats=(f"💻 LOCAL CPU · {done} frames · {total_det} detections · "
                               f"{speed:.1f} fps processing ({calls} forwards)"))
                    st.session_state.scrub = 0
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RESULTS ----------
    if st.session_state.frames is not None:
        st.subheader(f"Result — {st.session_state.stats}")
        tab1, tab2 = st.tabs(["▶ Interactive player", "🎞 Full video"])
        with tab1:
            _player(st.session_state.frames, st.session_state.fps)
        with tab2:
            st.video(st.session_state.video_bytes)
        st.download_button("⬇️ Download result video", st.session_state.video_bytes,
                           "result.mp4", "video/mp4")
        with st.expander("🔍 Low / no detections? Diagnose on middle frame"):
            if st.button("Run diagnosis"):
                import numpy as np
                raw = cv2.imdecode(np.frombuffer(st.session_state.mid_raw, np.uint8),
                                   cv2.IMREAD_COLOR)
                with st.spinner("Probing raw model output..."):
                    d = _diagnose(st.session_state.model, st.session_state.names, raw, imgsz)
                st.write(f"Input pipeline: **{d['arch']}** · max box score: **{d['max_score']}**")
                st.write("Boxes above threshold:", d["counts"])
                st.table(d["top5"])
                if d["max_score"] < 0.05:
                    st.warning("Max score is near zero — the weights may not match this video "
                               "(wrong classes/version), or the checkpoint didn't load fully. "
                               "Check the coverage % shown after Load Model.")
                elif d["counts"].get(0.25, 0) == 0 and d["counts"].get(0.05, 0) > 0:
                    st.info("Boxes exist at low conf — lower the conf slider (try 0.05–0.10) and reload the video.")


def _pick_device(choice="auto"):
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _speedup(model, device=None):
    """Throughput: device placement + all CPU threads + channels-last convs."""
    import os
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cpu":
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 4))
        except Exception:
            pass
    try:
        model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    return model


def _amp_enabled(model):
    try:
        return next(model.parameters()).is_cuda
    except Exception:
        return False


def _record_rtsp(st, url, seconds, tmp):
    """Record `seconds` of RTSP to a temp mp4. Returns path or None."""
    import time
    cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    except Exception:
        pass
    if not cap.isOpened():
        st.error("Could not open RTSP stream — check URL / network.")
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    out = tmp / "rtsp_in.mp4"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t0, n = time.time(), 0
    prog = st.progress(0, text="Recording RTSP...")
    with st.spinner(f"Recording {seconds}s from RTSP..."):
        while time.time() - t0 < seconds:
            ret, frame = cap.read()
            if not ret:
                break
            vw.write(frame)
            n += 1
            prog.progress(min((time.time() - t0) / seconds, 1.0),
                          text=f"Recording... {n} frames")
    cap.release()
    vw.release()
    prog.empty()
    if n == 0:
        st.error("No frames captured from RTSP.")
        return None
    return out


@torch.no_grad()
def _diagnose(model, names, frame_bgr, imgsz=640):
    """Raw model output stats on one frame: max score, box counts per threshold, top-5."""
    import numpy as np
    model.eval()
    legacy = bool(getattr(model, "yolox_legacy", False))
    h0, w0 = frame_bgr.shape[:2]
    s = min(imgsz / h0, imgsz / w0)
    if legacy:
        img = cv2.resize(frame_bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
    else:
        canvas, s, (_dw, _dh) = letterbox(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), imgsz)
    t = torch.from_numpy(canvas.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    t = t.contiguous(memory_format=torch.channels_last)
    dev = _model_device(model)
    with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
        raw = model(t.to(dev, non_blocking=True))
    if legacy:
        r = raw[0]
        scores = (r[:, 4:5] * r[:, 5:])  # obj * cls  [N, nc]
    else:
        scores = raw[0][:, 4:]          # cls scores [N, nc]
    best, _ = scores.max(-1)
    counts = {th: int((best > th).sum()) for th in (0.01, 0.05, 0.10, 0.25, 0.50)}
    k = min(5, best.shape[0])
    vals, idx = torch.topk(best, k)
    top = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        c = int(scores[i].argmax())
        top.append({"score": round(v, 4),
                    "class": names[c] if c < len(names) else c})
    return {"max_score": round(float(best.max()), 4), "counts": counts, "top5": top,
            "arch": "yolox-legacy (BGR)" if legacy else "yolov8 (RGB)"}


def _process_video(model, names, mp, conf, iou, imgsz, batch=4, stride=1,
                   max_frames=1200):
    """Fast batched inference. stride>1 runs the model every Nth frame and
    reuses the last boxes for frames in between."""
    import time
    cap = cv2.VideoCapture(str(mp))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, total_det, done, infer_calls = [], 0, 0, 0
    mid_raw, mid_idx = None, (n_frames // 2 if n_frames > 0 else 0)
    prog = st.progress(0, text="Processing video...")
    t0 = time.time()
    acc, last = [], None  # acc: [(frame, sampled)]
    empty = np_empty_dets()

    def flush():
        nonlocal last, total_det, infer_calls
        if not acc:
            return []
        sampled = [f for f, s in acc if s]
        new_dets = {}
        if sampled:
            infer_calls += 1
            for f, (_, dets) in zip(sampled, _infer_batch(model, sampled, names, conf, iou, imgsz)):
                new_dets[id(f)] = dets
        outs = []
        for f, s in acc:
            if s:
                last = new_dets[id(f)]
            dets = last if last is not None else empty
            total_det += len(dets) if s else 0
            outs.append((_shrink(_draw(f, dets, names)), dets))
        acc.clear()
        return outs

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            for (out, _d) in flush():
                frames.append(_jpg(out))
            break
        acc.append((frame, idx % stride == 0))
        if idx == mid_idx:
            mid_raw = frame.copy()
        if sum(1 for _, s in acc if s) >= batch:
            for (out, _d) in flush():
                frames.append(_jpg(out))
            done = len(frames)
        idx += 1
        if n_frames > 0:
            prog.progress(min(len(frames) / n_frames, 1.0),
                          text=f"Frame {len(frames)}/{n_frames} · {len(frames)/max(time.time()-t0,1e-3):.1f} fps")
    cap.release()
    if len(frames) >= max_frames:
        st.warning(f"Trimmed to first {max_frames} frames.")
    prog.empty()
    speed = len(frames) / max(time.time() - t0, 1e-3)
    return frames, fps, total_det, len(frames), infer_calls, speed, mid_raw


def np_empty_dets():
    import numpy as np
    return np.zeros((0, 6), dtype=np.float32)


def _shrink(out):
    if out.shape[1] <= 640:
        return out
    return cv2.resize(out, (640, int(out.shape[0] * 640 / out.shape[1])))


def _jpg(out):
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _write_mp4(frames_jpg, fps, out_path):
    import numpy as np
    first = cv2.imdecode(np.frombuffer(frames_jpg[0], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for b in frames_jpg:
        vw.write(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
    vw.release()


def _frames_from_mp4(mp4_path, max_frames=1200, max_w=640):
    """Decode an mp4 into preview JPEG bytes + fps (for the local player)."""
    import numpy as np
    cap = cv2.VideoCapture(str(mp4_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] > max_w:
            frame = cv2.resize(frame, (max_w, int(frame.shape[0] * max_w / frame.shape[1])))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frames.append(buf.tobytes())
    cap.release()
    return frames, fps


def _ssh_alive():
    try:
        c = st.session_state.get("ssh")
        return c is not None and c.get_transport() is not None and c.get_transport().is_active()
    except Exception:
        return False


def _export_onnx_tmp(model, imgsz=640):
    """Export the session model to a temp single-file ONNX (for GPU shipping)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _export_onnx_tmp_inner(model, imgsz)


def _export_onnx_tmp_inner(model, imgsz=640):
    import torch
    tmp = Path(tempfile.mkdtemp()) / "model_ship.onnx"
    was_training = model.training
    model.eval()
    dev = _model_device(model)
    model_cpu = model.to("cpu").float()
    dummy = torch.randn(1, 3, imgsz, imgsz)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    torch.onnx.export(model_cpu, dummy, str(tmp), opset_version=18,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      dynamo=False)
    model.to(dev)
    if was_training:
        model.train()
    return tmp


def _ship_model_to_gpu():
    """Export session model to ONNX and upload + script to the GPU box."""
    if st.session_state.model is None or not _ssh_alive():
        st.error("Load a model and connect to the GPU first.")
        return
    try:
        with st.spinner("Exporting ONNX + shipping to GPU..."):
            from tools.remote_gpu import prepare_job, ensure_env
            onnx_path = _export_onnx_tmp(st.session_state.model)
            tmp = Path(tempfile.mkdtemp())
            cls_txt = tmp / "classes.txt"
            cls_txt.write_text("\n".join(st.session_state.names), encoding="utf-8")
            st.session_state.cls_txt = str(cls_txt)
            st.session_state.onnx_path = str(onnx_path)
            prepare_job(st.session_state.ssh, onnx_path)
            st.session_state.remote_ready = True
            st.session_state.remote_log = "model.onnx + infer_onnx.py shipped to ~/yolo_remote"
            st.success("✅ Model shipped to GPU — videos will run remotely.")
    except Exception as e:
        st.session_state.remote_ready = False
        st.error(f"Ship failed: {e}")


def _load_video_remote(mp, conf, iou, imgsz, batch, stride):
    """Upload video, run infer_onnx.py on the GPU, download result, fill player."""
    import numpy as np
    try:
        with st.spinner("Checking GPU packages..."):
            from tools.remote_gpu import run_remote_infer, ensure_env
            env_log = ensure_env(st.session_state.ssh)
            st.session_state.remote_log = "\n".join(env_log)
            if any("INSTALL FAILED" in l for l in env_log):
                st.error("Server packages missing and auto-install failed. On the server run once:\n"
                         "python3 -m pip install onnxruntime-gpu opencv-python numpy")
                with st.expander("Server environment log"):
                    st.code(st.session_state.remote_log)
                return
        with st.spinner("Running inference on GPU server..."):
            cls_txt = st.session_state.cls_txt
            if cls_txt is None:
                tmp = Path(tempfile.mkdtemp()) / "classes.txt"
                tmp.write_text("\n".join(st.session_state.names), encoding="utf-8")
                cls_txt = str(tmp)
            local_out, log = run_remote_infer(
                st.session_state.ssh, mp, cls_txt, conf, iou, imgsz,
                batch, stride, remote=None)
            st.session_state.remote_log = log
            frames, fps = _frames_from_mp4(local_out)
            cap = cv2.VideoCapture(str(local_out))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(frames))
            mid = min(len(frames) - 1, max(0, n // 2))
            cap.release()
            st.session_state.update(
                frames=frames, fps=fps, frame=0, playing=False,
                video_bytes=Path(local_out).read_bytes(),
                mid_raw=frames[mid] if frames else None,
                stats=f"🖥️ GPU SERVER · {len(frames)} frames · GPU result ({log.splitlines()[-1] if log else ''})")
            st.session_state.scrub = 0
        with st.expander("🖥️ GPU run log"):
            st.code(st.session_state.remote_log or "(empty)")
    except Exception as e:
        st.error(f"Remote inference failed (falling back to local next time): {e}")
        st.session_state.remote_ready = False


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
