#!/usr/bin/env python3
"""Standalone ONNX video inference — the ONLY file (plus model.onnx + video)
that needs to go to the GPU server. No torch, no repo code needed.

Server setup:
  pip install onnxruntime-gpu opencv-python numpy
Run:
  python infer_onnx.py --model yolox_s5.onnx --source in.mp4 --out result.mp4 \\
      --classes classes.txt --conf 0.25 --imgsz 640 --batch 8 --stride 2
Needs: onnxruntime(-gpu), opencv-python, numpy.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def load_classes(path):
    if not path:
        return []
    names = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    if names and names[0].startswith("nc:"):
        import yaml  # ultralytics yaml fallback
        return [str(n) for n in yaml.safe_load("\n".join(names)).get("names", [])]
    return names


def preprocess_legacy(frame, imgsz):
    h, w = frame.shape[:2]
    s = min(imgsz / h, imgsz / w)
    img = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img  # BGR, top-left (YOLOX preproc)
    return canvas.transpose(2, 0, 1).astype(np.float32) / 255.0, s, 0, 0


def preprocess_v8(frame, imgsz):
    h, w = frame.shape[:2]
    s = min(imgsz / h, imgsz / w)
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    dw, dh = (imgsz - img.shape[1]) // 2, (imgsz - img.shape[0]) // 2
    canvas[dh:dh + img.shape[0], dw:dw + img.shape[1]] = img
    return canvas.transpose(2, 0, 1).astype(np.float32) / 255.0, s, dw, dh


def nms(boxes, scores, thres):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= thres]
    return np.array(keep, dtype=np.int64)


def decode(pred, legacy, conf, nms_thres):
    """pred [N, C] -> [M, 6] (x1,y1,x2,y2,conf,cls) in letterboxed pixels."""
    if legacy:
        boxes, obj, cls = pred[:, :4], pred[:, 4:5], pred[:, 5:]
        scores = obj * cls
    else:
        boxes, scores = pred[:, :4], pred[:, 4:]
    conf_max = scores.max(-1)
    cls_id = scores.argmax(-1)
    mask = conf_max > conf
    boxes, conf_max, cls_id = boxes[mask], conf_max[mask], cls_id[mask]
    if len(boxes) == 0:
        return np.zeros((0, 6), np.float32)
    if legacy:  # cxcywh -> xyxy
        boxes = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                          boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], -1)
    if len(boxes) > 2000:
        top = np.argsort(-conf_max)[:2000]
        boxes, conf_max, cls_id = boxes[top], conf_max[top], cls_id[top]
    rows = []
    for c in np.unique(cls_id):
        m = cls_id == c
        for i in nms(boxes[m], conf_max[m], nms_thres):
            j = np.where(m)[0][i]
            rows.append([*boxes[j], conf_max[j], c])
    if not rows:
        return np.zeros((0, 6), np.float32)
    det = np.array(rows, np.float32)
    return det[np.argsort(-det[:, 4])]


def draw(frame, dets, names):
    for x1, y1, x2, y2, cf, cls in dets:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)] if int(cls) < len(names) else int(cls)} {cf:.2f}"
        cv2.putText(frame, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def main():
    p = argparse.ArgumentParser("standalone ONNX video inference")
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="result.mp4")
    p.add_argument("--classes", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=100000)
    args = p.parse_args()

    import onnxruntime as ort
    names = load_classes(args.classes)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(args.model, providers=providers)
    print("provider:", sess.get_providers()[0])
    in_name = sess.get_inputs()[0].name
    dims = [d for d in sess.get_outputs()[0].shape if isinstance(d, int)]
    if dims:
        n_out = dims[-1]  # last concrete dim = channels
    else:  # fully symbolic shape -> probe with a dummy forward
        probe = sess.run(None, {in_name: np.zeros((1, 3, args.imgsz, args.imgsz),
                                                  dtype=np.float32)})[0]
        n_out = probe.shape[-1]
    if names and n_out - 5 == len(names):
        legacy = True
    elif names and n_out - 4 == len(names):
        legacy = False
    else:
        raise ValueError(f"output dim {n_out} matches neither legacy (5+nc) nor v8 (4+nc) "
                         f"for {len(names)} classes — check --classes")
    print(f"arch: {'yolox-legacy' if legacy else 'yolov8'} nc={len(names)}")
    pre = preprocess_legacy if legacy else preprocess_v8

    cap = cv2.VideoCapture(args.source)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    acc, last, done, total, calls, t0 = [], None, 0, 0, 0, time.time()

    def flush():
        nonlocal last, total, calls
        if not acc:
            return
        items = [(f, s, pre(f, args.imgsz)) for f, s in acc]
        sampled = [(f, m) for f, s, m in items if s]
        new = {}
        if sampled:
            calls += 1
            batch_in = np.stack([m[0] for _, m in sampled])
            for (f, _), pr in zip(sampled, sess.run(None, {in_name: batch_in})[0]):
                new[id(f)] = pr
        for f, s, (_, sc, dw, dh) in items:
            if s:
                last = new[id(f)]
            dets = decode(last, legacy, args.conf, args.nms) if last is not None else \
                np.zeros((0, 6), np.float32)
            if s:
                total += len(dets)
            d = dets.copy()
            d[:, [0, 2]] = (d[:, [0, 2]] - dw) / sc
            d[:, [1, 3]] = (d[:, [1, 3]] - dh) / sc
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, W)
            d[:, [1, 3]] = d[:, [1, 3]].clip(0, H)
            vw.write(draw(f, d, names))
        acc.clear()

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= args.max_frames:
            break
        acc.append((frame, idx % args.stride == 0))
        if sum(1 for _, s in acc if s) >= args.batch:
            flush()
        idx += 1
        if idx % 25 == 0:
            el = time.time() - t0
            print(f"frame {idx} · {idx / el:.1f} fps", flush=True)
    flush()
    cap.release()
    vw.release()
    el = time.time() - t0
    print(f"DONE frames={idx} detections={total} forwards={calls} "
          f"speed={idx / el:.1f} fps -> {args.out}")


if __name__ == "__main__":
    main()
