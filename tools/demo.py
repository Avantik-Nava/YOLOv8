"""tools/demo.py — image/video/webcam inference (YOLOX tools/demo.py style)."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo.data import letterbox  # noqa: E402


def load_model(version, num_classes, ckpt, device):
    from yolo.models import YOLOv8
    model = YOLOv8(version=version, num_classes=num_classes).to(device).eval()
    if ckpt and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        print(f"loaded {ckpt}")
    else:
        print("no checkpoint — using random init (demo of pipeline only)")
    return model


def draw(img, dets, names):
    for x1, y1, x2, y2, conf, cls in dets.tolist():
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{names[int(cls)] if int(cls) < len(names) else int(cls)} {conf:.2f}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="demo.jpg", help="image / video / 0 for webcam")
    p.add_argument("--version", default="s")
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", default="demo_out.jpg")
    p.add_argument("--names", nargs="*", default=["object"])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.version, args.num_classes, args.ckpt, device)

    if args.path == "0" or str(args.path).endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(0 if args.path == "0" else args.path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lb, s, (dw, dh) = letterbox(rgb, args.imgsz)
            t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            t0 = time.time()
            dets = model.predict(t, conf_threshold=args.conf)[0]
            print(f"{len(dets)} dets in {(time.time()-t0)*1000:.1f}ms")
            # map back from letterbox
            if len(dets):
                dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s
                dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
            frame = draw(frame, dets.cpu(), args.names)
            cv2.imshow("yolov8 demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        import numpy as np
        img0 = cv2.imread(args.path)
        if img0 is None:
            print(f"image not found: {args.path} — running on dummy image")
            img0 = np.full((640, 640, 3), 128, np.uint8)
        h0, w0 = img0.shape[:2]
        rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        lb, s, (dw, dh) = letterbox(rgb, args.imgsz)
        t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        dets = model.predict(t, conf_threshold=args.conf)[0]
        if len(dets):
            dets[:, [0, 2]] = (dets[:, [0, 2]] - dw) / s * (w0 / args.imgsz * args.imgsz / w0)
            dets[:, [1, 3]] = (dets[:, [1, 3]] - dh) / s
            # simpler exact: scale = min(imgsz/w0, imgsz/h0) already s; reuse
        print(dets)
        out = draw(img0, dets.cpu(), args.names)
        cv2.imwrite(args.save, out)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
