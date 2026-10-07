"""tools/track.py — video/webcam tracking (mode=track)."""

import argparse
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    from yolo.data import letterbox
    from yolo.engine import YOLO
    from yolo.engine.tracker import IoUTracker

    m = YOLO(args.weights, num_classes=args.num_classes)
    m.model.eval()
    tr = IoUTracker()
    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lb, _, _ = letterbox(rgb, args.imgsz)
        t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(m.device) / 255.0
        with torch.no_grad():
            dets = m.model.predict(t, conf_threshold=args.conf)[0].cpu().numpy()
        tracks = tr.update(dets)
        for x1, y1, x2, y2, conf, cls, tid in tracks:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"id{int(tid)} {conf:.2f}", (int(x1), max(int(y1) - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("track (q=quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
