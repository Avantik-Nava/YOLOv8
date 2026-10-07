"""IoU tracker (ByteTrack-lite/SORT-lite): track_id assignment for mode=track."""

import numpy as np


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-7
    return inter / ua


class IoUTracker:
    """Greedy IoU matching. dets [M,6] (x1,y1,x2,y2,conf,cls) -> [M,7] (+track_id)."""

    def __init__(self, iou_thres=0.3, max_age=30):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.tracks = []  # dicts: box, cls, id, age
        self.next_id = 1

    def update(self, dets):
        dets = np.asarray(dets).reshape(-1, 6) if len(dets) else np.zeros((0, 6))
        out = []
        used = set()
        for d in dets:
            best, bj = self.iou_thres, -1
            for j, t in enumerate(self.tracks):
                if j in used or t["cls"] != d[5]:
                    continue
                v = _iou(d[:4], t["box"])
                if v > best:
                    best, bj = v, j
            if bj >= 0:
                t = self.tracks[bj]
                t["box"] = d[:4].tolist()
                t["age"] = 0
                used.add(bj)
                tid = t["id"]
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks.append({"box": d[:4].tolist(), "cls": d[5], "id": tid, "age": 0})
            out.append([*d.tolist(), tid])
        for j, t in enumerate(self.tracks):
            if j not in used:
                t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return np.array(out, dtype=np.float32).reshape(-1, 7)
