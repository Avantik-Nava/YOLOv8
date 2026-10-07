#!/usr/bin/env python3
"""Full YOLOv8 test: all 5 tasks + augment + tracker + benchmark + COCO eval + YOLO API.
Run: python test_full.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch


def ok(msg):
    print(f"  [OK] {msg}")


def test_tasks():
    from yolox.models.tasks import create_model
    x = torch.randn(1, 3, 256, 256)
    m = create_model("detect", "n", 3).eval()
    with torch.no_grad():
        assert m(x).shape[-1] == 7
    ok("detect [B,N,4+nc]")
    m = create_model("segment", "n", 3).eval()
    with torch.no_grad():
        pred, mc, proto = m(x)
        assert mc.shape[-1] == 32 and proto.shape[1] == 32
    ok(f"segment pred {tuple(pred.shape)} mc {tuple(mc.shape)} proto {tuple(proto.shape)}")
    m = create_model("classify", "n", 10).eval()
    with torch.no_grad():
        assert m(x).shape == (1, 10)
    ok("classify logits [B,nc]")
    m = create_model("pose", "n", 1, nkpt=17).eval()
    with torch.no_grad():
        pred, kpts = m(x)
        assert kpts.shape[-2:] == (17, 3)
    ok(f"pose kpts {tuple(kpts.shape)}")
    m = create_model("obb", "n", 5).eval()
    with torch.no_grad():
        pred, ang = m(x)
        assert ang.shape[-1] == 1
    ok(f"obb angle {tuple(ang.shape)}")


def test_train_losses():
    from yolox.models.tasks import create_model
    md = create_model("detect", "n", 3)
    md.train()
    lb = torch.full((1, 2, 5), -1.0)
    lb[0, 0] = torch.tensor([0, 0.5, 0.5, 0.2, 0.2])
    out = md(torch.randn(1, 3, 256, 256), lb)
    out["total_loss"].backward()
    ok(f"detect loss {out['total_loss_value']:.3f}")
    ms = create_model("segment", "n", 3)
    ms.train()
    out = ms(torch.randn(1, 3, 256, 256), lb, masks=torch.rand(1, 2, 160, 160))
    out["total_loss"].backward()
    ok(f"segment loss {out['total_loss_value']:.3f} mask {out['mask_loss']:.3f}")
    mc = create_model("classify", "n", 4)
    mc.train()
    out = mc(torch.randn(2, 3, 224, 224), torch.tensor([1, 2]))
    out["total_loss"].backward()
    ok(f"classify loss {out['total_loss_value']:.3f}")


def test_augment_tracker_eval():
    import numpy as np
    from yolox.data import mosaic4, mixup, augment_hsv
    s = [(np.full((320, 320, 3), 128, np.uint8), np.array([[0, 0.5, 0.5, 0.2, 0.2]], np.float32))] * 4
    img, lb = mosaic4(s, 640)
    assert img.shape == (640, 640, 3) and len(lb) >= 1
    ok(f"mosaic {img.shape} boxes={len(lb)}")
    img2, lb2 = mixup(s[0][0], s[0][1], s[1][0], s[1][1])
    ok("mixup + hsv")
    augment_hsv(img)
    from yolox.engine.tracker import IoUTracker
    tr = IoUTracker()
    t = tr.update(np.array([[10, 10, 50, 50, 0.9, 0]]))
    t = tr.update(np.array([[12, 12, 52, 52, 0.9, 0]]))
    assert t[0, 6] == 1
    ok(f"tracker id-stable {t[0,6]:.0f}")
    from yolox.utils import coco_map
    m = coco_map([np.array([[10, 10, 50, 50, 0.9, 0]])], [np.array([[0, 10, 10, 50, 50]])])
    ok(f"coco mAP50={m['map50']:.2f} mAP50-95={m['map5095']:.2f}")


def test_api_benchmark():
    from yolox.engine import YOLO
    m = YOLO("yolov8n.pt", num_classes=3)
    m.info()
    ok("YOLO('yolov8n.pt') detect/n parsed")
    m = YOLO("yolov8n-seg.pt", num_classes=3)
    assert m.task == "segment"
    ok("YOLO('yolov8n-seg.pt') task=segment")
    r = m.benchmark(imgsz=256, runs=5)
    ok(f"benchmark {r['latency_ms']:.1f}ms {r['fps']:.0f}FPS params={r['params']:,}")


def main():
    print("=" * 60 + "\nFull YOLOv8 test (5 tasks + modes)\n" + "=" * 60)
    tests = [("tasks fwd", test_tasks), ("task losses", test_train_losses),
             ("aug/track/eval", test_augment_tracker_eval), ("YOLO API+bench", test_api_benchmark)]
    good = True
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [FAIL] {e}")
            good = False
    print("=" * 60 + ("\nALL FULL TESTS PASSED" if good else "\nSOME FAILED") + "\n" + "=" * 60)
    return good


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
