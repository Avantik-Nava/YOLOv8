#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Test script for YOLOv8 (YOLOX-style layout).
Mirrors YOLOX/test_model.py: verifies blocks, backbone, neck, head,
TAL assigner, DFL loss, full model eval + train.

Run:  python test_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch


def test_c2f():
    print("Testing C2f block...")
    from yolox.models import C2f
    m = C2f(64, 64, n=1)
    x = torch.randn(1, 64, 80, 80)
    y = m(x)
    assert y.shape == (1, 64, 80, 80), y.shape
    print(f"  [OK] C2f {tuple(x.shape)} -> {tuple(y.shape)}")
    return True


def test_sppf():
    print("\nTesting SPPF...")
    from yolox.models import SPPF
    m = SPPF(256, 256)
    y = m(torch.randn(1, 256, 20, 20))
    assert y.shape == (1, 256, 20, 20), y.shape
    print(f"  [OK] SPPF out {tuple(y.shape)}")
    return True


def test_backbone():
    print("\nTesting CSPDarknet-C2f backbone...")
    from yolox.models import CSPDarknetC2f
    for v in ["n", "s"]:
        m = CSPDarknetC2f(v).eval()
        with torch.no_grad():
            p3, p4, p5 = m(torch.randn(1, 3, 640, 640))
        print(f"  [OK] yolov8-{v}: P3 {tuple(p3.shape)} P4 {tuple(p4.shape)} P5 {tuple(p5.shape)}")
        assert p3.shape[-1] == 80 and p4.shape[-1] == 40 and p5.shape[-1] == 20
    return True


def test_neck():
    print("\nTesting YOLOv8 PAN-FPN neck...")
    from yolox.models import CSPDarknetC2f
    from yolox.models.neck import YOLOv8Neck
    bb = CSPDarknetC2f("s").eval()
    neck = YOLOv8Neck(bb.out_channels, "s").eval()
    with torch.no_grad():
        feats = bb(torch.randn(1, 3, 640, 640))
        out = neck(list(feats))
    for i, f in enumerate(out):
        print(f"  [OK] neck P{i+3} {tuple(f.shape)}")
    assert len(out) == 3
    return True


def test_head():
    print("\nTesting YOLOv8 anchor-free DFL head (no obj branch)...")
    from yolox.models import CSPDarknetC2f, YOLOv8Head
    from yolox.models.neck import YOLOv8Neck
    bb = CSPDarknetC2f("s").eval()
    neck = YOLOv8Neck(bb.out_channels, "s").eval()
    head = YOLOv8Head(80, neck.out_channels, reg_max=16).eval()
    with torch.no_grad():
        feats = neck(list(bb(torch.randn(1, 3, 640, 640))))
        cls_l, reg_l = head(feats)
        assert len(cls_l) == 3 and len(reg_l) == 3
        assert cls_l[0].shape[1] == 80 and reg_l[0].shape[1] == 4 * 16
        pred = head.predict(feats)
    print(f"  [OK] cls ch={cls_l[0].shape[1]} reg ch={reg_l[0].shape[1]} (4*reg_max, no obj)")
    print(f"  [OK] decoded {tuple(pred.shape)} (B,N,4+nc)")
    assert pred.shape[-1] == 4 + 80
    return True


def test_tal_dfl():
    print("\nTesting TaskAlignedAssigner + DFLoss...")
    from yolox.models import TaskAlignedAssigner, DFLoss
    tal = TaskAlignedAssigner(topk=10)
    B, N, C, M = 1, 8400, 80, 3
    ps = torch.rand(B, N, C)
    pb = torch.rand(B, N, 4) * 640
    pb[..., 2:] += pb[..., :2]
    ap = torch.rand(N, 2) * 640
    gl = torch.randint(0, C, (B, M, 1))
    gb = torch.rand(B, M, 4) * 500
    gb[..., 2:] += gb[..., :2] + 20
    mg = torch.ones(B, M, 1, dtype=torch.bool)
    tl, tb, ts, fg, nfg = tal(ps, pb, ap, gl, gb, mg)
    print(f"  [OK] TAL fg={nfg} fg_mask {tuple(fg.shape)}")
    dfl = DFLoss(16)
    loss = dfl(torch.randn(20, 16), torch.rand(20) * 15)
    print(f"  [OK] DFL loss={loss.item():.4f}")
    return True


def test_model_eval():
    print("\nTesting YOLOv8 full model (eval)...")
    from yolox.models import YOLOv8
    for v in ["n", "s"]:
        m = YOLOv8(version=v, num_classes=80).eval()
        n_params = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out = m(torch.randn(1, 3, 640, 640))
        print(f"  [OK] yolov8-{v} params={n_params:,} out={tuple(out.shape)}")
    return True


def test_model_train():
    print("\nTesting YOLOv8 full model (train + loss)...")
    from yolox.models import YOLOv8
    m = YOLOv8(version="s", num_classes=3)
    m.train()
    x = torch.randn(2, 3, 640, 640)
    labels = torch.full((2, 10, 5), -1.0)  # (cls,cx,cy,w,h) norm, -1 = pad
    labels[0, 0] = torch.tensor([0, 0.5, 0.5, 0.1, 0.1])
    labels[0, 1] = torch.tensor([1, 0.3, 0.7, 0.2, 0.15])
    labels[1, 0] = torch.tensor([2, 0.6, 0.4, 0.1, 0.1])
    out = m(x, labels)
    loss = out["total_loss"]
    loss.backward()
    print(f"  [OK] total={float(loss.item()):.4f} box={out['iou_loss']:.4f} "
          f"cls={out['cls_loss']:.4f} dfl={out['dfl_loss']:.4f} fg={out.get('num_fg')}")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad, "no gradients flowed"
    print("  [OK] backward pass, gradients flow")
    return True


def test_nms_predict():
    print("\nTesting predict() + NMS...")
    from yolox.models import YOLOv8
    m = YOLOv8(version="n", num_classes=3).eval()
    with torch.no_grad():
        dets = m.predict(torch.randn(1, 3, 640, 640), conf_threshold=0.25)
    print(f"  [OK] {len(dets)} image(s), det shape {tuple(dets[0].shape)} [x1,y1,x2,y2,conf,cls]")
    return True


def main():
    print("=" * 60)
    print("YOLOv8 (YOLOX-style) Implementation Test")
    print("Ref: https://docs.ultralytics.com/models/yolov8")
    print("=" * 60)
    tests = [
        ("C2f Block", test_c2f),
        ("SPPF", test_sppf),
        ("Backbone", test_backbone),
        ("Neck", test_neck),
        ("Head (anchor-free+DFL)", test_head),
        ("TAL + DFLoss", test_tal_dfl),
        ("Model (Eval)", test_model_eval),
        ("Model (Train)", test_model_train),
        ("Predict + NMS", test_nms_predict),
    ]
    results = []
    for name, fn in tests:
        print(f"\n{'=' * 60}\nRunning {name}...\n{'=' * 60}")
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    print("\n" + "=" * 60 + "\nTest Summary\n" + "=" * 60)
    ok = True
    for name, passed in results:
        print(f"  {name}: {'[PASS]' if passed else '[FAIL]'}")
        ok &= bool(passed)
    print("=" * 60)
    print("All tests passed! YOLOv8 setup OK." if ok else "Some tests failed.")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
