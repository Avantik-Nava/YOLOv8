# YOLOv8 in exact YOLOX structure

Official YOLOv8 (Ultralytics, Jan 2023) with **all YOLOv8 features**, laid out
exactly like official YOLOX (`yolox/`, `tools/`, `exps/`, `YOLOX_outputs/`).

- Docs reference: https://docs.ultralytics.com/models/yolov8
- Train exactly like YOLOX: `python tools/train.py -f exps/default/yolov8_s.py`

## Structure (same as YOLOX official)

```
YOLOv8/
├── test_model.py / test_full.py
├── yolo_cli.py / tools/app.py (local validation UI)
├── data/coco8.yaml seg.yaml pose.yaml obb.yaml custom.yaml
├── exps/default/yolov8_n/s/m/l/x.py   # depth/width like yolox_s.py
├── exps/example/yolox_voc/yolov8_voc_s.py
├── tools/train.py  # -f -n -b -d -c --resume --fp16 -o opts (YOLOX flags)
├── tools/eval.py   # -f -n -c -b --test-conf --nmsthre (YOLOX flags)
├── tools/demo.py   # image/video/webcam (YOLOX flags)
└── yolox/          # like YOLOX yolox/ package
    ├── exp/        # base_exp.py (BaseExp) + yolov8_base.py (Exp) + build.py
    ├── models/     # backbone/neck/head (C2f, DFL) + seg/cls/pose/obb + tasks.py
    ├── data/datasets/  # voc.py voc_classes.py coco.py coco_classes.py
    ├── evaluators/ # COCOEvaluator / VOCEvaluator
    ├── core/       # Trainer (warmup, AMP, EMA, no_aug closing, early-stop)
    ├── utils/      # boxes.py ema.py metrics.py coco_eval.py lr_scheduler.py
    └── engine/     # YOLO() API + tracker.py + benchmark.py
```

YOLOX official for comparison (`YOLOX/YOLOX/`): `yolox/ tools/ exps/ datasets/ demo/`

## YOLOv8 key facts (from docs)

- Anchor-free split head, **no objectness branch** (unlike YOLOX which has obj).
- Backbone/neck use **C2f** blocks + SPPF; head outputs `cls (nc)` + `reg (4*reg_max)` with **DFL (reg_max=16)**.
- Assigner: **TaskAlignedAssigner (topk=10)**; losses: **BCE cls + CIoU box + DFL**.
- Input 640; close mosaic last 10 epochs; SGD lr=0.01 momentum=0.937.

| Model | size | mAPval 50-95 | CPU ONNX (ms) | A100 TRT (ms) | params (M) | FLOPs (B) |
|-------|------|--------------|---------------|---------------|------------|-----------|
| YOLOv8n | 640 | 37.3 | 80.4 | 0.99 | 3.2 | 8.7 |
| YOLOv8s | 640 | 44.9 | 128.4 | 1.20 | 11.2 | 28.6 |
| YOLOv8m | 640 | 50.2 | 234.7 | 1.83 | 25.9 | 78.9 |
| YOLOv8l | 640 | 52.9 | 375.2 | 2.39 | 43.7 | 165.2 |
| YOLOv8x | 640 | 53.9 | 479.1 | 3.53 | 68.2 | 257.8 |

Variants here: `n: d=0.33 w=0.25 | s: d=0.33 w=0.50 | m: d=0.67 w=0.75 | l: d=1.0 w=1.0 | x: d=1.0 w=1.25`.

## Quick start

```bash
cd YOLOv8
pip install -r requirements.txt

# 1. sanity check
python test_model.py && python test_full.py

# 2. train exactly like YOLOX (YOLO-format dataset via data/custom.yaml):
python tools/train.py -f exps/default/yolov8_s.py -b 16
python tools/train.py -n yolov8-n -b 16 --fp16 -o num_classes=3 max_epoch=50
python tools/train.py -f exps/default/yolov8_s.py -c YOLOX_outputs/yolov8_s/last.pth --resume

# 3. eval / demo like YOLOX
python tools/eval.py -f exps/default/yolov8_s.py -c YOLOX_outputs/yolov8_s/best.pth -b 16
python tools/demo.py image -f exps/default/yolov8_s.py -c <ckpt> --path image.jpg --conf 0.25 --save_result
python tools/track.py --source 0 --weights YOLOX_outputs/yolov8_s/last.pth
python tools/benchmark.py --weights yolov8n.pt --imgsz 640 --runs 50
python tools/export.py --weights yolov8n.pt --format onnx -o yolov8n.onnx

# 4. local validation UI — upload class file + exp file + weights + image/video
streamlit run tools/app.py
# open http://localhost:8501, upload the 4 files, press Run inference
```

Full API (no `pip install ultralytics` needed — built in `yolox/engine/api.py`):

```python
from yolox.engine import YOLO
model = YOLO("yolov8n.pt"); model.info()
model.train(data="data/coco8.yaml", epochs=100, imgsz=640)
model.predict("image.jpg"); model.val(); model.export(format="onnx")
model.track("video.mp4"); model.benchmark()
# tasks: YOLO("yolov8n-seg.pt") / YOLO("yolov8n-cls.pt") / YOLO("yolov8n-pose.pt") / YOLO("yolov8n-obb.pt")
# CLI:   python yolo_cli.py task=segment mode=train model=yolov8s.pt data=data/seg.yaml epochs=10
```

Ultralytics equivalent (for reference, needs `pip install ultralytics`):

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
model("path/to/bus.jpg")
```

## Custom dataset (YOLO format)

```
dataset/
  images/train/*.jpg  images/val/*.jpg
  labels/train/*.txt  labels/val/*.txt   # one line per box: cls cx cy w h (normalized)
```

Edit `data/custom.yaml` (`path/nc/names`) + `config.yaml` (`num_classes`) — same flow as your `YOLOv8_Test/` + `YOLOv26` setup.

## VOC workflow — change only 3 files (like YOLOX)

Same as YOLOX (`voc_classes.py`, `yolox_voc_s.py`, `voc.py`). You never touch the training code:

| # | File | What you edit |
|---|------|---------------|
| 1 | `yolox/data/datasets/voc_classes.py` | class names, e.g. `VOC_CLASSES = ("person", "stick")` |
| 2 | `exps/example/yolox_voc/yolov8_voc_s.py` | `num_classes`, `depth`/`width`, `data_dir`, `train_sets`/`val_sets` |
| 3 | `yolox/data/datasets/voc.py` | only if your folder layout differs from `VOCdevkit/` |

```bash
python tools/train.py -f exps/example/yolox_voc/yolov8_voc_s.py -b 16
python tools/eval.py -f exps/example/yolox_voc/yolov8_voc_s.py -c YOLOX_outputs/yolov8_voc_s/best.pth
```

## YOLOX vs YOLOv8 (what changed in code)

- Backbone block: YOLOX `CSPBlock` → YOLOv8 `C2f`; SPPF kept.
- Neck: YOLOX FPN+PAN with CSP → YOLOv8 PAN-FPN with `C2f`, `Upsample(2x)` (this repo fixes YOLOv26's 4x/stride bug → proper P3/8 P4/16 P5/32).
- Head: YOLOX `cls+reg+obj` → YOLOv8 `cls+reg(DFL)`, anchor-free, no anchors file.
- Assigner/loss: YOLOX SimOTA → YOLOv8 **TaskAlignedAssigner + BCE + CIoU + DFL**.
```

