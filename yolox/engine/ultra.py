#!/usr/bin/env python3
"""Real-Ultralytics backend inside YOLOX flow (YOLOX code, NOT YOLOX features).

YOLOX code used: tools/train.py -f/-b/-c/--resume/-o, exps/*Exp, data/*.yaml,
datasets/ layout, YOLOX_outputs/ dir.
Ultralytics exact code does the model: C2f/SPPF/PAN-DFL/TAL/BCE+CIoU+DFL,
train/val/predict/export/track — no SimOTA, no obj branch, no CSPBlock.

Usage (same YOLOX CLI):
  python tools/train.py -f exps/default/yolov8_ultra_n.py -b 4
  python tools/train.py -f exps/default/yolov8_ultra_n.py -c yolov8n.pt -b 8
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _abs_data_yaml(exp_yaml="data/custom.yaml"):
    """Ultralytics needs train:/val: image dirs + path/nc/names.
    Our yaml uses YOLOX keys (train_images/train_labels/...), so translate
    to a tmp ultralytics-style yaml with absolute dirs. One yaml, both flows."""
    import tempfile
    import yaml
    src = Path(exp_yaml)
    d = yaml.safe_load(open(src))
    root = Path(d.get("path", "."))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    ti = root / d.get("train_images", "images/train")
    vi = root / d.get("val_images", "images/val")
    ud = {"path": str(root), "train": str(ti), "val": str(vi),
          "nc": d.get("nc"), "names": d.get("names")}
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(ud, tmp)
    tmp.close()
    return tmp.name, ud


def _voc_to_yolo_tmp(data_dir, train_sets, val_sets):
    """VOCdevkit (PRIMARY) -> tmp YOLO-txt dataset for real ultralytics.
    datasets/VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets} -> tmp/{images,labels}/{train,val}.
    Class ids = VOC_CLASSES order. Backup COCO/YOLO-txt untouched."""
    import shutil
    import tempfile
    import xml.etree.ElementTree as ET
    from yolox.data.datasets import VOC_CLASSES
    cls2id = {n: i for i, n in enumerate(VOC_CLASSES)}
    voc = Path(data_dir)
    tmp = Path(tempfile.mkdtemp(prefix="voc_yolo_"))
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (tmp / sub).mkdir(parents=True, exist_ok=True)

    def collect(sets):
        stems = []
        for year, split in sets:
            txt = voc / ("VOC" + str(year)) / "ImageSets" / "Main" / (split + ".txt")
            stems += [l.strip() for l in open(txt) if l.strip()]
        return stems

    import cv2
    for tag, sets in (("train", train_sets), ("val", val_sets)):
        for stem in collect(sets):
            year = sets[0][0]
            jpg = voc / ("VOC" + str(year)) / "JPEGImages" / (stem + ".jpg")
            xml = voc / ("VOC" + str(year)) / "Annotations" / (stem + ".xml")
            if not jpg.exists():
                for y, _ in sets:
                    cand = voc / ("VOC" + str(y)) / "JPEGImages" / (stem + ".jpg")
                    if cand.exists():
                        jpg, year = cand, y
                        xml = voc / ("VOC" + str(y)) / "Annotations" / (stem + ".xml")
                        break
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            h, w = img.shape[:2]
            shutil.copy2(jpg, tmp / "images" / tag / (stem + ".jpg"))
            rows = []
            if xml.exists():
                for o in ET.parse(xml).getroot().iter("object"):
                    name = o.find("name").text.strip()
                    if name not in cls2id:
                        continue
                    b = o.find("bndbox")
                    xmin = float(b.find("xmin").text); ymin = float(b.find("ymin").text)
                    xmax = float(b.find("xmax").text); ymax = float(b.find("ymax").text)
                    rows.append(f"{cls2id[name]} {(xmin+xmax)/2/w:.6f} {(ymin+ymax)/2/h:.6f} {(xmax-xmin)/w:.6f} {(ymax-ymin)/h:.6f}")
            open(tmp / "labels" / tag / (stem + ".txt"), "w").write("\n".join(rows) + ("\n" if rows else ""))
    return tmp


def _ultra_yaml_for_exp(exp):
    """PRIMARY: datasets/VOCdevkit when exp has data_dir=VOCdevkit (VOC sets).
    BACKUP: data/*.yaml YOLO-txt otherwise."""
    import tempfile
    import yaml
    data_dir = getattr(exp, "data_dir", None)
    if data_dir and "VOCdevkit" in str(data_dir):
        tmp = _voc_to_yolo_tmp(str((Path.cwd() / data_dir).resolve())
                               if not Path(data_dir).is_absolute() else data_dir,
                               getattr(exp, "train_sets", (("2007", "trainval"),)),
                               getattr(exp, "val_sets", (("2007", "test"),)))
        from yolox.data.datasets import VOC_CLASSES
        ud = {"path": str(tmp), "train": str(tmp / "images" / "train"),
              "val": str(tmp / "images" / "val"),
              "nc": len(VOC_CLASSES), "names": list(VOC_CLASSES)}
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(ud, f)
        f.close()
        print(f"[ultra-backend] PRIMARY VOCdevkit -> {tmp} nc={ud['nc']}")
        return f.name, ud
    return _abs_data_yaml(getattr(exp, "data_yaml", "data/custom.yaml"))


def train_ultra(exp, args):
    from ultralytics import YOLO
    ckpt = getattr(args, "ckpt", None) or f"yolov8{exp.version}.pt"
    data_yaml, d = _ultra_yaml_for_exp(exp)
    epochs = int(getattr(exp, "max_epoch", 50))
    imgsz = int(exp.input_size[0] if isinstance(exp.input_size, (list, tuple)) else 640)
    batch = int(getattr(args, "batch_size", None) or getattr(exp, "batch_size", 16) or 16)
    device = getattr(args, "devices", None)
    save_dir = (Path.cwd() / getattr(exp, "save_dir", "YOLOX_outputs") / exp.exp_name).resolve()
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ckpt))
    print(f"[ultra-backend] ckpt={ckpt} data={data_yaml} nc={d.get('nc')} "
          f"epochs={epochs} imgsz={imgsz} batch={batch}")
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                device=device if device is not None else "",
                project=str(save_dir.parent), name=save_dir.name,
                exist_ok=True, patience=getattr(exp, "patience", 50),
                close_mosaic=getattr(exp, "no_aug_epochs", 10),
                cache=getattr(args, "cache", False) or False,
                resume=bool(getattr(args, "resume", False)))
    return model


def val_ultra(exp, args):
    from ultralytics import YOLO
    ckpt = getattr(args, "ckpt", None) or f"yolov8{exp.version}.pt"
    data_yaml, _ = _ultra_yaml_for_exp(exp)
    model = YOLO(str(ckpt))
    return model.val(data=data_yaml,
                     imgsz=int(exp.test_size[0] if isinstance(exp.test_size, (list, tuple)) else 640),
                     batch=int(getattr(args, "batch_size", 16) or 16))
