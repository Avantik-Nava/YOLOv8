"""tools/val.py — full validation: NMS + COCO mAP50 / mAP50-95 per task."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_exp(path):
    spec = importlib.util.spec_from_file_location("exp_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp", default="exps/yolov8_s.py")
    p.add_argument("--task", default=None)
    p.add_argument("-c", "--ckpt", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    args = p.parse_args()

    from yolo.utils import coco_map
    exp = load_exp(args.exp)
    if args.task:
        exp.task = args.task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = exp.get_model().to(device).eval()
    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd.get("model_state_dict", sd), strict=False)

    import yaml
    from torch.utils.data import DataLoader
    from exps.default import YOLOv8Exp
    if type(exp).get_data is not YOLOv8Exp.get_data:
        _, loader, _ = exp.get_data()
    else:
        from yolo.data import YOLODataset, yolo_collate
        with open(exp.data_yaml) as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        ds = YOLODataset(root / d.get("val_images", "images/val"),
                         root / d.get("val_labels", "labels/val"), exp.test_size[0])
        loader = DataLoader(ds, batch_size=8, collate_fn=yolo_collate)
    all_dets, all_gts = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            dets = model.predict(imgs.to(device), conf_threshold=args.conf, iou_threshold=args.iou)
            B = imgs.shape[0]
            for b in range(B):
                all_dets.append(np.asarray(dets[b].cpu()))
                gt = []
                H = W = exp.test_size[0]
                for m in range(labels.shape[1]):
                    c = float(labels[b, m, 0])
                    if c < 0:
                        continue
                    _, cx, cy, w, h = labels[b, m].tolist()
                    gt.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                               (cx + w / 2) * W, (cy + h / 2) * H])
                all_gts.append(np.array(gt, dtype=np.float32).reshape(-1, 5))
    m = coco_map(all_dets, all_gts)
    print(f"task={exp.task} mAP50={m['map50']:.4f} mAP50-95={m['map5095']:.4f} "
          f"P={m['precision']:.4f} R={m['recall']:.4f} (pycocotools={m['pycocotools']})")


if __name__ == "__main__":
    main()
