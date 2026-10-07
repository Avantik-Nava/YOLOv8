"""tools/train.py — full training: AMP + EMA + warmup + early-stop + tasks.

Usage:
  python tools/train.py -e exps/yolov8_s.py [--task segment] [--epochs 100]
  python tools/train.py --config config.yaml
  torchrun --nproc_per_node=2 tools/train.py -e exps/yolov8_s.py  (DDP stub)
Tasks: detect | segment | classify | pose | obb  (via exp.task or --task)
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_exp(path):
    spec = importlib.util.spec_from_file_location("exp_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp", default="exps/yolov8_s.py")
    p.add_argument("--config", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--patience", type=int, default=None)
    return p.parse_args()


def build_loaders(exp, task):
    # YOLOX-style: if exp overrides get_data (e.g. exps/yolov8_voc_s.py),
    # use it directly so user only edits the exp file.
    from exps.default import YOLOv8Exp
    if type(exp).get_data is not YOLOv8Exp.get_data:
        return exp.get_data()
    from torch.utils.data import DataLoader
    with open(exp.data_yaml) as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", "."))
    imgsz = exp.input_size[0]
    if task == "classify":
        from yolo.data import ClsDataset
        tr = ClsDataset(root, imgsz)
        va = ClsDataset(root, imgsz)
        return (DataLoader(tr, batch_size=exp.batch_size, shuffle=True),
                DataLoader(va, batch_size=exp.batch_size), d)
    if task == "segment":
        from yolo.data import SegDataset, seg_collate
        def _mk(k_img, k_lb, aug):
            return SegDataset(root / d.get(k_img, f"images/{k_img}"),
                              root / d.get(k_lb, f"labels/{k_lb}"), imgsz, augment=aug)
        tr = _mk("train_images", "train_labels", True)
        va = _mk("val_images", "val_labels", False)
        return (DataLoader(tr, batch_size=exp.batch_size, shuffle=True, collate_fn=seg_collate),
                DataLoader(va, batch_size=exp.batch_size, collate_fn=seg_collate), d)
    if task == "pose":
        from yolo.data import PoseDataset, pose_collate
        tr = PoseDataset(root / d.get("train_images", "images/train"),
                         root / d.get("train_labels", "labels/train"), imgsz, augment=True)
        va = PoseDataset(root / d.get("val_images", "images/val"),
                         root / d.get("val_labels", "labels/val"), imgsz, augment=False)
        return (DataLoader(tr, batch_size=exp.batch_size, shuffle=True, collate_fn=pose_collate),
                DataLoader(va, batch_size=exp.batch_size, collate_fn=pose_collate), d)
    from yolo.data import YOLODataset, yolo_collate  # detect + obb share xywh loader
    tr = YOLODataset(root / d.get("train_images", "images/train"),
                     root / d.get("train_labels", "labels/train"), imgsz, augment=True)
    va = YOLODataset(root / d.get("val_images", "images/val"),
                     root / d.get("val_labels", "labels/val"), imgsz, augment=False)
    return (DataLoader(tr, batch_size=exp.batch_size, shuffle=True, collate_fn=yolo_collate),
            DataLoader(va, batch_size=exp.batch_size, collate_fn=yolo_collate), d)


def main():
    args = get_args()
    exp = load_exp(args.exp)
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            key = {"model_version": "version", "imgsz": None}.get(k, k)
            if key and hasattr(exp, key):
                setattr(exp, key, v)
        if "imgsz" in cfg:
            exp.input_size = (cfg["imgsz"], cfg["imgsz"])
    if args.task:
        exp.task = args.task
    for k, v in [("data_yaml", args.data), ("num_classes", args.num_classes),
                 ("epochs", args.epochs), ("batch_size", args.batch_size),
                 ("save_dir", args.save_dir), ("patience", args.patience)]:
        if v is not None:
            setattr(exp, k, v)
    task = exp.task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda" and getattr(exp, "amp", True)
    print(f"task={task} ver=yolov8-{exp.version} nc={exp.num_classes} "
          f"epochs={exp.epochs} amp={use_amp} device={device}")

    model = exp.get_model().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    train_loader, val_loader, _ = build_loaders(exp, task)

    opt = torch.optim.SGD(model.parameters(), lr=exp.lr, momentum=exp.momentum,
                          weight_decay=exp.weight_decay, nesterov=True)
    # warmup + cosine
    warmup = getattr(exp, "warmup_epochs", 3)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(exp.epochs - warmup, 1), eta_min=exp.lr * 0.0001)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    from yolo.utils import ModelEMA
    ema = ModelEMA(model) if getattr(exp, "ema", True) else None

    save_dir = Path(exp.save_dir) / f"{exp.exp_name}-{task}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best, bad = float("inf"), 0
    patience = getattr(exp, "patience", 50)

    for epoch in range(exp.epochs):
        model.train()
        if epoch < warmup:  # linear warmup
            for g in opt.param_groups:
                g["lr"] = exp.lr * (0.1 + 0.9 * (epoch + 1) / warmup)
        if epoch >= exp.epochs - exp.close_mosaic and hasattr(train_loader.dataset, "augment"):
            train_loader.dataset.augment = False
        tot, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            if task == "segment":
                imgs, labels, masks = [b.to(device) for b in batch]
                extra = {"masks": masks}
            elif task == "pose":
                imgs, labels, kpts = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                extra = {"kpts": kpts}
            elif task == "classify":
                imgs, labels = batch[0].to(device), batch[1].to(device)
                extra = {}
            else:
                imgs, labels = batch[0].to(device), batch[1].to(device)
                extra = {}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(imgs, labels, **extra) if extra else (
                    model(imgs, labels) if task != "classify" else model(imgs, labels))
                loss = out["total_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)
            tot += float(loss.item())
            n += 1
            if n % exp.print_interval == 0:
                keys = " ".join(f"{k}={v:.3f}" for k, v in out.items()
                                if k.endswith("_loss") and isinstance(v, float))
                print(f"ep{epoch+1} b{n}/{len(train_loader)} loss={loss.item():.4f} {keys}")
        if epoch >= warmup:
            sched_cos.step()
        avg = tot / max(n, 1)
        print(f"[ep{epoch+1}] loss={avg:.4f} {time.time()-t0:.1f}s lr={opt.param_groups[0]['lr']:.6f}")
        if avg < best:
            best, bad = avg, 0
            sd = ema.ema.state_dict() if ema else model.state_dict()
            torch.save({"epoch": epoch, "model_state_dict": sd}, save_dir / "best.pth")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stop @ep{epoch+1}")
                break
        if (epoch + 1) % exp.save_interval == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       save_dir / f"ckpt_{epoch+1}.pth")
    torch.save({"model_state_dict": model.state_dict()}, save_dir / "last.pth")
    print(f"done {save_dir} best={best:.4f}")


if __name__ == "__main__":
    main()
