"""Unified YOLO API (Ultralytics-style): YOLO('yolov8n.pt') etc.

  from yolo.engine import YOLO
  model = YOLO('yolov8n.pt')          # task auto-parsed: detect/seg/cls/pose/obb
  model.info()
  model.train(data='data/custom.yaml', epochs=10, imgsz=640)
  model.predict('image.jpg')
  model.val()
  model.export(format='onnx')
  model.track('video.mp4')
  model.benchmark()

Weight files: local .pth checkpoints from tools/train.py load when shapes
match; official ultralytics *.pt need conversion (shapes differ) — documented.
"""

from pathlib import Path

import torch

from ..models.tasks import create_model

TASK_SUFFIX = {"": "detect", "-seg": "segment", "-cls": "classify",
               "-pose": "pose", "-obb": "obb"}
VERSIONS = ["n", "s", "m", "l", "x"]


def parse_weights(name):
    s = str(name).lower().replace(".pt", "").replace(".pth", "")
    task = "detect"
    for suf, t in TASK_SUFFIX.items():
        if suf and s.endswith(suf):
            task = t
            s = s[: -len(suf)]
            break
    version = "s"
    for v in VERSIONS:
        if s.endswith(v):
            version = v
            break
    return task, version


class YOLO:
    def __init__(self, weights="yolov8n.pt", num_classes=80, device="auto", **task_kwargs):
        self.weights = weights
        self.task, version = parse_weights(weights)
        if "version" in task_kwargs:
            version = task_kwargs.pop("version")
        if "task" in task_kwargs:
            self.task = task_kwargs.pop("task")
        self.version = version
        self.num_classes = num_classes
        self.task_kwargs = task_kwargs
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = create_model(self.task, self.version, self.num_classes, **self.task_kwargs)
        p = Path(str(weights))
        if p.exists():
            try:
                sd = torch.load(p, map_location="cpu")
                self.model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
                print(f"loaded {p}")
            except Exception as e:
                print(f"could not load {p}: {e} (random init kept)")
        self.model.to(self.device)

    def info(self):
        n = sum(p.numel() for p in self.model.parameters())
        print(f"YOLOv8-{self.version} task={self.task} nc={self.num_classes} params={n:,} device={self.device}")
        return {"task": self.task, "version": self.version, "params": n}

    def train(self, data="data/custom.yaml", epochs=10, imgsz=640, batch=16,
              lr=0.01, save_dir="YOLOv8_outputs", **kw):
        from ..data import YOLODataset, yolo_collate
        from ..utils import ModelEMA
        import yaml
        from torch.utils.data import DataLoader
        with open(data) as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        nc = int(d.get("nc", self.num_classes))
        if nc != self.num_classes:
            print(f"rebuilding model nc {self.num_classes}->{nc}")
            self.num_classes = nc
            self.model = create_model(self.task, self.version, nc, **self.task_kwargs).to(self.device)
        exp_lr = kw.get("lr", lr)
        opt = torch.optim.SGD(self.model.parameters(), lr=exp_lr, momentum=0.937,
                              weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ema = ModelEMA(self.model) if kw.get("ema", True) else None
        amp = kw.get("amp", torch.cuda.is_available())
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        patience = kw.get("patience", 20)
        best, bad = float("inf"), 0
        out_dir = Path(save_dir) / f"yolov8{self.version}-{self.task}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _split(key_img, key_lb, aug):
            di = d.get(key_img, f"images/{key_img.split('_')[0]}")
            dl = d.get(key_lb, f"labels/{key_lb.split('_')[0]}")
            ds = YOLODataset(root / di, root / dl, imgsz, augment=aug)
            return DataLoader(ds, batch_size=batch, shuffle=aug, collate_fn=yolo_collate)

        try:
            train_loader = _split("train_images", "train_labels", True)
        except Exception:
            train_loader = _split("train", "train", True)
        for epoch in range(epochs):
            self.model.train()
            tot, n = 0.0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=amp):
                    out = self.model(imgs, labels)
                    loss = out["total_loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                scaler.step(opt)
                scaler.update()
                if ema:
                    ema.update(self.model)
                tot += float(loss.item())
                n += 1
            sched.step()
            avg = tot / max(n, 1)
            print(f"epoch {epoch+1}/{epochs} loss={avg:.4f}")
            if avg < best:
                best, bad = avg, 0
                torch.save({"model_state_dict": (ema.ema.state_dict() if ema else self.model.state_dict())},
                           out_dir / "best.pth")
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stop at epoch {epoch+1}")
                    break
        torch.save({"model_state_dict": self.model.state_dict()}, out_dir / "last.pth")
        return {"best_loss": best, "dir": str(out_dir)}

    @torch.no_grad()
    def predict(self, source, conf=0.25, iou=0.7, imgsz=640):
        import cv2
        import numpy as np
        from ..data import letterbox
        self.model.eval()
        img0 = cv2.imread(str(source)) if isinstance(source, (str, Path)) else None
        if img0 is None:
            img0 = np.full((imgsz, imgsz, 3), 128, np.uint8)
        rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        lb, _, _ = letterbox(rgb, imgsz)
        t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        if self.task == "detect":
            dets = self.model.predict(t, conf_threshold=conf, iou_threshold=iou)[0]
            return dets.cpu()
        out = self.model(t)
        return out

    def val(self, data="data/custom.yaml", conf=0.25, iou=0.7, imgsz=640):
        from ..data import YOLODataset, yolo_collate
        import yaml
        from torch.utils.data import DataLoader
        with open(data) as f:
            d = yaml.safe_load(f)
        root = Path(d.get("path", "."))
        ds = YOLODataset(root / d.get("val_images", "images/val"),
                         root / d.get("val_labels", "labels/val"), imgsz, augment=False)
        loader = DataLoader(ds, batch_size=8, collate_fn=yolo_collate)
        self.model.eval()
        n_det, n_img = 0, 0
        with torch.no_grad():
            for imgs, _ in loader:
                dets = self.model.predict(imgs.to(self.device), conf_threshold=conf, iou_threshold=iou)
                for x in dets:
                    n_det += len(x)
                n_img += len(dets)
        print(f"val images={n_img} dets={n_det}")
        return {"images": n_img, "detections": n_det}

    def export(self, format="onnx", imgsz=640, path=None):
        path = path or f"yolov8{self.version}-{self.task}.{format}"
        if format == "onnx":
            self.model.eval()
            dummy = torch.randn(1, 3, imgsz, imgsz)
            torch.onnx.export(self.model, dummy, path, opset_version=18,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        elif format == "torchscript":
            self.model.eval()
            ex = torch.jit.trace(self.model, torch.randn(1, 3, imgsz, imgsz))
            torch.jit.save(ex, path)
        else:
            raise ValueError("formats: onnx, torchscript (openvino/tensorrt via onnx)")
        print(f"exported {path}")
        return path

    def track(self, source=0, conf=0.25, iou=0.7, imgsz=640):
        import cv2
        import numpy as np
        from ..data import letterbox
        from .tracker import IoUTracker
        tr = IoUTracker()
        cap = cv2.VideoCapture(source)
        self.model.eval()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lb, _, _ = letterbox(rgb, imgsz)
            t = torch.from_numpy(lb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                dets = self.model.predict(t, conf_threshold=conf, iou_threshold=iou)[0].cpu().numpy()
            print(tr.update(dets))
            break
        cap.release()
        return tr

    def benchmark(self, imgsz=640, runs=50):
        from .benchmark import benchmark_torch
        r = benchmark_torch(self.model, imgsz, str(self.device), runs)
        print(r)
        return r
