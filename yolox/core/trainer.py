#!/usr/bin/env python3
# YOLOX-style Trainer (yolox/core/trainer.py API): warmup scheduler per-iter,
# AMP, EMA, no_aug closing, eval_interval, best/last/history ckpts, resume.

import time
from pathlib import Path

import torch

from yolox.utils import ModelEMA


class Trainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = getattr(args, "fp16", False) and self.device.type == "cuda"
        self.max_epoch = exp.max_epoch
        self.no_aug_epochs = exp.no_aug_epochs
        out = getattr(exp, "output_dir", None) or getattr(exp, "save_dir", "YOLOX_outputs")
        self.file_name = str(getattr(args, "experiment_name", None) or exp.exp_name)
        self.save_dir = Path(out) / self.file_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_history_ckpt = getattr(exp, "save_history_ckpt", True)
        self.eval_interval = getattr(exp, "eval_interval", 1)
        self.print_interval = getattr(exp, "print_interval", 10)
        self.patience = getattr(exp, "patience", 50)
        self.start_epoch = getattr(args, "start_epoch", 0) or 0

    def train(self):
        exp, args = self.exp, self.args
        model = exp.get_model().to(self.device)
        exp.model = model
        optimizer = exp.get_optimizer(args.batch_size)
        train_loader = exp.get_data_loader(args.batch_size, False)
        iters_per_epoch = max(len(train_loader), 1)
        scheduler = exp.get_lr_scheduler(
            exp.basic_lr_per_img * args.batch_size, iters_per_epoch)
        ema = ModelEMA(model) if exp.ema else None
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

        if getattr(args, "ckpt", None):
            ckpt = torch.load(args.ckpt, map_location=self.device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            print(f"resumed {args.ckpt}")

        best_map, bad, num_iter = 0.0, 0, 0
        no_aug_closed = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch >= self.max_epoch - self.no_aug_epochs and not no_aug_closed:
                train_loader = exp.get_data_loader(args.batch_size, False, no_aug=True)
                iters_per_epoch = max(len(train_loader), 1)
                no_aug_closed = True
                print("closed mosaic/mixup for last epochs")
            model.train()
            tot, n, t0 = 0.0, 0, time.time()
            for imgs, labels in train_loader:
                num_iter += 1
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                scheduler.update_lr(num_iter, optimizer)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.fp16):
                    out = model(imgs, labels)
                    loss = out["total_loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                if ema:
                    ema.update(model)
                tot += float(loss.item())
                n += 1
                if n % self.print_interval == 0:
                    keys = " ".join(f"{k}={v:.3f}" for k, v in out.items()
                                    if k.endswith("_loss") and isinstance(v, float))
                    print(f"ep{epoch+1}/{self.max_epoch} it{n}/{len(train_loader)} "
                          f"loss={loss.item():.4f} {keys} lr={optimizer.param_groups[0]['lr']:.6f}")
            print(f"[ep{epoch+1}] avg={tot/max(n,1):.4f} {time.time()-t0:.1f}s")

            if self.save_history_ckpt and (epoch + 1) % getattr(exp, "save_interval", 10) == 0:
                self._save(model, optimizer, epoch, f"ckpt_epoch_{epoch+1}.pth", ema=None)
            self._save(model, optimizer, epoch, "last.pth", ema=None)
            if ema:
                torch.save({"model_state_dict": ema.ema.state_dict()},
                           self.save_dir / "best_ema.pth")

            if (epoch + 1) % self.eval_interval == 0 or (epoch + 1) == self.max_epoch:
                evaluator = exp.get_evaluator(args.batch_size, False)
                evalmodel = ema.ema if ema else model
                map5095, map50 = exp.eval(evalmodel, evaluator, False)
                if map5095 > best_map:
                    best_map, bad = map5095, 0
                    self._save(model, optimizer, epoch, "best.pth", ema=ema)
                else:
                    bad += 1
                    if bad >= self.patience:
                        print(f"early stop @ep{epoch+1} best_mAP={best_map:.4f}")
                        break
        print(f"done {self.save_dir} best_mAP50-95={best_map:.4f}")

    def _save(self, model, optimizer, epoch, name, ema=None):
        sd = ema.ema.state_dict() if ema is not None else model.state_dict()
        torch.save({"epoch": epoch, "model_state_dict": sd,
                    "optimizer": optimizer.state_dict()}, self.save_dir / name)
        print(f"saved {self.save_dir / name}")
