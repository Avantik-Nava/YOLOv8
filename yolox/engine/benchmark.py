"""Benchmark: latency / FPS / params / FLOPs across formats (detect task)."""

import time

import torch


def benchmark_torch(model, imgsz=640, device="cpu", runs=50, warmup=10):
    model = model.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(runs):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / runs
    n_params = sum(p.numel() for p in model.parameters())
    return {"latency_ms": dt * 1000, "fps": 1 / max(dt, 1e-9),
            "params": n_params, "device": device, "imgsz": imgsz}
