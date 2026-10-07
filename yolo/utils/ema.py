"""EMA + checkpoint helpers (YOLOX-style)."""

import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema = self._clone(model)

    @staticmethod
    def _clone(model):
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.ema.parameters(), model.parameters()):
            e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()
