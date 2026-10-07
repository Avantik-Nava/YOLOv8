"""YOLOv8 main model (anchor-free, DFL, no objectness branch).

API mirrors YOLOX/YOLOv26 for familiarity:
  model(x)              -> eval: Tensor [B, N, 4+nc] (boxes xyxy px + cls scores)
  model(x, labels)      -> train: dict(total_loss, iou_loss, cls_loss, dfl_loss)
  model.predict(x, ...) -> NMS detections list [M,6] per image

labels Tensor [B, M, 5] = (cls, cx, cy, w, h) normalized 0..1 (pad cls with -1).
"""

import torch
import torch.nn as nn

from .backbone import CSPDarknetC2f
from .head import YOLOv8Head
from .losses import YOLOv8Loss
from .neck import YOLOv8Neck


class YOLOv8(nn.Module):
    def __init__(self, version="s", num_classes=80, reg_max=16, strides=(8, 16, 32)):
        super().__init__()
        self.version = version
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = list(strides)

        self.backbone = CSPDarknetC2f(version)
        self.neck = YOLOv8Neck(self.backbone.out_channels, version)
        self.head = YOLOv8Head(num_classes, self.neck.out_channels, reg_max, strides)
        self.criterion = YOLOv8Loss(num_classes, reg_max)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # restore head prior biases after zeroing
        self.head._init_bias()

    # ---------- anchors ----------
    def _make_anchors(self, feats, device):
        points, stride_t = [], []
        for f, s in zip(feats, self.strides):
            h, w = f.shape[-2:]
            yv, xv = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            pts = torch.stack((xv, yv), -1).float().view(-1, 2)  # cells
            pts = (pts + 0.5) * s  # pixels
            points.append(pts)
            stride_t.append(torch.full((pts.shape[0], 1), float(s), device=device))
        return torch.cat(points, 0), torch.cat(stride_t, 0)  # [N,2], [N,1]

    def _prep_targets(self, labels, img_size, device):
        """labels [B,M,5] (cls,cxcywh norm) -> gt_labels, gt_bboxes xyxy px, mask."""
        B, M, _ = labels.shape
        gt_labels = torch.zeros((B, M, 1), dtype=torch.long, device=device)
        gt_bboxes = torch.zeros((B, M, 4), device=device)
        mask = torch.zeros((B, M, 1), dtype=torch.bool, device=device)
        H, W = img_size, img_size
        for b in range(B):
            for m in range(M):
                c = labels[b, m, 0].item()
                if c < 0 or torch.isnan(labels[b, m, 1:]).any():
                    continue
                cx, cy, w, h = labels[b, m, 1:].tolist()
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
                gt_labels[b, m, 0] = int(c)
                gt_bboxes[b, m] = torch.tensor([x1, y1, x2, y2], device=device)
                mask[b, m, 0] = True
        return gt_labels, gt_bboxes, mask

    # ---------- forward ----------
    def forward(self, x, labels=None):
        feats = self.neck(list(self.backbone(x)))
        if labels is None:
            return self.head.predict(feats)  # [B,N,4+nc]
        # train mode
        if isinstance(labels, dict):  # compat with YOLOv26-style dataset dict
            # dict: {'labels': [B,5] or [B,M], 'bboxes': [B,M,4] cxcywh norm} -> pack
            raise ValueError("Pass labels as Tensor [B,M,5] (cls,cx,cy,w,h). See tools/train.py.")
        img_size = x.shape[-1]
        gt_labels, gt_bboxes, mask = self._prep_targets(labels, img_size, x.device)
        cls_l, reg_l = self.head(feats)
        anchors, stride_t = self._make_anchors(cls_l, x.device)
        total, items = self.criterion(cls_l, reg_l, anchors, stride_t,
                                      gt_labels, gt_bboxes, mask)
        items["total_loss_tensor"] = total
        # keep YOLOX-style keys: total_loss as tensor too
        # keep tensor under 'total_loss' (YOLOX-style), float copy under 'total_loss_value'
        out = {k: v for k, v in items.items() if k not in ("total_loss", "total_loss_tensor")}
        out["total_loss"] = total
        out["total_loss_value"] = float(total.item())
        out["total_loss_tensor"] = total
        out.update({k: items[k] for k in ("iou_loss", "cls_loss", "dfl_loss", "num_fg") if k in items})
        return out

    @torch.no_grad()
    def predict(self, x, conf_threshold=0.25, iou_threshold=0.7, max_det=300):
        from ..utils.boxes import non_max_suppression
        self.eval()
        pred = self.forward(x)  # [B,N,4+nc]
        return non_max_suppression(pred, conf_threshold, iou_threshold, max_det)

    def load_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {path}")

    def export_onnx(self, path="yolov8.onnx", img_size=640):
        self.eval()
        dummy = torch.randn(1, 3, img_size, img_size)
        torch.onnx.export(self, dummy, path, opset_version=11,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        print(f"ONNX exported: {path}")
