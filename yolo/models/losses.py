"""YOLOv8 detection loss: BCE cls + CIoU box + DFL.

Mirrors Ultralytics v8Loss (TaskAlignedAssigner + DFL), simplified to run
without external deps. Consumes raw head outputs + normalized xywh labels.

Labels accepted:
  Tensor [B, M, 5] = (cls, cx, cy, w, h) normalized 0..1, padded with -1/NaN-safe
  or dict from yolo.data.dataset with 'labels'/'bboxes' (handled in yolov8.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.boxes import bbox2dist, box_iou
from .assigner import TaskAlignedAssigner


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        # pred_dist [A, reg_max] logits, target [A] distances 0..reg_max-0.01
        target = target.clamp(0, self.reg_max - 1e-3)
        tl = target.long()
        tr = (tl + 1).clamp(max=self.reg_max - 1)
        wl = (tr.float() - target)
        wr = 1.0 - wl
        loss = F.cross_entropy(pred_dist, tl, reduction="none") * wl + \
               F.cross_entropy(pred_dist, tr, reduction="none") * wr
        return loss.mean() if loss.numel() else pred_dist.sum() * 0


class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes=80, reg_max=16, topk=10, tal_alpha=0.5, tal_beta=6.0,
                 box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.assigner = TaskAlignedAssigner(topk=topk, alpha=tal_alpha, beta=tal_beta)
        self.dfl_loss = DFLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, cls_logits_list, box_dist_list, anchors, strides, gt_labels, gt_bboxes, mask_gt):
        """Compute loss from raw head outputs.

        cls_logits_list: list of [B, nc, H, W]
        box_dist_list:   list of [B, 4*reg_max, H, W]
        anchors: [N, 2], strides: [N, 1]
        gt_labels [B, M, 1], gt_bboxes [B, M, 4] xyxy pixels, mask_gt [B, M, 1]
        """
        B = cls_logits_list[0].shape[0]
        device = cls_logits_list[0].device

        # flatten predictions
        pred_scores_l, pred_dist_l = [], []
        for cls, reg in zip(cls_logits_list, box_dist_list):
            b, _, h, w = cls.shape
            pred_scores_l.append(cls.view(b, self.num_classes, -1).permute(0, 2, 1))
            pred_dist_l.append(reg.view(b, 4, self.reg_max, -1).permute(0, 3, 1, 2))
        pred_scores = torch.cat(pred_scores_l, 1)          # [B,N,C] logits
        pred_dists = torch.cat(pred_dist_l, 1)             # [B,N,4,reg_max]

        # decode boxes for assignment: softmax DFL -> ltrb cells -> * stride -> xyxy
        prob = pred_dists.softmax(-1)
        proj = torch.arange(self.reg_max, device=device, dtype=prob.dtype)
        dist_cells = (prob * proj.view(1, 1, 1, -1)).sum(-1)  # [B,N,4]
        stride = strides.to(device).view(1, -1, 1)            # [1,N,1]
        dist_px = dist_cells * stride
        ac = anchors.to(device).view(1, -1, 2)
        x1 = ac[..., 0:1] - dist_px[..., 0:1]
        y1 = ac[..., 1:2] - dist_px[..., 1:2]
        x2 = ac[..., 0:1] + dist_px[..., 2:3]
        y2 = ac[..., 1:2] + dist_px[..., 3:4]
        pred_boxes = torch.cat([x1, y1, x2, y2], -1)  # [B,N,4]

        _, target_bboxes, target_scores, fg_mask, num_fg = self.assigner(
            pred_scores.sigmoid(), pred_boxes.detach(), ac.squeeze(0),
            gt_labels, gt_bboxes, mask_gt,
        )

        # ---- cls loss (BCE over all anchors, targets = iou-weighted one-hots)
        cls_loss = self.bce(pred_scores, target_scores)

        # ---- box + dfl loss on foreground only
        if fg_mask.sum() > 0:
            iou = box_iou(pred_boxes[fg_mask], target_bboxes[fg_mask], ciou=True)
            # box_iou batched pairwise diag: inputs aligned -> take diag per element?
            # Our box_iou returns [K,K]; take diagonal for matched pairs.
            iou = iou.diag() if iou.dim() == 2 and iou.shape[0] == iou.shape[1] else iou
            box_loss = (1.0 - iou).mean()

            # DFL: target ltrb in cells = (gt xyxy -> dist px) / stride
            fg_idx = torch.where(fg_mask)
            fg_stride = stride[0, fg_idx[1], 0]  # [K]
            fg_points = ac[0, fg_idx[1]]         # [K,2]
            tgt_dist_px = torch.stack([
                fg_points[:, 0] - target_bboxes[fg_mask][:, 0],
                fg_points[:, 1] - target_bboxes[fg_mask][:, 1],
                target_bboxes[fg_mask][:, 2] - fg_points[:, 0],
                target_bboxes[fg_mask][:, 3] - fg_points[:, 1],
            ], -1).clamp(0, self.reg_max - 1e-3)
            tgt_dist_cells = tgt_dist_px / fg_stride[:, None]
            # pred logits for fg: [K,4,reg_max]
            K = fg_idx[0].shape[0]
            fg_pred = pred_dists[fg_idx[0], fg_idx[1]]  # [K,4,reg_max]
            dfl = self.dfl_loss(fg_pred.reshape(-1, self.reg_max),
                                tgt_dist_cells.reshape(-1))
        else:
            box_loss = pred_boxes.sum() * 0
            dfl = pred_boxes.sum() * 0

        total = self.box_gain * box_loss + self.cls_gain * cls_loss + self.dfl_gain * dfl
        return total, {
            "iou_loss": float((self.box_gain * box_loss).item()),
            "cls_loss": float((self.cls_gain * cls_loss).item()),
            "dfl_loss": float((self.dfl_gain * dfl).item()),
            "total_loss": float(total.item()),
            "num_fg": int(fg_mask.sum().item()),
        }
