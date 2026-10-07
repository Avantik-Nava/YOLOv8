"""Task-Aligned Assigner (simplified TAL, YOLOv8 style).

Ultralytics YOLOv8 uses TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0).
This is a compact, dependency-free version with identical I/O contract
so tools/train.py and test_model.py behave like the official pipeline.
"""

import torch

from ..utils.boxes import box_iou


class TaskAlignedAssigner:
    def __init__(self, topk=10, alpha=0.5, beta=6.0, eps=1e-7):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def __call__(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """Assign GT to anchors.

        Args:
            pred_scores: [B, N, C] sigmoid scores
            pred_boxes:  [B, N, 4] xyxy (pixels)
            anchor_points: [N, 2] centers (pixels)
            gt_labels: [B, M, 1] long (may contain -1 for padding)
            gt_bboxes: [B, M, 4] xyxy (pixels)
            mask_gt:   [B, M, 1] bool valid
        Returns:
            target_labels [B, N] long, target_bboxes [B, N, 4],
            target_scores [B, N, C], fg_mask [B, N] bool, num_fg int
        """
        B, N, C = pred_scores.shape
        M = gt_bboxes.shape[1]
        device = pred_scores.device

        target_labels = torch.zeros((B, N), dtype=torch.long, device=device)
        target_bboxes = torch.zeros((B, N, 4), device=device)
        target_scores = torch.zeros((B, N, C), device=device)
        fg_mask = torch.zeros((B, N), dtype=torch.bool, device=device)

        if M == 0:
            return target_labels, target_bboxes, target_scores, fg_mask, 0

        num_fg_total = 0
        for b in range(B):
            valid = mask_gt[b, :, 0]
            if valid.sum() == 0:
                continue
            g_boxes = gt_bboxes[b][valid]          # [G,4]
            g_cls = gt_labels[b][valid].squeeze(-1).long().clamp(0, C - 1)  # [G]
            G = g_boxes.shape[0]

            ious = box_iou(pred_boxes[b].detach(), g_boxes)  # [N,G]
            # class scores for each gt class: [N,G]
            cls_scores = pred_scores[b].detach()  # [N,C]
            pos = cls_scores[:, g_cls]  # [N,G]
            align = (pos.clamp(min=1e-7) ** self.alpha) * (ious.clamp(min=1e-7) ** self.beta)

            k = min(self.topk, N)
            _, topk_idx = align.topk(k, dim=0)  # [k,G]
            cand = torch.zeros_like(align, dtype=torch.bool)
            cand.scatter_(0, topk_idx, True)

            # resolve multi-gt overlap: keep gt with max IoU per anchor
            is_fg = cand.any(-1)  # [N]
            if is_fg.sum() == 0:
                continue
            best_gt = ious[is_fg].argmax(-1)  # [n_fg]
            fg_idx = torch.where(is_fg)[0]

            target_labels[b, fg_idx] = g_cls[best_gt]
            target_bboxes[b, fg_idx] = g_boxes[best_gt]
            # cls target = one-hot scaled by IoU (like Ultralytics: iou-weighted)
            iou_fg = ious[is_fg, best_gt].clamp(0, 1)
            target_scores[b, fg_idx, g_cls[best_gt]] = iou_fg
            fg_mask[b, fg_idx] = True
            num_fg_total += int(is_fg.sum().item())

        return target_labels, target_bboxes, target_scores, fg_mask, num_fg_total
