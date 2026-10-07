#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Inference-only replica of the DFL-modified YOLOX head found in customized
# YOLOX repos (Megvii YOLOX, Apache-2.0, yolox/models/yolo_head.py + DFL ext).
# Module names/shapes are identical so legacy checkpoints load directly.
# Eval math replicates the original forward/decode path exactly.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        reg_max=16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.reg_max = reg_max

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        reg_channels = 4 * reg_max if reg_max > 0 else 4

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(int(256 * width), int(256 * width), ksize=3, stride=1, act=act),
                        Conv(int(256 * width), int(256 * width), ksize=3, stride=1, act=act),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(int(256 * width), int(256 * width), ksize=3, stride=1, act=act),
                        Conv(int(256 * width), int(256 * width), ksize=3, stride=1, act=act),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), reg_channels, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(int(256 * width), 1, kernel_size=1, stride=1, padding=0)
            )

        if reg_max > 0:
            self.dfl_proj = nn.ModuleList()
            for _ in range(len(in_channels)):
                self.dfl_proj.append(nn.Conv2d(reg_max, 1, kernel_size=1, bias=False))

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def _decode_dfl(self, reg_output, k, batch_size, hsize, wsize):
        reg_output = reg_output.view(batch_size, 4, self.reg_max, hsize, wsize)
        reg_output = F.softmax(reg_output, dim=2)
        proj_weight = self.dfl_proj[k].weight.view(1, 1, self.reg_max, 1, 1)
        reg_output = (reg_output * proj_weight).sum(dim=2)
        return reg_output.view(batch_size, 4, hsize, wsize)

    def forward(self, xin):
        """Eval only. Returns [B, N, 4+1+nc] (cx,cy,w,h,obj,cls), pixels."""
        outputs = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.reg_max > 0:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output_decoded = self._decode_dfl(reg_output, k, batch_size, hsize, wsize)
                output = torch.cat(
                    [reg_output_decoded, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        self.hw = [x.shape[-2:] for x in outputs]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(outputs, xin[0].device, xin[0].dtype)
        return outputs

    def decode_outputs(self, outputs, device, dtype):
        grids, strides = [], []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid(
                torch.arange(hsize, device=device), torch.arange(wsize, device=device),
                indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), stride, device=device))
        grids = torch.cat(grids, dim=1).to(dtype)
        strides = torch.cat(strides, dim=1).to(dtype)

        if self.reg_max > 0:
            outputs = torch.cat([
                outputs[..., 0:2] + grids * strides,
                outputs[..., 2:4],
                outputs[..., 4:],
            ], dim=-1)
        else:
            outputs = torch.cat([
                (outputs[..., 0:2] + grids) * strides,
                torch.exp(outputs[..., 2:4]) * strides,
                outputs[..., 4:],
            ], dim=-1)
        return outputs
