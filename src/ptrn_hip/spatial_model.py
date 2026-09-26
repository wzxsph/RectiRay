"""Versioned four-arm corner-localization comparison on the G0 DenseNet201.

The model accepts ImageNet-normalized NHWC RGB tensors and returns four ordered
TL/TR/BR/BL corners in input pixel-centre coordinates. It has no ratio or action
head. ``RawPixelModel`` supplies a traceable RGB 0..255 deployment entrypoint.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import keras
from .ratio_model import corner_tensor
from .unified_model import build_unified


ARMS = frozenset({"gap_h8", "gap_quad", "spatial_quad", "spatial_heatmap"})
SIZES = frozenset({224, 384})
TAPS = (
    "conv2_block6_concat",
    "conv3_block12_concat",
    "conv4_block48_concat",
    "conv5_block32_concat",
)
CONTRACT = "ptrn-spatial-corners-v2"


def contract(size: int, arm: str | None = None) -> dict:
    """Shared four-arm output contract; an optional arm adds readout detail."""
    if (arm is not None and arm not in ARMS) or size not in SIZES:
        raise ValueError("Unsupported spatial arm or input size")
    return {
        "contract": CONTRACT,
        "arm": arm,
        "input_shape": [None, size, size, 3],
        "input_dtype": "float32",
        "input_range": "RGB 0..255 through RawPixelModel",
        "input_preprocessing": "RGB/255, then ImageNet mean/std; NHWC",
        "output_shape": [None, 4, 2],
        "output_dtype": "float32",
        "output_fields": "TL,TR,BR,BL input pixel-centre x,y",
        "coordinate_range": f"0..{size - 1} represents the input boundary; direct arms may predict outside",
        "spatial_readout_grid": 56 if arm == "spatial_quad" else None,
        "ratio": None,
        "action": None,
        "initialization": "strict approved B1.1/G0 encoder; gap_h8 also copies G0 H8",
    }


def _initial_quad(value: np.ndarray | None, size: int) -> torch.Tensor:
    if value is None:
        fraction = np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]], np.float32)
        value = fraction * (size - 1)
    result = np.asarray(value, np.float32)
    if result.shape != (4, 2) or not np.isfinite(result).all():
        raise ValueError("initial_quad must be finite [4,2] input pixel-centre coordinates")
    if (result < 0).any() or (result > size - 1).any():
        raise ValueError("initial_quad must lie inside the input pixel-centre bounds")
    return torch.from_numpy(result.copy())


class SpatialModel(nn.Module):
    """Single G0 encoder and one of four mutually exclusive corner readouts."""

    def __init__(self, backbone, g0_kernel, g0_bias, arm: str, size: int,
                 stage: str, initial_quad: np.ndarray | None):
        super().__init__()
        if arm not in ARMS or size not in SIZES:
            raise ValueError("Unsupported spatial arm or input size")
        self.arm, self.size = arm, size
        if arm.startswith("spatial_"):
            outputs = [backbone.get_layer(name).output for name in TAPS]
            self.encoder = keras.Model(backbone.input, outputs, name="g0_spatial_features")
            channels = [int(output.shape[-1]) for output in outputs]
            self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in channels])
            self.smooth = nn.ModuleList([nn.Conv2d(64, 64, 3, padding=1) for _ in channels])
            self.spatial_logits = nn.Conv2d(64, 4, 1)
            if arm == "spatial_quad":
                # Fixed readout grid permits the same architecture at 224/384.
                self.direct = nn.Linear(4 * 56 * 56, 8)
                # Two consecutive zero layers would block every decoder gradient.
                nn.init.normal_(self.direct.weight, mean=0., std=1e-5)
                with torch.no_grad():
                    self.direct.bias.copy_(_initial_quad(initial_quad, size).reshape(-1) / (size - 1))
            else:
                self.direct = None
                nn.init.zeros_(self.spatial_logits.weight)
                nn.init.zeros_(self.spatial_logits.bias)
            self.gap_head = None
        else:
            self.encoder = backbone
            self.gap_head = nn.Linear(1920, 8)
            if arm == "gap_h8":
                kernel, bias = np.asarray(g0_kernel), np.asarray(g0_bias)
                if kernel.shape != (1920, 9) or bias.shape != (9,):
                    raise ValueError("G0-derived Dense9 has the wrong shape")
                with torch.no_grad():
                    self.gap_head.weight.copy_(torch.from_numpy(kernel[:, :8].T.copy()))
                    self.gap_head.bias.copy_(torch.from_numpy(bias[:8].copy()))
            else:
                nn.init.zeros_(self.gap_head.weight)
                with torch.no_grad():
                    self.gap_head.bias.copy_(_initial_quad(initial_quad, size).reshape(-1) / (size - 1))
            self.lateral = self.smooth = self.spatial_logits = self.direct = None
        self.set_stage(stage)

    def set_stage(self, stage: str) -> None:
        if stage not in {"head", "last_block"}:
            raise ValueError("Stage must be head or last_block")
        self.stage = stage
        self.encoder.trainable = stage == "last_block"
        if stage == "last_block":
            for layer in self.encoder.layers:
                layer.trainable = (layer.name.startswith("conv5_")
                                   and not isinstance(layer, keras.layers.BatchNormalization))
        # Encoder calls use training=False even when conv5 weights are trainable.

    def extract_features(self, normalized_nhwc: torch.Tensor):
        if normalized_nhwc.ndim != 4 or tuple(normalized_nhwc.shape[1:]) != (self.size, self.size, 3):
            raise ValueError("Expected normalized NHWC input at the selected size")
        if self.stage == "head":
            with torch.no_grad():
                value = self.encoder(normalized_nhwc, training=False)
        else:
            value = self.encoder(normalized_nhwc, training=False)
        if self.arm.startswith("spatial_"):
            return tuple(t.permute(0, 3, 1, 2).contiguous() for t in value)
        return value

    def _spatial_logits(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError("Spatial arm requires four NCHW feature maps")
        mapped = [conv(t.float()) for conv, t in zip(self.lateral, features, strict=True)]
        x = F.relu(self.smooth[3](mapped[3]))
        for index in (2, 1, 0):
            x = F.relu(self.smooth[index](mapped[index] + F.interpolate(
                x, size=mapped[index].shape[-2:], mode="bilinear", align_corners=True)))
        return self.spatial_logits(x)

    def forward_features(self, features) -> torch.Tensor:
        if self.arm.startswith("gap_"):
            if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 1920:
                raise ValueError("GAP arm requires [N,1920] features")
            values = self.gap_head(features.float())
            if self.arm == "gap_h8":
                return corner_tensor(values, self.size).float()
            return values.reshape(-1, 4, 2) * (self.size - 1)
        logits = self._spatial_logits(features).float()
        if self.arm == "spatial_quad":
            pooled = F.adaptive_avg_pool2d(logits, (56, 56))
            return self.direct(pooled.flatten(1)).reshape(-1, 4, 2) * (self.size - 1)
        n, corners, height, width = logits.shape
        weights = torch.softmax(logits.reshape(n, corners, height * width), dim=-1)
        # The first/last grid centres map exactly to input pixel centres 0/size-1.
        y, x = torch.meshgrid(
            torch.linspace(0, self.size - 1, height, device=logits.device, dtype=logits.dtype),
            torch.linspace(0, self.size - 1, width, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        grid = torch.stack((x.flatten(), y.flatten()), dim=-1)
        return torch.matmul(weights, grid).float()

    def forward(self, normalized_nhwc: torch.Tensor) -> torch.Tensor:
        return self.forward_features(self.extract_features(normalized_nhwc))

    def raw_pixels(self) -> "RawPixelModel":
        return RawPixelModel(self)


class RawPixelModel(nn.Module):
    """Torch-only preprocessing wrapper suitable for CPU tracing/export."""

    def __init__(self, model: SpatialModel):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([.485, .456, .406], dtype=torch.float32))
        self.register_buffer("std", torch.tensor([.229, .224, .225], dtype=torch.float32))

    def forward(self, pixels_nhwc: torch.Tensor) -> torch.Tensor:
        normalized = (pixels_nhwc.float() / 255.0 - self.mean) / self.std
        return self.model(normalized)


def build_spatial(weights, g0_checkpoint, arm: str, size: int = 224,
                  stage: str = "head", initial_quad: np.ndarray | None = None) -> SpatialModel:
    """Validate the approved checkpoint, then share its encoder weights."""
    if arm not in ARMS or size not in SIZES:
        raise ValueError("Unsupported spatial arm or input size")
    legacy, backbone, head = build_unified(weights, g0_checkpoint, size)
    kernel, bias = head.get_weights()
    result = SpatialModel(backbone, kernel, bias, arm, size, stage, initial_quad)
    del legacy
    return result


def parameter_groups(model: SpatialModel, stage: str):
    model.set_stage(stage)
    heads = []
    for module in (model.gap_head, model.lateral, model.smooth, model.spatial_logits, model.direct):
        if module is not None:
            heads.extend(p for p in module.parameters() if p.requires_grad)
    trunk = [value.value for value in model.encoder.trainable_weights]
    if not heads or (stage == "last_block" and not trunk):
        raise ValueError("No trainable head or conv5 parameters")
    if len({id(p) for p in heads + trunk}) != len(heads) + len(trunk):
        raise ValueError("Duplicate optimizer parameters")
    groups = [{"params": heads, "lr": 1e-4, "name": "corners_head"}]
    if trunk:
        groups.append({"params": trunk, "lr": 1e-5, "name": "conv5"})
    return groups


def coordinate_loss(predicted: torch.Tensor, target: torch.Tensor, size: int) -> torch.Tensor:
    """Corner loss accepts finite, temporarily non-convex predictions during fitting."""
    if size not in SIZES or predicted.ndim != 3 or predicted.shape[1:] != (4, 2):
        raise ValueError("Expected [N,4,2] corners and supported input size")
    truth = torch.as_tensor(target, device=predicted.device, dtype=torch.float32)
    if truth.shape != predicted.shape or not torch.isfinite(truth).all():
        raise ValueError("Target must be finite [N,4,2] corners")
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Non-finite predicted corners")
    error = (predicted.float() - truth) / (.01 * math.hypot(size, size))
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction="mean")
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite coordinate loss")
    return loss
