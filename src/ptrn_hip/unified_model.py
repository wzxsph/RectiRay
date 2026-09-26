"""One Dense9 head: H8 plus a log-ratio residual, initialized strictly from G0.

The raw ninth value is delta, never the complete log ratio of ratio-v1 models.
Training uses current predicted corners even when encoder features are cached.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np

from .model import keras, torch
from .ratio_data import sha256
from .ratio_model import B11_SHA, build_ratio, corner_tensor, tensor_input

G0_SHA = "e3cb051205fed108d35e2603cc5266e69a18d394de7466d30759dbdab5cd9e32"
CONTRACT = "ptrn-unified-geometry-v1"
SIZES = (224, 384)
FEATURE_DIM = 1920
_cycle_collection_calls = 0


def _collect_unreachable_cycles():
    """Bound delayed Keras/Torch graph cycles without touching live tensors.

    Python's generation-2 collection may be delayed by the large source-view
    metadata heap. Fixed call intervals keep discarded CUDA graphs from growing
    until OOM; collection never changes sampling, parameters or live gradients.
    """
    global _cycle_collection_calls
    _cycle_collection_calls += 1
    if _cycle_collection_calls % 16 == 0:
        gc.collect()


def _size(size):
    if isinstance(size, bool) or size not in SIZES:
        raise ValueError("Unified model input size must be 224 or 384")
    return int(size)


def contract(size):
    """Public TorchScript input is NHWC RGB 0..255; raw output is H8 + delta."""
    size = _size(size)
    return {
        "contract": CONTRACT,
        "input_size": size,
        "input_shape": [None, size, size, 3],
        "input_dtype": "float32",
        "input_range": "RGB 0..255",
        "input_preprocessing": {"scale": 255.0, "mean": [.485, .456, .406],
                                "std": [.229, .224, .225], "layout": "NHWC"},
        "output_shape": [None, 9],
        "output_dtype": "float32",
        "output_fields": ["h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "delta_log_ratio"],
        "ninth_value": "delta_log_ratio; NOT complete log(W/H)",
        "ratio_decode": "log(W/H)=log(max(top,bottom)/max(left,right))+output[8]",
        "ratio_geometry": "edge lengths from predicted corners in original photo pixel coordinates after inverse letterbox",
        "ratio_gradient": "detach log(edge_ratio) only; supervise H8 with corner loss",
        "homography_direction": "source_normalized_TL_TR_BR_BL_to_input_normalized",
        "homography_layout": "row_major_first8_h33_equals1",
        "normalization": "[-1,1] maps pixel centers 0..size-1",
        "padding": "isotropic center-span letterbox; black; OpenCV warpAffine INTER_LINEAR",
        "source_corners": "(0,0),(W-1,0),(W-1,H-1),(0,H-1); boundaries +/-0.5",
        "ratio_target": "original EXIF full canvas W/H, including original borders; not physical anatomy",
        "ratio_evidence": "model_estimated",
        "independent_real_ratio_validated": False,
        "incompatible_contracts": ["ptrn-ratio-v1-h8-logratio9", "B1.1-size-normalization"],
    }


def _checked_payload(checkpoint, expected_sha=None):
    """Check the external digest/config before accepting a local checkpoint."""
    checkpoint = Path(checkpoint)
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    digest = sha256(checkpoint)
    if digest != sidecar.get("sha256") or (expected_sha is not None and digest != expected_sha):
        raise ValueError("Checkpoint SHA mismatch")
    # Payloads written by our runner contain tensors and primitive containers only.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a state_dict")
    if not isinstance(payload.get("config"), dict):
        raise ValueError("Checkpoint must contain a configuration mapping")
    if payload.get("config") != sidecar.get("config"):
        raise ValueError("Checkpoint configuration differs from SHA sidecar")
    if "step" in sidecar and payload.get("step") != sidecar["step"]:
        raise ValueError("Checkpoint step differs from SHA sidecar")
    return payload


def read_unified_checkpoint(checkpoint):
    payload = _checked_payload(checkpoint)
    config = payload.get("config", {})
    size = _size(config.get("input_size"))
    if config.get("contract") != contract(size):
        raise ValueError("Checkpoint is not the explicit H8+delta unified contract")
    return payload


def build_unified(weights, g0_checkpoint, size=224):
    """Return (model, backbone, head), with G0 H8 copied and delta exactly zero.

    Model calls accept normalized NHWC tensors; use tensor_input/predict for RGB.
    Encoder and BN start frozen. All nine head outputs remain trainable.
    """
    size = _size(size)
    if sha256(Path(weights)) != B11_SHA:
        raise ValueError("B1.1 architecture initialization SHA mismatch")
    payload = _checked_payload(g0_checkpoint, G0_SHA)
    config = payload["config"]
    if (payload.get("step") != 1250 or config.get("group") != "G0"
            or config.get("input_size") != 224 or config.get("ratio_head") is not False
            or config.get("initial_weights_sha256") != B11_SHA):
        raise ValueError("Approved G0 checkpoint identity/configuration mismatch")
    legacy, backbone = build_ratio(Path(weights), 224, False, 0.0)
    legacy.load_state_dict(payload["state_dict"], strict=True)
    old_kernel, old_bias = legacy.get_layer("homography").get_weights()
    if old_kernel.shape != (FEATURE_DIM, 8) or old_bias.shape != (8,):
        raise ValueError("G0 head must be Dense(1920, 8)")
    inputs = keras.Input((size, size, 3), name="photo_unified_geometry_v1")
    # Fix BN inference behavior even if a caller requests training=True on model.
    features = backbone(inputs, training=False)
    head = keras.layers.Dense(9, activation="linear", kernel_initializer="zeros",
                              bias_initializer="zeros", dtype="float32", name="unified_geometry")
    model = keras.Model(inputs, head(features), name="ptrn_unified_geometry_v1")
    kernel = np.zeros((FEATURE_DIM, 9), dtype=np.float32)
    bias = np.zeros(9, dtype=np.float32)
    kernel[:, :8], bias[:8] = old_kernel, old_bias
    head.set_weights([kernel, bias])
    parameter_groups(model, backbone, head, "head")
    return model, backbone, head


def load_unified(weights, g0_checkpoint, checkpoint, size=None):
    """Load an audited unified checkpoint; size override enables the 384 control."""
    payload = read_unified_checkpoint(checkpoint)
    chosen_size = payload["config"]["input_size"] if size is None else _size(size)
    model, backbone, head = build_unified(weights, g0_checkpoint, chosen_size)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, backbone, head


def parameter_groups(model, backbone, head, stage):
    """Adam parameter groups: whole Dense9 1e-4; optional conv5 block 1e-5."""
    if stage not in {"head", "last_block"}:
        raise ValueError("Stage must be head or last_block")
    model.trainable = True
    backbone.trainable = stage == "last_block"
    if stage == "last_block":
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("conv5_") and not isinstance(layer, keras.layers.BatchNormalization)
    head.trainable = True
    heads = [value.value for value in head.trainable_weights]
    trunk = [value.value for value in backbone.trainable_weights]
    if len(heads) != 2 or sum(p.numel() for p in heads) != FEATURE_DIM * 9 + 9:
        raise ValueError("Unified training requires the whole Dense9 kernel and bias")
    if stage == "last_block" and not trunk:
        raise ValueError("No trainable conv5 parameters found")
    groups = [{"params": heads, "lr": 1e-4, "name": "unified_head"}]
    if trunk:
        groups.append({"params": trunk, "lr": 1e-5, "name": "conv5"})
    if len({id(p) for group in groups for p in group["params"]}) != sum(len(group["params"]) for group in groups):
        raise ValueError("Duplicate optimizer parameter")
    return groups


def _valid_quad(quad):
    if not torch.isfinite(quad).all():
        raise FloatingPointError("Non-finite predicted corners")
    edges = torch.roll(quad, -1, dims=1) - quad
    following = torch.roll(edges, -1, dims=1)
    cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
    if not torch.isfinite(cross).all() or (cross <= 1e-7).any():
        raise FloatingPointError("Degenerate, reflected or non-convex predicted quadrilateral")


def decode_prediction(prediction, size, inverse_letterbox=None):
    """Return network corners, photo corners, and complete log ratio (tensors).

    inverse_letterbox is [N,3,3]. None is appropriate for no letterbox or an
    isotropic letterbox when only the scale-invariant edge ratio is needed.
    Neither invalid projective depths nor invalid edges are silently clamped.
    """
    size = _size(size)
    if prediction.ndim != 2 or prediction.shape[1] != 9 or prediction.shape[0] == 0:
        raise ValueError("Unified prediction must have shape [N,9], N > 0")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Non-finite unified prediction")
    network_quad = corner_tensor(prediction, size)
    photo_quad = network_quad
    if inverse_letterbox is not None:
        inverse = torch.as_tensor(inverse_letterbox, device=prediction.device, dtype=torch.float32).detach()
        if inverse.shape != (prediction.shape[0], 3, 3):
            raise ValueError("inverse_letterbox must have shape [N,3,3]")
        if not torch.isfinite(inverse).all():
            raise FloatingPointError("Non-finite inverse letterbox")
        homogeneous = torch.cat([network_quad, torch.ones_like(network_quad[..., :1])], dim=2)
        projected = torch.matmul(homogeneous, inverse.transpose(1, 2))
        if not torch.isfinite(projected).all() or (projected[..., 2] <= 1e-7).any():
            raise FloatingPointError("Invalid inverse-letterbox projection")
        photo_quad = projected[..., :2] / projected[..., 2:]
    _valid_quad(photo_quad)
    edge = torch.linalg.vector_norm(torch.roll(photo_quad, -1, dims=1) - photo_quad, dim=2)
    if not torch.isfinite(edge).all() or (edge <= 1e-7).any():
        raise FloatingPointError("Zero-length predicted edge")
    horizontal = torch.maximum(edge[:, 0], edge[:, 2])
    vertical = torch.maximum(edge[:, 1], edge[:, 3])
    log_edge = torch.log(horizontal) - torch.log(vertical)
    log_ratio = log_edge.detach() + prediction[:, 8].float()
    return network_quad, photo_quad, log_ratio


def losses(prediction, quad, log_ratio, size, inverse_letterbox=None):
    _collect_unreachable_cycles()
    predicted, _, predicted_log_ratio = decode_prediction(prediction, size, inverse_letterbox)
    truth = torch.as_tensor(quad, dtype=torch.float32, device=prediction.device)
    ratio_truth = torch.as_tensor(log_ratio, dtype=torch.float32, device=prediction.device)
    if truth.shape != predicted.shape or ratio_truth.shape != (prediction.shape[0],):
        raise ValueError("Targets require [N,4,2] corners and [N] log ratios")
    if not torch.isfinite(truth).all() or not torch.isfinite(ratio_truth).all():
        raise FloatingPointError("Non-finite geometry target")
    normalized_error = (predicted - truth) / (.01 * math.hypot(size, size))
    quad_loss = torch.nn.functional.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), beta=1.0, reduction="mean")
    ratio_error = (predicted_log_ratio - ratio_truth) / .05
    ratio_loss = torch.nn.functional.smooth_l1_loss(ratio_error, torch.zeros_like(ratio_error), beta=1.0, reduction="mean")
    total = quad_loss + .25 * ratio_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite unified training loss; no batch skipped")
    return total, quad_loss, ratio_loss


def forward_features(head, features):
    """Differentiable head forward from cached f, always computing current H8."""
    _collect_unreachable_cycles()
    features = torch.as_tensor(features, device=head.kernel.value.device, dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or features.shape[0] == 0:
        raise ValueError("Cached features must have shape [N,1920], N > 0")
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite cached features")
    return head(features, training=False).float()


def feature_predict(backbone, pixels, device):
    """Frozen-encoder FP32 cache values; same preprocessing as online predict."""
    _collect_unreachable_cycles()
    if backbone.trainable_weights:
        raise ValueError("Feature caching requires a frozen encoder")
    with torch.no_grad():
        features = backbone(tensor_input(pixels, device), training=False).float()
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not torch.isfinite(features).all():
        raise FloatingPointError("Invalid frozen encoder feature output")
    return features.detach().cpu().numpy()


def predict(model, pixels, device):
    _collect_unreachable_cycles()
    with torch.no_grad():
        output = model(tensor_input(pixels, device), training=False).float()
    if output.ndim != 2 or output.shape[1] != 9 or not torch.isfinite(output).all():
        raise FloatingPointError("Invalid unified model output")
    return output.detach().cpu().numpy()
