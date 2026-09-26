"""Isolated candidate objectives for the spatial model's later C experiments.

These functions do not alter the frozen B fitting objective. Coordinates are
ordered TL/TR/BR/BL and refer to input pixel centres. Ratios describe the
original full source canvas, not anatomy or a crop of bright X-ray content.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class CoverageTerms(NamedTuple):
    total: torch.Tensor
    missed: torch.Tensor
    background: torch.Tensor
    invalid: torch.Tensor
    photo_boundary: torch.Tensor


class GridTerms(NamedTuple):
    total: torch.Tensor
    grid: torch.Tensor
    angle: torch.Tensor


def _quads(predicted, truth):
    if predicted.ndim != 3 or predicted.shape[1:] != (4, 2):
        raise ValueError("Predicted corners must have shape [N,4,2]")
    target = torch.as_tensor(truth, device=predicted.device, dtype=torch.float32)
    if target.shape != predicted.shape or not torch.isfinite(target).all():
        raise ValueError("GT corners must be finite [N,4,2]")
    if not torch.isfinite(predicted).all():
        raise FloatingPointError("Non-finite predicted corners")
    return predicted.float(), target


def _photo_size(photo_size):
    if len(photo_size) != 2:
        raise ValueError("photo_size must be (width,height)")
    width, height = map(int, photo_size)
    if width < 2 or height < 2:
        raise ValueError("Photo dimensions must be at least two pixels")
    return width, height


def _soft_quad_mask(quad, width, height, temperature_px):
    """Smooth signed-edge intersection; finite even for folded/collapsed quads."""
    y, x = torch.meshgrid(
        torch.arange(height, device=quad.device, dtype=quad.dtype),
        torch.arange(width, device=quad.device, dtype=quad.dtype),
        indexing="ij",
    )
    points = torch.stack((x, y), dim=-1)[None, :, :, None, :]
    start = quad[:, None, None, :, :]
    edge = torch.roll(quad, -1, dims=1) - quad
    edge = edge[:, None, None, :, :]
    offset = points - start
    cross = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
    # Smooth length floor retains a finite gradient at zero-length edges.
    length = torch.sqrt(edge.square().sum(dim=-1) + 1e-6)
    distance = cross / length
    return torch.sigmoid(distance / temperature_px).prod(dim=-1)


def _invalid_quad_penalty(quad, width, height):
    diag = math.hypot(width - 1, height - 1)
    edge = torch.roll(quad, -1, dims=1) - quad
    following = torch.roll(edge, -1, dims=1)
    turns = (edge[..., 0] * following[..., 1]
             - edge[..., 1] * following[..., 0]) / (diag * diag)
    # Positive turns are required by TL/TR/BR/BL image-coordinate ordering.
    turn_penalty = torch.log1p(F.relu((1e-4 - turns) / 1e-4)).mean()
    lengths = torch.linalg.vector_norm(edge, dim=-1) / diag
    short_penalty = (F.relu(.01 - lengths) / .01).mean()
    return turn_penalty + short_penalty


def _photo_boundary_penalty(quad, inverse_letterbox, photo_sizes):
    """Mean photo-diagonal-normalized corner excursion beyond captured pixels."""
    n = len(quad)
    inverse = torch.as_tensor(inverse_letterbox, device=quad.device, dtype=quad.dtype)
    sizes = torch.as_tensor(photo_sizes, device=quad.device, dtype=quad.dtype)
    if inverse.shape != (n, 3, 3) or sizes.shape != (n, 2):
        raise ValueError("Expected inverse [N,3,3] and photo sizes [N,2] (width,height)")
    if (not torch.isfinite(inverse).all() or not torch.isfinite(sizes).all()
            or (sizes < 2).any()):
        raise ValueError("Photo geometry must be finite with dimensions >= 2")
    homogeneous = torch.cat((quad, torch.ones_like(quad[..., :1])), dim=-1)
    projected = torch.matmul(homogeneous, inverse.transpose(1, 2))
    depth = projected[..., 2]
    if not torch.isfinite(projected).all() or (depth.abs() <= 1e-6).any():
        raise FloatingPointError("Unstable inverse letterbox in photo boundary loss")
    photo_quad = projected[..., :2] / depth.unsqueeze(-1)
    extent = sizes[:, None, :] - 1
    excursion = F.relu(-photo_quad) + F.relu(photo_quad - extent)
    diagonal = torch.linalg.vector_norm(sizes - 1, dim=-1).unsqueeze(-1)
    return (torch.linalg.vector_norm(excursion, dim=-1) / diagonal).mean()


def coverage_loss(pred_quad: torch.Tensor, truth_quad: torch.Tensor,
                  canvas_size: tuple[int, int], *, inverse_letterbox: torch.Tensor,
                  photo_sizes: torch.Tensor, temperature_px: float = 1.5,
                  missed_weight: float = 3., background_weight: float = 1.,
                  invalid_weight: float = 1., photo_boundary_weight: float = 5.) -> CoverageTerms:
    """Area-normalized coverage plus a true-photo-boundary legality cost.

    GT is rasterized with the same smooth rule as the prediction, so an exact
    match gives zero. Missed/background are means over letterbox canvas pixels.
    The extra boundary term uses only acquisition geometry, never GT film edges.
    """
    pred, truth = _quads(pred_quad, truth_quad)
    width, height = _photo_size(canvas_size)
    if min(temperature_px, missed_weight, background_weight, invalid_weight,
           photo_boundary_weight) <= 0:
        raise ValueError("Temperature and loss weights must be positive")
    pred_mask = _soft_quad_mask(pred, width, height, temperature_px)
    truth_mask = _soft_quad_mask(truth, width, height, temperature_px)
    missed = F.relu(truth_mask - pred_mask).mean()
    background = F.relu(pred_mask - truth_mask).mean()
    invalid = _invalid_quad_penalty(pred, width, height)
    photo_boundary = _photo_boundary_penalty(pred, inverse_letterbox, photo_sizes)
    total = (missed_weight * missed + background_weight * background
             + invalid_weight * invalid + photo_boundary_weight * photo_boundary)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite soft coverage loss")
    return CoverageTerms(total, missed, background, invalid, photo_boundary)


def _solve_homography(source: torch.Tensor, destination: torch.Tensor):
    """Differentiable four-point DLT with h33 fixed to 1."""
    x, y = source.unbind(dim=-1)
    u, v = destination.unbind(dim=-1)
    one, zero = torch.ones_like(x), torch.zeros_like(x)
    row_u = torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=2).reshape(-1, 8, 8)
    rhs = torch.stack((u, v), dim=2).reshape(-1, 8, 1)
    try:
        solution = torch.linalg.solve(matrix, rhs).reshape(-1, 8)
    except RuntimeError as error:
        raise FloatingPointError("Singular four-corner homography in grid loss") from error
    homography = torch.cat((solution, torch.ones_like(solution[:, :1])), dim=1).reshape(-1, 3, 3)
    if not torch.isfinite(homography).all():
        raise FloatingPointError("Non-finite homography in grid loss")
    return homography


def _project(points: torch.Tensor, homography: torch.Tensor):
    homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    warped = torch.matmul(homogeneous, homography.transpose(1, 2))
    depth = warped[..., 2]
    if not torch.isfinite(warped).all() or (depth.abs() <= 1e-6).any():
        raise FloatingPointError("Unstable projective denominator in grid loss")
    return warped[..., :2] / depth.unsqueeze(-1)


def grid_geometry_loss(pred_quad: torch.Tensor, truth_quad: torch.Tensor,
                       pred_ratio: torch.Tensor, truth_ratio: torch.Tensor,
                       photo_size: tuple[int, int], *, angle_weight: float = 0.) -> GridTerms:
    """GT 9x9 lattice after predicted rectification, with one isotropic scale.

    The lattice comes from the GT quad; the correction comes only from the
    predicted quad and predicted full-canvas ratio. Translation and one common
    scale are fitted, but x/y are never scaled independently and rotation is
    not aligned away. Bad homographies fail explicitly rather than skipping a
    sample. ``angle`` is a separate, optional rotation penalty.
    """
    pred, truth = _quads(pred_quad, truth_quad)
    width, height = _photo_size(photo_size)
    n = pred.shape[0]
    ratio = torch.as_tensor(pred_ratio, device=pred.device, dtype=torch.float32).reshape(-1)
    target_ratio = torch.as_tensor(truth_ratio, device=pred.device, dtype=torch.float32).reshape(-1)
    if ratio.shape != (n,) or target_ratio.shape != (n,):
        raise ValueError("Ratios must each have shape [N]")
    if (not torch.isfinite(ratio).all() or not torch.isfinite(target_ratio).all()
            or (ratio <= 0).any() or (target_ratio <= 0).any() or angle_weight < 0):
        raise ValueError("Ratios must be positive finite and angle_weight nonnegative")
    scale = pred.new_tensor([width - 1, height - 1])
    pred_input = pred * (2 / scale) - 1
    truth_input = truth * (2 / scale) - 1
    unit = pred.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    unit = unit.unsqueeze(0).expand(n, -1, -1)
    rectangle = torch.stack((unit[..., 0] * ratio[:, None] / 2,
                             unit[..., 1] / 2), dim=-1)
    gt_to_photo = _solve_homography(unit, truth_input)
    photo_to_output = _solve_homography(pred_input, rectangle)
    axis = torch.linspace(-1., 1., 9, device=pred.device, dtype=pred.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    lattice = torch.stack((xx.flatten(), yy.flatten()), dim=-1).unsqueeze(0).expand(n, -1, -1)
    photo_grid = _project(lattice, gt_to_photo)
    observed = _project(photo_grid, photo_to_output)
    canonical = torch.stack((lattice[..., 0] * target_ratio[:, None] / 2,
                             lattice[..., 1] / 2), dim=-1)
    observed_centered = observed - observed.mean(dim=1, keepdim=True)
    canonical_centered = canonical - canonical.mean(dim=1, keepdim=True)
    energy = observed_centered.square().sum(dim=(1, 2))
    if (energy <= 1e-10).any() or not torch.isfinite(energy).all():
        raise FloatingPointError("Collapsed output lattice in grid loss")
    common_scale = (observed_centered * canonical_centered).sum(dim=(1, 2)) / energy
    aligned = observed_centered * common_scale.clamp_min(0)[:, None, None]
    diag = torch.sqrt(target_ratio.square() + 1)[:, None, None]
    normalized = (aligned - canonical_centered) / diag
    grid = F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=.005)
    horizontal = observed.reshape(n, 9, 9, 2)[:, 4, -1] - observed.reshape(n, 9, 9, 2)[:, 4, 0]
    angle = (1 - horizontal[:, 0] / torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-8)).mean()
    total = grid + angle_weight * angle
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite grid geometry loss")
    return GridTerms(total, grid, angle)


def detached_log_edge_ratio(corners: torch.Tensor) -> torch.Tensor:
    """Photo edge-ratio baseline without any gradient into localization."""
    if corners.ndim != 3 or corners.shape[1:] != (4, 2) or not torch.isfinite(corners).all():
        raise ValueError("Finite [N,4,2] corners required")
    q = corners.detach().float()
    edge = torch.linalg.vector_norm(torch.roll(q, -1, dims=1) - q, dim=-1)
    if (edge <= 1e-6).any():
        raise FloatingPointError("Degenerate edge in ratio baseline")
    horizontal = torch.maximum(edge[:, 0], edge[:, 2])
    vertical = torch.maximum(edge[:, 1], edge[:, 3])
    return horizontal.log() - vertical.log()


class RatioResidual(nn.Module):
    """Independent small δ(log W/H) module on detached corners and features."""

    def __init__(self, feature_dim: int = 1920, hidden: int = 64, size: int = 224):
        super().__init__()
        if min(feature_dim, hidden, size) < 2:
            raise ValueError("Invalid ratio-module dimensions")
        self.feature_dim, self.size = feature_dim, size
        self.network = nn.Sequential(nn.Linear(feature_dim + 8, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, corners: torch.Tensor, pooled_features: torch.Tensor) -> torch.Tensor:
        if (corners.ndim != 3 or corners.shape[1:] != (4, 2)
                or pooled_features.ndim != 2 or pooled_features.shape != (corners.shape[0], self.feature_dim)):
            raise ValueError("Expected corners [N,4,2] and pooled features [N,C]")
        if not torch.isfinite(corners).all() or not torch.isfinite(pooled_features).all():
            raise ValueError("Non-finite ratio-module inputs")
        geometry = (corners.detach().float() / (self.size - 1)).reshape(-1, 8)
        features = pooled_features.detach().float()
        return self.network(torch.cat((geometry, features), dim=1)).reshape(-1)

    def decode_log_ratio(self, corners: torch.Tensor, pooled_features: torch.Tensor) -> torch.Tensor:
        return detached_log_edge_ratio(corners) + self(corners, pooled_features)


def ratio_residual_loss(pred_log_ratio: torch.Tensor, truth_log_ratio: torch.Tensor) -> torch.Tensor:
    truth = torch.as_tensor(truth_log_ratio, device=pred_log_ratio.device, dtype=torch.float32)
    if truth.shape != pred_log_ratio.shape or not torch.isfinite(truth).all():
        raise ValueError("Finite [N] log-ratio targets required")
    error = (pred_log_ratio.float() - truth) / .05
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite ratio loss")
    return loss
