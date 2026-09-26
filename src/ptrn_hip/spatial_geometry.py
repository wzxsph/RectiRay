"""Predicted-only spatial action policy for the v2 research experiment.

The policy estimates a canonical 9x9 film grid from predicted photo corners and
the existing full-canvas W/H ratio. It never reads ground-truth source geometry.
An optional output margin is an *uncalibrated preset*, not an uncertainty bound.
"""
from __future__ import annotations

import cv2
import numpy as np

from .ratio_synthesis import UNIT, pixel_corners, project
from .unified_geometry import (RELIABLE_SOURCES, _bounded_size, edge_ratio,
                               fit_rotation_rectangle, quad_reasons)

CONTRACT = 'ptrn-spatial-policy-v2'
DEFAULT_THRESHOLDS = {
    'shape': .005, 'ratio': .01, 'border': .005, 'angle_deg': .5,
    'boundary_band': .10, 'background': .05, 'max_output_pixels': 24_000_000,
    'margin_fraction': 0.,
}


def _confirmation(photo_size, reason, metrics=None, *, valid=False):
    return {'action': 'needs_confirmation', 'matrix': None,
            'output_size': tuple(map(int, photo_size)), 'photo_size': tuple(map(int, photo_size)),
            'metrics': metrics or {}, 'valid': bool(valid), 'reason': reason,
            'contract': CONTRACT}


def _result(action, matrix, output_size, photo_size, reason, metrics):
    return {'action': action, 'matrix': matrix, 'output_size': tuple(map(int, output_size)),
            'photo_size': tuple(map(int, photo_size)), 'metrics': metrics,
            'valid': True, 'reason': reason, 'contract': CONTRACT}


def _limits(thresholds):
    limits = dict(DEFAULT_THRESHOLDS)
    aliases = {'d_shape': 'shape', 'd_ratio': 'ratio', 'd_border': 'border',
               'rotation_deg': 'angle_deg', 'rotation_degrees': 'angle_deg',
               'ambiguity_fraction': 'boundary_band', 'safety_margin': 'margin_fraction'}
    supplied = {aliases.get(k, k): v for k, v in dict(thresholds or {}).items()}
    unknown = supplied.keys() - limits.keys()
    if unknown:
        raise ValueError('unknown_thresholds:' + ','.join(sorted(unknown)))
    limits.update(supplied)
    if any(not np.isfinite(v) or v <= 0 for k, v in limits.items()
           if k not in ('boundary_band', 'margin_fraction')):
        raise ValueError('thresholds_must_be_finite_positive')
    if not 0 <= limits['boundary_band'] < 1:
        raise ValueError('invalid_boundary_band')
    if not np.isfinite(limits['margin_fraction']) or not 0 <= limits['margin_fraction'] < .25:
        raise ValueError('invalid_margin_fraction')
    return limits


def _lattice(quad, ratio, matrix, output_size):
    """Similarity-invariant grid max, ratio and angle from q/r only."""
    fractions = np.linspace(0., 1., 9)
    xx, yy = np.meshgrid(fractions, fractions)
    uv = np.c_[xx.ravel(), yy.ravel()]
    h = cv2.getPerspectiveTransform(UNIT.astype(np.float32), np.asarray(quad, np.float32))
    actual = project(project(uv * 2 - 1, h), matrix)
    bounds = np.asarray(output_size, np.float64) - 1
    if bounds.min() < 1:
        raise ValueError('bad_output_size')
    observed = (actual - bounds / 2) * (1024 / max(bounds))
    target = np.c_[ratio * (uv[:, 0] - .5), uv[:, 1] - .5]
    target *= 1024 / max(ratio, 1.)
    diagonal = np.hypot(ratio, 1.) * 1024 / max(ratio, 1.)
    ac, tc = observed - observed.mean(axis=0), target - target.mean(axis=0)
    u, singular, vt = np.linalg.svd(ac.T @ tc)
    signs = np.array([1., -1. if np.linalg.det(u @ vt) < 0 else 1.])
    rotation = (u * signs) @ vt
    energy = float(np.sum(ac * ac))
    if energy < 1e-12:
        raise ValueError('collapsed_output_grid')
    aligned = ac @ rotation * (np.sum(singular * signs) / energy) + target.mean(axis=0)
    errors = np.linalg.norm(aligned - target, axis=1) / diagonal
    transformed = project(np.asarray(quad, np.float64), matrix)
    ratio_error = float(np.expm1(abs(np.log(edge_ratio(transformed) / ratio))))
    angle = abs(fit_rotation_rectangle(transformed)['angle_deg'])
    return {'grid_error_max': float(errors.max()), 'grid_error_p90': float(np.percentile(errors, 90)),
            'ratio_error': ratio_error, 'angle_deg': angle}


def _footprint(quad, matrix, output_size):
    transformed = project(np.asarray(quad, np.float64), matrix)
    bounds = np.asarray(output_size, np.float64) - 1
    if bounds.min() < 1:
        raise ValueError('bad_output_size')
    inside = bool(((transformed >= -.5) & (transformed <= bounds + .5)).all())
    area = float(cv2.contourArea(transformed.astype(np.float32)))
    background = max(0., 1 - min(1., area / float(np.prod(bounds))))
    return {'predicted_corners_inside': inside, 'predicted_background_fraction': background}


def _crop(photo_size, quad, maximum, margin):
    fitted = fit_rotation_rectangle(quad)
    spans = fitted['spans']
    lower = fitted['lower'] - spans * margin
    enlarged = spans * (1 + 2 * margin)
    scale, output_size = _bounded_size(enlarged, maximum, enclose=True)
    matrix = np.eye(3)
    matrix[:2, :2] = fitted['axes'] * scale
    matrix[:2, 2] = -lower * scale
    return matrix, output_size


def _warp(quad, ratio, photo_size, maximum, margin):
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    height = min(max(lengths[1], lengths[3]), max(lengths[0], lengths[2]) / ratio)
    spans = np.array([height * ratio, height]) / (1 - 2 * margin)
    _, output_size = _bounded_size(spans, maximum, enclose=False)
    bounds = np.asarray(output_size, np.float64) - 1
    inset = bounds * margin
    destination = pixel_corners(*output_size).astype(np.float64)
    destination = inset + destination * (1 - 2 * margin)
    matrix = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination.astype(np.float32))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('bad_warp_transform')
    return matrix, output_size


def _near(value, threshold, band):
    return abs(value / threshold - 1) <= band + 1e-12


def _margin_feasible(matrix, output_size, photo_size, margin):
    """Check the actual output footprint against captured input photo pixels.

    Radially expanding the predicted quad is not equivalent to inverse-mapping
    the corners of a perspective output rectangle. A feasible-looking quad can
    still require original-photo pixels beyond a side after the warp.
    """
    if margin == 0:
        return True
    source_corners = project(pixel_corners(*output_size),
                             np.linalg.inv(np.asarray(matrix, np.float64)))
    bounds = np.asarray(photo_size, np.float64) - 1
    tolerance = max(1e-5, float(max(bounds)) * 2e-7)
    return bool(np.isfinite(source_corners).all() and
                ((source_corners >= -tolerance) &
                 (source_corners <= bounds + tolerance)).all())


def plan_transform(photo_size, quad, ratio, thresholds=None, source_kind='unknown', confirmed=False):
    """Choose keep/crop/warp/confirmation with predicted-only final geometry.

    Signature and plan keys match unified_geometry.plan_transform. A reliable
    provenance bypass is exact. Unknown full-canvas predictions must be valid;
    a corner outside the photo is never clamped into a keep decision.
    """
    limits = _limits(thresholds)
    photo_size = tuple(map(int, photo_size))
    if len(photo_size) != 2 or min(photo_size) < 2:
        raise ValueError('invalid_image_size')
    if confirmed or source_kind in RELIABLE_SOURCES:
        return _result('keep', np.eye(3), photo_size, photo_size,
                       'reliable_existing_geometry', {'provenance_bypass': True})
    reasons = quad_reasons(quad, photo_size)
    if reasons:
        return _confirmation(photo_size, ','.join(reasons))
    try:
        q, r = np.asarray(quad, np.float64), float(ratio)
        if not np.isfinite(r) or not .01 <= r <= 100:
            return _confirmation(photo_size, 'invalid_ratio')
        margin = float(limits['margin_fraction'])
        crop_matrix, crop_size = _crop(photo_size, q, limits['max_output_pixels'], margin)
        crop = {**_lattice(q, r, crop_matrix, crop_size), **_footprint(q, crop_matrix, crop_size)}
        crop.update(margin_fraction=margin, margin_calibrated=False)
        if not all(np.isfinite(v) for v in crop.values() if isinstance(v, (float, np.floating))):
            return _confirmation(photo_size, 'nonfinite_candidate_metrics', crop)
        shape, ratio_error = crop['grid_error_max'], crop['ratio_error']
        band = limits['boundary_band']
        if _near(shape, limits['shape'], band) or _near(ratio_error, limits['ratio'], band):
            return _confirmation(photo_size, 'correction_threshold_boundary', crop, valid=True)
        can_crop = (shape < limits['shape'] * (1 - band) and
                    ratio_error < limits['ratio'] * (1 - band) and
                    crop['angle_deg'] <= limits['angle_deg'] and
                    crop['predicted_corners_inside'])
        if can_crop:
            width, height = photo_size
            border = float(np.linalg.norm(q - pixel_corners(width, height), axis=1).max() /
                           np.hypot(width - 1, height - 1))
            kept = {**_lattice(q, r, np.eye(3), photo_size), **_footprint(q, np.eye(3), photo_size)}
            crop.update(d_border=border, predicted_keep=kept)
            # Only abstain at a keep boundary when keep is otherwise plausible.
            # A tilted photo with substantial exterior is unambiguously a crop,
            # even if its raw angle happens to be near the keep angle threshold.
            keep_plausible = (border <= limits['border'] * (1 + band) and
                              kept['predicted_background_fraction'] <= limits['background'] and
                              kept['grid_error_max'] < limits['shape'] * (1 - band) and
                              kept['ratio_error'] < limits['ratio'] * (1 - band))
            if keep_plausible and (_near(border, limits['border'], band) or
                                   _near(kept['angle_deg'], limits['angle_deg'], band)):
                return _confirmation(photo_size, 'keep_threshold_boundary', crop, valid=True)
            if (keep_plausible and border < limits['border'] * (1 - band) and
                    kept['angle_deg'] < limits['angle_deg'] * (1 - band) and
                    kept['predicted_corners_inside']):
                return _result('keep', np.eye(3), photo_size, photo_size,
                               'predicted_full_canvas', crop)
            if not _margin_feasible(crop_matrix, crop_size, photo_size, margin):
                return _confirmation(photo_size, 'margin_exceeds_captured_photo', crop, valid=True)
            if crop['predicted_background_fraction'] > limits['background']:
                return _confirmation(photo_size, 'predicted_crop_background_exceeds_limit', crop, valid=True)
            return _result('crop', crop_matrix, crop_size, photo_size,
                           'predicted_lattice_crop_acceptable', crop)
        warp_matrix, warp_size = _warp(q, r, photo_size, limits['max_output_pixels'], margin)
        if not _margin_feasible(warp_matrix, warp_size, photo_size, margin):
            return _confirmation(photo_size, 'margin_exceeds_captured_photo', crop, valid=True)
        warp = {**_lattice(q, r, warp_matrix, warp_size), **_footprint(q, warp_matrix, warp_size)}
        warp.update(margin_fraction=margin, margin_calibrated=False, crop_candidate=crop)
        if not warp['predicted_corners_inside'] or warp['predicted_background_fraction'] > limits['background']:
            return _confirmation(photo_size, 'predicted_warp_coverage_or_background', warp, valid=True)
        return _result('warp', warp_matrix, warp_size, photo_size,
                       'predicted_lattice_requires_warp', warp)
    except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        return _confirmation(photo_size, str(error))
