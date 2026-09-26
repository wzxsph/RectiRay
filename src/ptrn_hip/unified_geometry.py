"""Versioned H8 + delta decoding and minimum-necessary, single-sample rectification.

Coordinates are pixel centres. Quad order is always TL, TR, BR, BL in the
source's orientation, including portrait sources; a long edge never defines width.
"""
from __future__ import annotations

import cv2
import numpy as np

from .ratio_synthesis import UNIT, pixel_corners, project

CONTRACT = 'ptrn-unified-geometry-v1'
DEFAULT_THRESHOLDS = {
    'shape': .005, 'ratio': .01, 'border': .005, 'angle_deg': .5,
    'boundary_band': .10, 'max_output_pixels': 24_000_000,
}
RELIABLE_SOURCES = frozenset({'reliable_digital', 'verified_digital', 'reliable_dicom',
                              'verified_dicom', 'confirmed_working', 'already_rectified'})


def edge_ratio(quad):
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    if not np.isfinite(lengths).all() or lengths.min() <= 1e-8:
        raise ValueError('degenerate_edge')
    return float(max(lengths[0], lengths[2]) / max(lengths[1], lengths[3]))


def inverse_letterbox_matrix(photo_size, size):
    width, height = map(float, photo_size)
    if min(width, height, size) < 2:
        raise ValueError('invalid_image_size')
    scale = (size - 1) / max(width - 1, height - 1)
    return np.array([[1 / scale, 0, -(size - 1 - scale * (width - 1)) / (2 * scale)],
                     [0, 1 / scale, -(size - 1 - scale * (height - 1)) / (2 * scale)],
                     [0, 0, 1]], dtype=np.float64)


def quad_reasons(quad, photo_size):
    q = np.asarray(quad, np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return ['invalid_coordinates']
    reasons = []
    if not cv2.isContourConvex(q.astype(np.float32)) or cv2.contourArea(q.astype(np.float32), oriented=True) <= 0:
        reasons.append('nonconvex_or_mirrored')
    bounds = np.asarray(photo_size, np.float64) - 1
    # Float32 homography export can shift a full-canvas boundary by <0.001 px.
    tolerance = max(1e-4, float(max(bounds)) * 2e-7)
    if (q < -tolerance).any() or (q > bounds + tolerance).any():
        reasons.append('outside_photo')
    if np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1).min() < 1:
        reasons.append('degenerate_edge')
    return reasons


def decode_prediction(prediction, size, photo_size, inverse_letterbox=None):
    """Decode only this contract; nine-vector final value is *delta*, not log r.

    Invalid coordinates remain available for error measurement where possible;
    `valid` additionally checks the homography denominators and photo boundaries.
    """
    result = {'quad': None, 'quad_input': None, 'log_ratio': None, 'ratio': None,
              'r_edge': None, 'valid': False, 'reasons': []}
    try:
        p = np.asarray(prediction, np.float64).reshape(-1)
        if p.shape != (9,) or not np.isfinite(p).all():
            raise ValueError('expected_nine_finite_unified_parameters')
        h = np.r_[p[:8], 1.].reshape(3, 3)
        denominators = np.c_[UNIT, np.ones(4)] @ h[2]
        q_input = (project(UNIT, h) + 1) * (size - 1) / 2
        inverse = inverse_letterbox_matrix(photo_size, size) if inverse_letterbox is None else np.asarray(inverse_letterbox, np.float64)
        q = project(q_input, inverse)
        r_edge = edge_ratio(q)
        log_ratio = float(np.log(r_edge) + p[8])
        if not np.isfinite(log_ratio) or abs(log_ratio) > 30:
            raise ValueError('unreasonable_ratio')
        ratio = float(np.exp(log_ratio))
        reasons = quad_reasons(q, photo_size)
        if (denominators <= 1e-7).any():
            reasons.append('nonpositive_projective_denominator')
        result.update(quad=q, quad_input=q_input, log_ratio=log_ratio, ratio=ratio,
                      r_edge=r_edge, valid=not reasons, reasons=reasons)
    except (ValueError, cv2.error, FloatingPointError, OverflowError) as error:
        result['reasons'] = [str(error)]
    return result


def fit_rotation_rectangle(quad):
    """Least-squares oriented rectangle, with the source width direction fixed.

    Maximise squared projections of opposite-edge averages on orthogonal axes;
    this is a two-dimensional symmetric eigenproblem. The enclosing rectangle
    used for sampling is separate from this least-squares residual rectangle.
    """
    q = np.asarray(quad, np.float64)
    centre = q.mean(axis=0)
    a = (q[1] + q[2] - q[0] - q[3]) / 4
    b = (q[2] + q[3] - q[0] - q[1]) / 4
    turned_b = np.array([b[1], -b[0]])
    _, vectors = np.linalg.eigh(np.outer(a, a) + np.outer(turned_b, turned_b))
    u = vectors[:, -1]
    if np.dot(u, a) < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    half_width, half_height = float(a @ u), float(b @ v)
    if min(half_width, half_height) <= 1e-7:
        raise ValueError('invalid_ordered_rectangle')
    fitted = centre + UNIT[:, :1] * half_width * u + UNIT[:, 1:] * half_height * v
    axes = np.stack([u, v])
    coordinates = q @ axes.T
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    spans = upper - lower
    return {'quad': fitted, 'axes': axes, 'lower': lower, 'upper': upper,
            'spans': spans, 'ratio': float(spans[0] / spans[1]),
            'fitted_ratio': half_width / half_height,
            'angle_deg': float(np.degrees(np.arctan2(u[1], u[0])))}


def _bounded_size(spans, max_pixels, *, enclose):
    if max_pixels < 4:
        raise ValueError('output_pixel_budget_too_small')
    quantize = np.ceil if enclose else np.floor
    def dimensions(scale):
        return tuple(map(int, np.maximum(2, quantize(spans * scale - (1e-9 if enclose else 0)).astype(int) + 1)))
    scale = 1.
    if np.prod(dimensions(scale)) > max_pixels:
        lower, upper = 0., 1.
        for _ in range(50):
            middle = (lower + upper) / 2
            if np.prod(dimensions(middle)) <= max_pixels:
                lower = middle
            else:
                upper = middle
        scale = lower
    return scale, dimensions(scale)


def similarity_crop(photo_size, quad, max_output_pixels=24_000_000):
    """Rotate and enclose all predicted corners with one uniform scale."""
    fitted = fit_rotation_rectangle(quad)
    spans = fitted['spans']
    # Ceil encloses extrema; account for rounding while enforcing the pixel cap.
    scale, output_size = _bounded_size(spans, max_output_pixels, enclose=True)
    matrix = np.eye(3)
    matrix[:2, :2] = fitted['axes'] * scale
    matrix[:2, 2] = -fitted['lower'] * scale
    return matrix, output_size, fitted


def _confirmation(photo_size, reason, metrics=None, valid=False):
    return {'action': 'needs_confirmation', 'matrix': None, 'output_size': tuple(map(int, photo_size)),
            'metrics': metrics or {}, 'valid': bool(valid), 'reason': reason}


def plan_transform(photo_size, quad, ratio, thresholds=None, source_kind='unknown', confirmed=False):
    """Plan a photo-to-output matrix, respecting trustworthy existing geometry.

    `confirmed=True` means an already confirmed *working image*, not permission
    to force an invalid prediction. Source provenance must be established by the
    caller; image appearance is never used as a substitute for that provenance.
    Ratio guards cover finite/legal numeric ranges only. No unapproved numerical
    threshold is invented for a subjective 'geometry/ratio contradiction'.
    """
    limits = dict(DEFAULT_THRESHOLDS)
    supplied = dict(thresholds or {})
    aliases = {'d_shape': 'shape', 'd_ratio': 'ratio', 'd_border': 'border',
               'rotation_deg': 'angle_deg', 'rotation_degrees': 'angle_deg',
               'ambiguity_fraction': 'boundary_band'}
    supplied = {aliases.get(key, key): value for key, value in supplied.items()}
    unknown = supplied.keys() - limits.keys()
    if unknown:
        raise ValueError('unknown_thresholds:' + ','.join(sorted(unknown)))
    limits.update(supplied)
    if any(not np.isfinite(value) or value <= 0 for key, value in limits.items() if key != 'boundary_band'):
        raise ValueError('thresholds_must_be_finite_positive')
    if not 0 <= limits['boundary_band'] < 1:
        raise ValueError('invalid_boundary_band')
    width, height = map(int, photo_size)
    if min(width, height) < 2:
        raise ValueError('invalid_image_size')
    if confirmed or source_kind in RELIABLE_SOURCES:
        return {'action': 'keep', 'matrix': np.eye(3), 'output_size': (width, height),
                'metrics': {}, 'valid': True, 'reason': 'reliable_existing_geometry', 'photo_size': (width, height)}
    reasons = quad_reasons(quad, photo_size)
    if reasons:
        return _confirmation(photo_size, ','.join(reasons))
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        return _confirmation(photo_size, 'invalid_ratio')
    if not np.isfinite(ratio) or ratio <= 0 or ratio > 100 or ratio < .01:
        return _confirmation(photo_size, 'invalid_ratio')
    try:
        matrix, output_size, fitted = similarity_crop(photo_size, quad, limits['max_output_pixels'])
        diagonal = float(np.hypot(width - 1, height - 1))
        metrics = {
            'd_shape': float(np.linalg.norm(np.asarray(quad) - fitted['quad'], axis=1).max() / diagonal),
            'd_ratio': float(np.expm1(abs(np.log(ratio / fitted['fitted_ratio'])))),
            'd_border': float(np.linalg.norm(np.asarray(quad) - pixel_corners(width, height), axis=1).max() / diagonal),
            'angle_deg': abs(fitted['angle_deg']), 'r_box': fitted['fitted_ratio'],
            'r_crop': fitted['ratio'],
        }
        if any(not np.isfinite(value) for value in metrics.values()):
            return _confirmation(photo_size, 'nonfinite_geometry_metrics')
        # Threshold bands are abstentions, never silent threshold clamping.
        pairs = [('d_shape', 'shape'), ('d_ratio', 'ratio')]
        low = all(metrics[name] < limits[key] * (1 - limits['boundary_band']) for name, key in pairs)
        boundary = any(abs(metrics[name] / limits[key] - 1) <= limits['boundary_band'] + 1e-12 for name, key in pairs)
        if boundary:
            return _confirmation(photo_size, 'correction_threshold_boundary', metrics, valid=True)
        if low:
            border_boundary = abs(metrics['d_border'] / limits['border'] - 1) <= limits['boundary_band'] + 1e-12
            angle_boundary = abs(metrics['angle_deg'] / limits['angle_deg'] - 1) <= limits['boundary_band'] + 1e-12
            if border_boundary or angle_boundary:
                return _confirmation(photo_size, 'keep_threshold_boundary', metrics, valid=True)
            if metrics['d_border'] < limits['border'] and metrics['angle_deg'] < limits['angle_deg']:
                return {'action': 'keep', 'matrix': np.eye(3), 'output_size': (width, height),
                        'metrics': metrics, 'valid': True, 'reason': 'full_canvas_within_tolerance', 'photo_size': (width, height)}
            return {'action': 'crop', 'matrix': matrix, 'output_size': output_size,
                    'metrics': metrics, 'valid': True, 'reason': 'similarity_crop_preserves_all_corners', 'photo_size': (width, height)}
        lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        out_height = min(max(lengths[1], lengths[3]), max(lengths[0], lengths[2]) / ratio)
        spans = np.array([out_height * ratio, out_height])
        _, output_size = _bounded_size(spans, limits['max_output_pixels'], enclose=False)
        destination = pixel_corners(*output_size).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            return _confirmation(photo_size, 'invalid_output_transform', metrics)
        return {'action': 'warp', 'matrix': matrix, 'output_size': output_size,
                'metrics': metrics, 'valid': True, 'reason': 'geometry_exceeds_similarity_tolerance', 'photo_size': (width, height)}
    except (ValueError, cv2.error, np.linalg.LinAlgError, FloatingPointError) as error:
        return _confirmation(photo_size, str(error))


def apply_transform(image, plan):
    """Sample from original pixels once; reliable keep is exactly unchanged."""
    if plan['action'] == 'needs_confirmation' or not plan['valid']:
        return None
    if plan['action'] == 'keep':
        return image
    if plan['action'] not in ('crop', 'warp'):
        raise ValueError('unknown_transform_action')
    return cv2.warpPerspective(image, np.asarray(plan['matrix'], np.float64),
                               tuple(map(int, plan['output_size'])), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
