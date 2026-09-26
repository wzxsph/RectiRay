"""Visible-content rectification. V3 virtual corners may be outside the photo."""
from dataclasses import dataclass
import cv2
import numpy as np
from .ratio_synthesis import pixel_corners, project

CONTRACT = 'ptrn-visible-plane-v3'


def valid_quad(quad):
    q = np.asarray(quad, np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError('Expected finite TL/TR/BR/BL corners')
    edges = np.roll(q, -1, axis=0) - q
    nxt = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
    if np.min(turns) <= 1e-6 or np.min(np.linalg.norm(edges, axis=1)) <= 1e-3:
        raise ValueError('Folded, mirrored or degenerate plane')
    return q


def visible_polygon(quad, width, height):
    q = valid_quad(quad)
    if min(width, height) < 2:
        raise ValueError('Photo must be at least 2x2')
    area, polygon = cv2.intersectConvexConvex(q.astype(np.float32),
                                            pixel_corners(width, height).astype(np.float32))
    if polygon is None or area <= 1e-6:
        raise ValueError('Plane has no visible area')
    return polygon.reshape(-1, 2).astype(np.float64)


def visibility(quad, transform, plane_size, photo_size):
    w, h = photo_size
    poly = visible_polygon(quad, w, h)
    plane = project(poly, np.linalg.inv(transform))
    pw, ph = plane_size
    fraction = float(np.clip(cv2.contourArea(plane.astype(np.float32)) / ((pw-1)*(ph-1)), 0, 1))
    q = np.asarray(quad)
    corners = ((q >= 0) & (q <= [w-1, h-1])).all(axis=1)
    # Length in photo coordinates of each original edge that was captured.
    lengths = []
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        lo, hi = 0., 1.
        for dim, limit in enumerate((w-1, h-1)):
            delta = b[dim]-a[dim]
            if abs(delta) < 1e-10:
                if a[dim] < 0 or a[dim] > limit: hi = -1.
            else:
                u, v = sorted(((0-a[dim])/delta, (limit-a[dim])/delta))
                lo, hi = max(lo, u), min(hi, v)
        lengths.append(max(0., hi-lo))
    return {'visible_fraction': fraction, 'missing_fraction': 1-fraction,
            'visible_polygon': poly.tolist(), 'corner_visible': corners.tolist(),
            'edge_visible_fraction': lengths}


@dataclass
class Rectified:
    pixels: np.ndarray
    valid_mask: np.ndarray
    photo_to_output: np.ndarray
    info: dict


def _uniform_minification(transform, polygon):
    """Certify that the photo-to-output map shrinks every visible direction.

    The Jacobian numerator is affine over the convex visible polygon. Its
    spectral norm is therefore bounded by the largest vertex norm; the
    projective denominator's absolute value is bounded below by its smallest
    vertex value. This bound can decline anti-aliasing for a strongly varying
    perspective, but cannot select it for a locally magnified region.
    """
    p = np.asarray(polygon, np.float64)
    depth = p @ transform[2, :2] + transform[2, 2]
    if not np.isfinite(depth).all() or np.min(depth) * np.max(depth) <= 0:
        return False
    numerator = p @ transform[:2, :2].T + transform[:2, 2]
    derivatives = (transform[:2, :2][None] * depth[:, None, None]
                   - numerator[:, :, None] * transform[2, :2][None, None])
    upper_bound = max(np.linalg.svd(matrix, compute_uv=False)[0] for matrix in derivatives)
    return upper_bound / np.min(np.abs(depth))**2 < 0.98


def _area_sample_visible(photo, transform, out_size, mask):
    """Pixel-center-aligned 3x warp, area reduction and coverage normalization.

    Tiles bound the temporary 3x image size even when max_side is large.
    Coverage normalization prevents the black warp border from darkening
    partially covered pixels; the original nearest-neighbor valid mask stays
    authoritative for which output pixels exist.
    """
    width, height = map(int, out_size)
    pixels = np.zeros((height, width, 3), np.uint8)
    source_coverage = np.full(photo.shape[:2], 255, np.uint8)
    factor = 3
    center_aligned = np.array([[factor, 0, (factor - 1) / 2],
                               [0, factor, (factor - 1) / 2], [0, 0, 1.]], np.float64)
    high_transform = center_aligned @ transform
    tile_side = 512
    for y0 in range(0, height, tile_side):
        for x0 in range(0, width, tile_side):
            tile_width = min(tile_side, width - x0)
            tile_height = min(tile_side, height - y0)
            local = np.array([[1., 0., -factor*x0], [0., 1., -factor*y0],
                              [0., 0., 1.]]) @ high_transform
            high_size = (factor*tile_width, factor*tile_height)
            high_rgb = cv2.warpPerspective(photo, local, high_size, flags=cv2.INTER_LINEAR)
            high_coverage = cv2.warpPerspective(source_coverage, local, high_size,
                                                flags=cv2.INTER_LINEAR)
            area_rgb = cv2.resize(high_rgb.astype(np.float32), (tile_width, tile_height),
                                  interpolation=cv2.INTER_AREA)
            area_coverage = cv2.resize(high_coverage.astype(np.float32),
                                       (tile_width, tile_height), interpolation=cv2.INTER_AREA) / 255.
            normalized = area_rgb / np.maximum(area_coverage[..., None], 1e-5)
            pixels[y0:y0+tile_height, x0:x0+tile_width] = np.clip(
                np.rint(normalized), 0, 255).astype(np.uint8)
    pixels[mask == 0] = 0
    return pixels


def rectify_visible(photo, quad, ratio, *, max_side=2048, trusted_original=False):
    """One original-photo sample; missing input is transparent, never inpainted.

    Ratio describes the full reference plane, NOT the cropped output canvas.
    This is a research candidate renderer, not an automatic approval policy.
    """
    if photo.ndim != 3 or photo.shape[2] != 3 or photo.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB photo')
    h, w = photo.shape[:2]
    if trusted_original:
        return Rectified(photo.copy(), np.ones((h, w), np.uint8)*255, np.eye(3),
                         {'contract': CONTRACT, 'action': 'keep', 'pixel_exact': True})
    q = valid_quad(quad)
    if not np.isfinite(ratio) or not .05 <= ratio <= 20 or not 2 <= max_side <= 8192:
        raise ValueError('Invalid ratio or output size budget')
    poly = visible_polygon(q, w, h)
    rect = np.array([[0, 0], [ratio, 0], [ratio, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(q.astype(np.float32), rect)
    depths = np.c_[q, np.ones(4)] @ transform[2]
    if not np.isfinite(transform).all() or np.min(depths)*np.max(depths) <= 0:
        raise ValueError('Unstable projective plane')
    projected = project(poly, transform)
    low, high = projected.min(axis=0), projected.max(axis=0)
    span = high-low
    if np.min(span) <= 1e-8:
        raise ValueError('Collapsed visible output')
    scale = min(max_side-1, max(w, h)-1) / np.max(span)
    out_size = np.ceil(span*scale).astype(int)+1
    offset = np.array([[scale, 0, -low[0]*scale], [0, scale, -low[1]*scale], [0, 0, 1.]])
    final = offset @ transform
    # Map the visible polygon to output, and intersect it with sampled photo support.
    mask = np.zeros((out_size[1], out_size[0]), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(project(poly, final)).astype(np.int32), 255)
    support = cv2.warpPerspective(np.ones((h, w), np.uint8)*255, final, tuple(out_size),
                                  flags=cv2.INTER_NEAREST)
    mask = np.minimum(mask, support)
    minifying = _uniform_minification(final, poly)
    if minifying:
        pixels = _area_sample_visible(photo, final, out_size, mask)
    else:
        pixels = cv2.warpPerspective(photo, final, tuple(out_size), flags=cv2.INTER_LINEAR)
        pixels[mask == 0] = 0
    return Rectified(pixels, mask, final, {'contract': CONTRACT, 'action': 'rectify_visible',
                     'full_plane_ratio': float(ratio), 'output_size': out_size.tolist(),
                     'outside_corners': int(np.sum(((q < 0) | (q > [w-1, h-1])).any(axis=1))),
                     'resampling': 'area_3x' if minifying else 'bilinear',
                     'automatic_approval': False})
