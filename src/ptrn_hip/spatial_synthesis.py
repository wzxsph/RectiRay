"""Versioned spatial-training views with distinct film and viewer masks.

Only synthetic exteriors and UI are drawn here. Source pixels are read only.
The requested canvas aspect and roll are sampled independently of source shape
and perspective mode; infeasible combinations are resampled, counted, and never
made feasible by stretching or shrinking the film below its occupancy stratum.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .ratio_synthesis import appearance, letterbox, normalization, pixel_corners, project
from .unified_synthesis import (DOMAINS, KINDS, LEVELS, MODES, OCCUPANCY_BOUNDS,
                                TILT_BOUNDS, _rotation, quad_area)

SEED = 20260923
TOPOLOGIES = ('dark_viewer', 'light_page', 'dual_panel', 'physical_screen')
CANVAS_ASPECTS = (.45, .75, 1., 1.33, 1.78, 2.5)
MAX_ATTEMPTS = 400


@dataclass
class SpatialView:
    clean: np.ndarray
    noisy: np.ndarray
    quad: np.ndarray
    film_mask: np.ndarray
    screen_mask: np.ndarray
    log_ratio: float
    info: dict

    def input(self, size, noisy=True):
        pixels, lb = letterbox(self.noisy if noisy else self.clean, size)
        q = project(self.quad, lb)
        h = cv2.getPerspectiveTransform(
            np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32),
            project(q, normalization(size, size)).astype(np.float32))
        return pixels, np.r_[h.reshape(-1)[:8], self.log_ratio].astype(np.float32), q.astype(np.float32)


def _stream(source_index, view_index, domain, channel, seed):
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), DOMAINS[domain], int(source_index), int(view_index), int(channel)]))


def _roll(rng, band):
    # The roll band is fixed for one view; pose/canvas retries cannot silently
    # replace a requested hard rotation with an easy small one.
    lo, hi = ((0., 10.), (10., 30.), (30., 60.))[band]
    value = float(rng.uniform(lo, hi))
    return float(rng.choice((-1., 1.)) * value)


def _canvas(rng):
    anchor = float(rng.choice(CANVAS_ASPECTS))
    aspect = anchor * float(np.exp(rng.uniform(-.10, .10)))
    width, height = ((768, round(768 / aspect)) if aspect >= 1 else
                     (round(768 * aspect), 768))
    return max(2, width), max(2, height), anchor


def _camera(width0, height0, source_index, view_index, domain, seed, mode,
            occupancy_bin, pure_roll):
    pose_rng = _stream(source_index, view_index, domain, 0, seed)
    canvas_rng = _stream(source_index, view_index, domain, 3, seed)
    roll_rng = _stream(source_index, view_index, domain, 4, seed)
    source_corners = pixel_corners(width0, height0)
    scale0 = float(max(width0 - 1, height0 - 1))
    plane = np.array([[1 / scale0, 0., -(width0 - 1) / (2 * scale0)],
                      [0., 1 / scale0, -(height0 - 1) / (2 * scale0)],
                      [0., 0., 1.]])
    lower, upper = OCCUPANCY_BOUNDS[occupancy_bin]
    # Strong roll and >75% film occupancy are geometrically incompatible for
    # narrow/wide source canvases. Allocate that stratum to 30-50% occupancy,
    # while keeping the overall intended ~70/25/5 roll mixture. This is a
    # planned joint distribution, not a retry-time downgrade.
    roll_probabilities = ((.32, .355, .325), (.60, .40, 0.), (1., 0., 0.))
    requested_roll_band = (None if mode == 'front' and not pure_roll else
                           int(roll_rng.choice(3, p=roll_probabilities[occupancy_bin])))
    rejected = {'canvas_or_pose': 0, 'occupancy': 0, 'screen_or_boundary': 0}
    proposed_roll_bands = {'0': 0, '1': 0, '2': 0, 'zero': 0}
    proposed_canvas_anchors = {str(anchor): 0 for anchor in CANVAS_ASPECTS}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        low_tilt, high_tilt = TILT_BOUNDS[mode]
        tilt = (0. if mode == 'front' else
                float(pose_rng.uniform(np.nextafter(low_tilt, high_tilt), high_tilt)))
        azimuth = float(pose_rng.uniform(-180., 180.))
        roll = 0. if requested_roll_band is None else _roll(roll_rng, requested_roll_band)
        proposed_roll_bands['zero' if requested_roll_band is None else str(requested_roll_band)] += 1
        width, height, anchor = _canvas(canvas_rng)
        proposed_canvas_anchors[str(anchor)] += 1
        rotation = _rotation(tilt, azimuth, roll)
        distance = float(pose_rng.uniform(2.4, 5.2))
        extrinsic = np.column_stack((rotation[:, 0], rotation[:, 1], [0., 0., distance]))
        normalized_h = extrinsic @ plane
        normalized_quad = project(source_corners, normalized_h)
        spans = np.ptp(normalized_quad, axis=0)
        area = quad_area(normalized_quad)
        available = np.array([width - 3., height - 3.])
        if (spans <= 0).any() or area <= 0 or (available <= 0).any():
            rejected['canvas_or_pose'] += 1
            continue
        max_focal = float(np.min(available / spans))
        feasible = area * max_focal ** 2 / (width * height)
        floor = lower + (1e-6 if occupancy_bin else 0.)
        ceiling = min(upper, feasible * .998)
        if ceiling <= floor + 1e-5:
            rejected['occupancy'] += 1
            continue
        target = float(pose_rng.uniform(floor, ceiling))
        focal = float(np.sqrt(target * width * height / area))
        scaled = normalized_quad * focal
        room = available - np.ptp(scaled, axis=0)
        if (room < 0).any():
            rejected['screen_or_boundary'] += 1
            continue
        offset = 1. + pose_rng.random(2) * room - np.min(scaled, axis=0)
        intrinsic = np.array([[focal, 0., offset[0]], [0., focal, offset[1]], [0., 0., 1.]])
        camera = intrinsic @ extrinsic
        transform = camera @ plane
        quad = project(source_corners, transform)
        occupancy = quad_area(quad) / (width * height)
        if (not cv2.isContourConvex(quad.astype(np.float32)) or
                cv2.contourArea(quad.astype(np.float32), oriented=True) <= 0 or
                (quad < 0).any() or (quad > [width - 1, height - 1]).any() or
                not (floor - 1e-12 <= occupancy <= upper + 1e-12)):
            raise RuntimeError('Spatial physical camera failed final film geometry audit')
        info = {'tilt': tilt, 'roll': roll,
                'roll_band': 'zero' if requested_roll_band is None else requested_roll_band,
                'requested_roll_band': 'zero' if requested_roll_band is None else requested_roll_band,
                'tilt_azimuth': azimuth, 'camera_distance': distance,
                'focal_length': focal, 'camera_attempts': attempt,
                'camera_rejections': rejected, 'proposed_roll_bands': proposed_roll_bands,
                'proposed_canvas_anchors': proposed_canvas_anchors,
                'requested_canvas_anchor': anchor,
                'common_size': [width, height], 'photo_size': [width, height],
                'photo_aspect': width / height, 'occupancy': occupancy,
                'projected_bbox_fill': occupancy * width * height / np.prod(np.ptp(quad, axis=0))}
        return transform, quad, camera, plane, info
    raise RuntimeError('No complete physical pose meets the requested film occupancy '
                       f'after {MAX_ATTEMPTS} attempts; no quota was downgraded')


def _poly(canvas, points, color):
    cv2.fillConvexPoly(canvas, np.rint(points).astype(np.int32), color)


def _photo(source, source_index, view_index, *, mode, occupancy_bin,
           level, pure_roll, domain, seed):
    height0, width0 = source.shape[:2]
    transform, quad, camera, plane, info = _camera(
        width0, height0, source_index, view_index, domain, seed,
        mode, occupancy_bin, pure_roll)
    width, height = info['photo_size']
    background = _stream(source_index, view_index, domain, 1, seed)
    exterior = _stream(source_index, view_index, domain, 2, seed)
    topology = TOPOLOGIES[int(background.integers(len(TOPOLOGIES)))]
    small = background.integers(35, 185, (5, 5, 3), dtype=np.uint8)
    canvas = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    film_plane = project(pixel_corners(width0, height0), plane)
    left, top = film_plane.min(axis=0)
    right, bottom = film_plane.max(axis=0)
    fw, fh = right - left, bottom - top
    margin = background.uniform(.035, .12, 4)
    sl, st = left - margin[0] * fw, top - margin[1] * fh
    sr, sb = right + margin[2] * fw, bottom + margin[3] * fh
    screen_plane = np.array([[sl, st], [sr, st], [sr, sb], [sl, sb]])
    screen_quad = project(screen_plane, camera)
    screen_mask = np.zeros((height, width), np.float32)
    _poly(screen_mask, screen_quad, 1.)
    if topology == 'light_page':
        canvas[:] = np.clip(canvas.astype(np.int16) + 50, 0, 245).astype(np.uint8)
        viewer_color = (30, 34, 40)
    elif topology == 'dual_panel':
        viewer_color = (22, 25, 30)
        sidebar = project(np.array([[sr, st], [sr + .24 * fw, st],
                                    [sr + .24 * fw, sb], [sr, sb]]), camera)
        _poly(canvas, sidebar, (205, 210, 215))
    elif topology == 'physical_screen':
        viewer_color = (38, 42, 48)
        bezel = project(np.array([[sl - .10 * fw, st - .10 * fh],
                                  [sr + .10 * fw, st - .10 * fh],
                                  [sr + .10 * fw, sb + .10 * fh],
                                  [sl - .10 * fw, sb + .10 * fh]]), camera)
        _poly(canvas, bezel, (9, 11, 15))
    else:
        viewer_color = (16, 19, 24)
    _poly(canvas, screen_quad, viewer_color)
    toolbar = project(np.array([[sl, st], [sr, st], [sr, top], [sl, top]]), camera)
    _poly(canvas, toolbar, (62, 67, 74) if topology != 'light_page' else (205, 208, 213))
    for fraction in np.linspace(.08, .88, 7):
        x = sl + fraction * (sr - sl)
        button = project(np.array([[x, st + .18 * (top - st)],
                                   [x + .025 * fw, st + .18 * (top - st)],
                                   [x + .025 * fw, st + .72 * (top - st)],
                                   [x, st + .72 * (top - st)]]), camera)
        _poly(canvas, button, (170, 178, 185))
    warped = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR)
    film_mask = cv2.warpPerspective(np.ones((height0, width0), np.float32), transform,
                                    (width, height), flags=cv2.INTER_LINEAR)
    clean = np.clip(warped.astype(np.float32) + canvas * (1 - film_mask[..., None]),
                    0, 255).astype(np.uint8)
    noisy, app_info = appearance(clean, screen_mask, exterior, level)
    info.update({'kind': 'photo', 'mode': mode, 'occupancy_bin': int(occupancy_bin),
                 'pure_roll': bool(pure_roll), 'domain': domain, 'source_size': [width0, height0],
                 'source_to_photo': transform.tolist(), 'quad_photo': quad.tolist(),
                 'topology': topology,
                 'requested_appearance': level, 'appearance': level, **app_info})
    return SpatialView(clean, noisy, quad, film_mask, screen_mask,
                       float(np.log(width0 / height0)), info)


def sample(source, source_index, view_index, *, kind, mode='front', occupancy_bin=1,
           level='clear', pure_roll=False, domain='train', seed=SEED):
    """Deterministic replacement render for the old schedule row signature."""
    if kind not in KINDS or mode not in MODES or level not in LEVELS or domain not in DOMAINS:
        raise ValueError('Unknown spatial synthesis stratum')
    if occupancy_bin not in range(3) or (pure_roll and mode != 'front'):
        raise ValueError('Invalid occupancy or pure-roll assignment')
    if int(source_index) < 0 or int(view_index) < 0:
        raise ValueError('Source and view ordinals must be nonnegative')
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8 or min(source.shape[:2]) < 2:
        raise ValueError('Expected at least 2x2 uint8 RGB source')
    if kind == 'photo':
        if mode == 'hard' and level not in ('clear', 'light'):
            raise ValueError('Hard geometry permits only clear/light appearance')
        return _photo(source, source_index, view_index, mode=mode, occupancy_bin=occupancy_bin,
                      level=level, pure_roll=pure_roll, domain=domain, seed=seed)
    from .unified_synthesis import sample as old_sample
    old = old_sample(source, source_index, view_index, kind=kind, mode=mode,
                     occupancy_bin=occupancy_bin, level=level, pure_roll=pure_roll,
                     domain=domain, seed=seed)
    height, width = source.shape[:2]
    film_mask = np.ones((height, width), np.float32)
    screen_mask = np.zeros((height, width), np.float32)
    info = dict(old.info)
    info.update({'photo_size': [width, height], 'topology': 'source_canvas',
                 'quad_photo': old.quad.tolist(),
                 'camera_rejections': {'canvas_or_pose': 0, 'occupancy': 0, 'screen_or_boundary': 0}})
    return SpatialView(old.clean, old.noisy, old.quad, film_mask, screen_mask,
                       old.log_ratio, info)
