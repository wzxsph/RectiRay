"""Deterministic physical-plane views for the unified geometry experiment.

All original pixels are read-only. The projected *film* area, rather than its
axis-aligned box or surrounding screen, determines the occupancy stratum.
"""
from __future__ import annotations

import cv2
import numpy as np

from .ratio_synthesis import View, appearance, pixel_corners, project

SEED = 20260922
DOMAINS = {'train': 3101, 'dev': 3102, 'test': 3103}
KINDS = ('digital', 'appearance', 'rectified', 'photo')
MODES = ('front', 'mild', 'medium', 'hard')
LEVELS = ('clear', 'light', 'medium', 'strong')
TILT_BOUNDS = {'front': (0., 0.), 'mild': (0., 8.), 'medium': (8., 15.), 'hard': (15., 25.)}
ROLL_BOUNDS = {'front': 12., 'mild': 6., 'medium': 12., 'hard': 20.}
OCCUPANCY_BOUNDS = ((.30, .50), (.50, .75), (.75, .90))


def _stream(source_index, view_index, domain, channel, seed):
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed), DOMAINS[domain], int(source_index), int(view_index), int(channel)]))


def quad_area(quad):
    q = np.asarray(quad, np.float64)
    return float(.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) -
                          np.dot(q[:, 1], np.roll(q[:, 0], -1))))


def _rotation(tilt, azimuth, roll):
    """Rotate about an in-plane axis, then about the camera optical axis.

    This constructs a total normal tilt directly; independent yaw/pitch bounds
    would incorrectly permit an accumulated tilt above the specified limit.
    """
    t, a, r = np.deg2rad([tilt, azimuth, roll])
    axis = np.array([np.cos(a), np.sin(a), 0.])
    rotation, _ = cv2.Rodrigues(axis * t)
    cr, sr = np.cos(r), np.sin(r)
    return np.array([[cr, -sr, 0.], [sr, cr, 0.], [0., 0., 1.]]) @ rotation


def _camera(width0, height0, rng, mode, occupancy_bin, pure_roll):
    """Find a physical pose/camera crop satisfying the actual area quota.

    Canvas aspect follows the projected film; focal length and principal point
    implement an isotropic magnification and translation. No source-axis
    stretching is used. High roll and high occupancy can be incompatible for
    elongated sources, so poses are resampled inside the approved bounds.
    """
    scale0 = float(max(width0 - 1, height0 - 1))
    plane = np.array([[1 / scale0, 0., -(width0 - 1) / (2 * scale0)],
                      [0., 1 / scale0, -(height0 - 1) / (2 * scale0)],
                      [0., 0., 1.]])
    original = pixel_corners(width0, height0)
    lower, upper = OCCUPANCY_BOUNDS[occupancy_bin]
    for attempt in range(1, 101):
        lo, hi = TILT_BOUNDS[mode]
        tilt = 0. if mode == 'front' else float(rng.uniform(np.nextafter(lo, hi), hi))
        azimuth = float(rng.uniform(-180., 180.))
        if mode == 'front':
            roll = float(rng.choice((-1., 1.)) * rng.uniform(1., 12.)) if pure_roll else 0.
        else:
            roll = float(rng.uniform(-ROLL_BOUNDS[mode], ROLL_BOUNDS[mode]))
        rotation = _rotation(tilt, azimuth, roll)
        distance = float(rng.uniform(2.4, 5.2))
        extrinsic = np.column_stack((rotation[:, 0], rotation[:, 1], [0., 0., distance]))
        unit_h = extrinsic @ plane
        q = project(original, unit_h)
        span = np.ptp(q, axis=0)
        area = quad_area(q)
        aspect = float(span[0] / span[1] * np.exp(rng.uniform(-.055, .055)))
        width, height = ((768, max(2, round(768 / aspect))) if aspect >= 1 else
                         (max(2, round(768 * aspect)), 768))
        # One-pixel safety margin is in addition to complete pixel-center corners.
        available = np.array([width - 3., height - 3.])
        if (available <= 0).any():
            continue
        max_focal = float(np.min(available / span))
        feasible = area * max_focal ** 2 / (width * height)
        ceiling = min(upper, feasible * .998)
        # Strictly >50/>75 for the upper strata, with a numerical guard.
        floor = lower + (1e-6 if occupancy_bin else 0.)
        if ceiling <= floor + 1e-5:
            continue
        target = float(rng.uniform(floor, ceiling))
        focal = float(np.sqrt(target * width * height / area))
        scaled = q * focal
        room = np.array([width - 3., height - 3.]) - np.ptp(scaled, axis=0)
        offset = 1. + rng.random(2) * room - np.min(scaled, axis=0)
        intrinsic = np.array([[focal, 0., offset[0]], [0., focal, offset[1]], [0., 0., 1.]])
        camera = intrinsic @ extrinsic
        transform = camera @ plane
        quad = project(original, transform)
        actual = quad_area(quad) / (width * height)
        if (not cv2.isContourConvex(quad.astype(np.float32)) or
                cv2.contourArea(quad.astype(np.float32), oriented=True) <= 0 or
                (quad < 0).any() or (quad > [width - 1, height - 1]).any() or
                not (floor - 1e-12 <= actual <= upper + 1e-12)):
            raise RuntimeError('Physical camera failed its final geometry audit')
        return transform, quad, camera, plane, {
            'tilt': tilt, 'roll': roll, 'tilt_azimuth': azimuth,
            'camera_distance': distance, 'focal_length': focal,
            'camera_attempts': attempt, 'common_size': [width, height],
            'photo_aspect': width / height, 'occupancy': actual,
            'projected_bbox_fill': quad_area(quad) / np.prod(np.ptp(quad, axis=0)),
        }
    raise RuntimeError('No complete physical pose satisfies the requested occupancy after 100 attempts')


def _render_photo(source, source_index, view_index, *, mode, occupancy_bin,
                  level, pure_roll, domain, seed):
    height0, width0 = source.shape[:2]
    geometry = _stream(source_index, view_index, domain, 0, seed)
    background = _stream(source_index, view_index, domain, 1, seed)
    exterior = _stream(source_index, view_index, domain, 2, seed)
    transform, quad, camera, plane, info = _camera(
        width0, height0, geometry, mode, occupancy_bin, pure_roll)
    width, height = info['common_size']
    film_plane = project(pixel_corners(width0, height0), plane)
    left, top = film_plane.min(axis=0)
    right, bottom = film_plane.max(axis=0)
    film_width, film_height = right - left, bottom - top
    margins = background.uniform(.02, .08, 4)
    screen_left, screen_top = left - margins[0] * film_width, top - margins[1] * film_height
    screen_right, screen_bottom = right + margins[2] * film_width, bottom + margins[3] * film_height
    screen = np.array([[screen_left, screen_top], [screen_right, screen_top],
                       [screen_right, screen_bottom], [screen_left, screen_bottom]])
    projected_screen = project(screen, camera)
    small = background.integers(15, 180, (5, 5, 3), dtype=np.uint8)
    canvas = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    screen_mask = np.zeros((height, width), np.float32)
    cv2.fillConvexPoly(screen_mask, np.rint(projected_screen).astype(np.int32), 1.)
    screen_color = int(background.choice([18, 38, 70, 160, 220]))
    cv2.fillConvexPoly(canvas, np.rint(projected_screen).astype(np.int32), (screen_color,) * 3)
    toolbar = project(np.array([[screen_left, screen_top], [screen_right, screen_top],
                               [screen_right, top], [screen_left, top]]), camera)
    cv2.fillConvexPoly(canvas, np.rint(toolbar).astype(np.int32), (48, 57, 68))
    for fraction in np.linspace(.08, .88, 7):
        x = screen_left + fraction * (screen_right - screen_left)
        button = project(np.array([[x, screen_top + .2 * (top - screen_top)],
                                   [x + .03 * film_width, screen_top + .2 * (top - screen_top)],
                                   [x + .03 * film_width, screen_top + .75 * (top - screen_top)],
                                   [x, screen_top + .75 * (top - screen_top)]]), camera)
        cv2.fillConvexPoly(canvas, np.rint(button).astype(np.int32), (160, 170, 180))
    warped = cv2.warpPerspective(source, transform, (width, height), flags=cv2.INTER_LINEAR)
    film_mask = cv2.warpPerspective(np.ones((height0, width0), np.float32), transform,
                                    (width, height), flags=cv2.INTER_LINEAR)
    clean = np.clip(warped.astype(np.float32) + canvas * (1 - film_mask[..., None]), 0, 255).astype(np.uint8)
    if level == 'clear':
        noisy, appearance_info = clean, {'moire': False, 'camera': 'none'}
    else:
        noisy, appearance_info = appearance(clean, screen_mask, exterior, level)
    info.update({'source_size': [width0, height0], 'source_to_photo': transform.tolist(),
                 'kind': 'photo', 'mode': mode, 'occupancy_bin': int(occupancy_bin),
                 'pure_roll': bool(pure_roll), 'domain': domain,
                 'requested_appearance': level, 'appearance': level, **appearance_info})
    return View(clean, noisy, quad, screen_mask, float(np.log(width0 / height0)), info)


def sample(source, source_index, view_index, *, kind, mode='front', occupancy_bin=1,
           level='clear', pure_roll=False, domain='train', seed=SEED):
    """Return the existing View interface without changing ``source``.

    Clear/noisy schedule pairs share source/view/domain identifiers. Appearance
    does not consume the geometry/background random streams. ``source_to_photo``
    always maps the supplied source's pixel centers to the returned image.
    """
    if kind not in KINDS or mode not in MODES or level not in LEVELS or domain not in DOMAINS:
        raise ValueError('Unknown unified synthesis stratum')
    if occupancy_bin not in range(3) or (pure_roll and mode != 'front'):
        raise ValueError('Invalid occupancy or pure-rotation stratum')
    if int(source_index) < 0 or int(view_index) < 0:
        raise ValueError('Source and view ordinals must be nonnegative')
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8 or min(source.shape[:2]) < 2:
        raise ValueError('Expected at least 2x2 uint8 RGB source')
    if kind == 'photo':
        if mode == 'hard' and level not in ('clear', 'light'):
            raise ValueError('Hard geometry permits only clear/light appearance')
        return _render_photo(source, source_index, view_index, mode=mode,
                             occupancy_bin=occupancy_bin, level=level, pure_roll=pure_roll,
                             domain=domain, seed=seed)
    height, width = source.shape[:2]
    info = {'kind': kind, 'mode': 'identity', 'occupancy_bin': None, 'domain': domain,
            'source_size': [width, height], 'source_to_photo': np.eye(3).tolist(),
            'common_size': [width, height], 'occupancy': quad_area(pixel_corners(width, height)) / (width * height),
            'tilt': 0., 'roll': 0., 'pure_roll': False, 'appearance': level,
            'requested_appearance': level, 'camera_attempts': 0}
    screen_mask = np.ones((height, width), np.float32)
    if kind == 'rectified':
        projected = _render_photo(source, source_index, view_index, mode='mild', occupancy_bin=1,
                                  level=level, pure_roll=False, domain=domain, seed=seed)
        inverse = np.linalg.inv(np.asarray(projected.info['source_to_photo']))
        clean = cv2.warpPerspective(projected.clean, inverse, (width, height), flags=cv2.INTER_LINEAR)
        noisy = (clean if level == 'clear' else
                 cv2.warpPerspective(projected.noisy, inverse, (width, height), flags=cv2.INTER_LINEAR))
        info.update({'rectified_from_tilt': projected.info['tilt'],
                     'rectified_from_roll': projected.info['roll'],
                     'moire': projected.info['moire'], 'camera': projected.info['camera']})
    elif kind == 'digital':
        if level != 'clear':
            raise ValueError('Digital-original stratum must remain clear')
        clean = source.copy()
        noisy = clean
        info.update({'moire': False, 'camera': 'none'})
    else:
        clean = source.copy()
        noisy, appearance_info = appearance(clean, screen_mask,
                                             _stream(source_index, view_index, domain, 2, seed), level)
        info.update(appearance_info)
    return View(clean, noisy, pixel_corners(width, height), screen_mask, float(np.log(width / height)), info)
