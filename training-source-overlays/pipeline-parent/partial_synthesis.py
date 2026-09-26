"""Deterministic pre-source and post-photo crops with full-plane labels."""
from dataclasses import dataclass
import cv2
import numpy as np
from .ratio_synthesis import letterbox, project, pixel_corners, appearance
from .spatial_synthesis import sample as complete_sample
from .partial_geometry import visibility

FAMILIES = ('viewer_screen', 'coco_context', 'near_color', 'mixed_color')


@dataclass
class PartialView:
    photo: np.ndarray
    quad: np.ndarray
    film_mask: np.ndarray
    screen_mask: np.ndarray
    transform: np.ndarray
    ratio: float
    info: dict

    def input(self, size):
        pixels, lb = letterbox(self.photo, size)
        masks = [cv2.warpAffine(m.astype(np.float32), lb[:2], (size, size), flags=cv2.INTER_LINEAR)
                 for m in (self.film_mask, np.ones(self.photo.shape[:2], np.float32))]
        return {'pixels': pixels, 'quad': project(self.quad, lb).astype(np.float32),
                'log_ratio': np.float32(np.log(self.ratio)), 'mask': masks[0],
                'photo_valid': masks[1], 'inverse_letterbox': np.linalg.inv(lb).astype(np.float32)}


def pre_crop(source, rng, enabled):
    h, w = source.shape[:2]
    if not enabled: return source, [0, 0, w, h]
    # Each retained axis >=sqrt(.7), so their product remains >=.7.
    cw, ch = max(2, int(np.ceil(w*rng.uniform(np.sqrt(.7), 1)))), max(2, int(np.ceil(h*rng.uniform(np.sqrt(.7), 1))))
    x, y = int(rng.integers(w-cw+1)), int(rng.integers(h-ch+1))
    return source[y:y+ch, x:x+cw], [x, y, x+cw, y+ch]


def _background(source, rng, shape, family, background):
    h, w = shape
    if family == 'coco_context':
        if background is None: raise ValueError('COCO family requires an explicit split-matched background')
        bh, bw = background.shape[:2]
        side = float(rng.uniform(.65, 1))
        ch, cw = max(2, int(bh*side)), max(2, int(bw*side))
        y, x = int(rng.integers(bh-ch+1)), int(rng.integers(bw-cw+1))
        return cv2.resize(background[y:y+ch, x:x+cw], (w, h))
    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1])).astype(float)
    color = np.median(border, axis=0)
    if family == 'near_color':
        anchors = np.broadcast_to(color, (3, 3, 3)).copy() + rng.normal(0, 4, (3, 3, 3))
    elif family == 'mixed_color':
        mix = rng.uniform(0, 1)
        anchors = mix*rng.uniform(0, 255, (3, 3, 3)) + (1-mix)*color
    else: raise ValueError('Unknown background family')
    return np.clip(cv2.resize(anchors.astype(np.float32), (w, h)), 0, 255).astype(np.uint8)


def crop_to_missing(view, plane_size, missing, rng):
    """Crop AFTER projection. Solve missing area in the reference plane, not photo area."""
    h, w = view.photo.shape[:2]
    # One whole side or two adjacent sides; rotation also yields single-corner cuts.
    edges = ([int(rng.integers(4))] if rng.random() < .65 else
             [int(rng.integers(4))])
    if rng.random() < .35: edges.append((edges[0]+1) % 4)
    def bounds(t):
        b = [0, 0, w, h]
        for side in edges:
            if side == 0: b[0] = min(w-2, int(round(t*(w-1))))
            if side == 1: b[1] = min(h-2, int(round(t*(h-1))))
            if side == 2: b[2] = max(2, w-int(round(t*(w-1))))
            if side == 3: b[3] = max(2, h-int(round(t*(h-1))))
        return b
    best = None
    low, high = 0., .95
    for _ in range(30):
        t = (low+high)/2
        x, y, right, bottom = bounds(t)
        shift = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1.]])
        transform = shift @ view.transform
        q = project(view.quad, shift)
        try: vis = visibility(q, transform, plane_size, (right-x, bottom-y))
        except ValueError:
            high = t
            continue
        error = abs(vis['missing_fraction']-missing)
        if best is None or error < best[0]: best = (error, (x, y, right, bottom), transform, q, vis)
        if vis['missing_fraction'] < missing: low = t
        else: high = t
    if best is None or best[0] > .008:
        raise RuntimeError('Unable to satisfy requested missing-area stratum')
    _, (x, y, right, bottom), transform, q, vis = best
    info = {**view.info, **vis, 'post_photo_crop': [x, y, right, bottom]}
    return PartialView(view.photo[y:bottom, x:right].copy(), q,
                       view.film_mask[y:bottom, x:right].copy(), view.screen_mask[y:bottom, x:right].copy(),
                       transform, view.ratio, info)


def sample(source, source_index, view_index, *, seed=20260925, domain='train',
           kind='photo', integrity='complete', family='viewer_screen', source_crop=False,
           high_occupancy=True, background=None, mode='mild', level='light', pure_roll=False):
    if integrity not in ('complete', 'light', 'moderate') or family not in FAMILIES:
        raise ValueError('Unknown integrity/background stratum')
    if kind not in ('keep', 'photo') or domain not in ('train', 'dev'):
        raise ValueError('Unknown sample kind/domain')
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('Expected uint8 RGB source')
    if kind == 'keep' and (source_crop or integrity != 'complete'):
        raise ValueError('Keep must preserve the original source')
    rng = np.random.default_rng(np.random.SeedSequence([seed, source_index, view_index, 913, domain == 'dev']))
    plane, crop = pre_crop(source, rng, source_crop)
    ph, pw = plane.shape[:2]
    if kind == 'keep':
        quad = pixel_corners(pw, ph)
        info = visibility(quad, np.eye(3), (pw, ph), (pw, ph))
        return PartialView(plane.copy(), quad, np.ones((ph, pw), np.float32),
                           np.ones((ph, pw), np.float32), np.eye(3), pw/ph,
                           {**info, 'kind': 'keep', 'source_pre_crop': crop, 'integrity': 'complete',
                            'occupancy': 1., 'family': 'none', 'source_index': source_index, 'view_index': view_index})
    base = complete_sample(plane, source_index, view_index, seed=seed, domain=domain,
                           kind='photo', mode=mode, pure_roll=pure_roll,
                           occupancy_bin=1 if high_occupancy else 0, level='clear')
    transform = np.asarray(base.info['source_to_photo'])
    photo = base.clean
    screen = base.screen_mask
    if family != 'viewer_screen':
        exterior = _background(plane, rng, photo.shape[:2], family, background)
        h, w = photo.shape[:2]
        warped = cv2.warpPerspective(plane, transform, (w, h), flags=cv2.INTER_LINEAR)
        photo = np.clip(warped.astype(np.float32) + exterior*(1-base.film_mask[..., None]), 0, 255).astype(np.uint8)
        screen = np.zeros((h, w), np.float32)
    photo, app = appearance(photo, screen, rng, level)
    info = {**base.info, **app, 'family': family, 'integrity': integrity, 'source_pre_crop': crop,
            'source_index': source_index, 'view_index': view_index, 'seed': seed,
            'plane_size': [pw, ph], 'post_photo_crop': [0, 0, photo.shape[1], photo.shape[0]],
            'requested_high_occupancy': high_occupancy}
    info.update(visibility(base.quad, transform, (pw, ph), (photo.shape[1], photo.shape[0])))
    view = PartialView(photo, base.quad, base.film_mask, screen, transform, pw/ph, info)
    if integrity != 'complete':
        missing = float(rng.uniform(.02, .09) if integrity == 'light' else rng.uniform(.12, .28))
        view = crop_to_missing(view, (pw, ph), missing, rng)
    view.info['occupancy'] = float(view.film_mask.mean())
    minimum = .5 if high_occupancy else .3
    if view.info['occupancy'] < minimum:
        raise RuntimeError('Visible film occupancy below requested quota; sample not silently downgraded')
    view.info['source_to_photo'] = view.transform.tolist()
    view.info['quad_photo'] = view.quad.tolist()
    view.info['photo_size'] = [view.photo.shape[1], view.photo.shape[0]]
    return view
