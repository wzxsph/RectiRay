"""Deterministic pre-source and post-photo crops with full-plane labels."""
from dataclasses import dataclass
import cv2
import numpy as np
from .ratio_synthesis import letterbox, project, pixel_corners, appearance
from .spatial_synthesis import sample as complete_sample
from .partial_geometry import visibility

FAMILIES = ('viewer_screen', 'coco_context', 'near_color', 'mixed_color')
STEP2000_SCENE = 'ptrn-step2000-unified-v1'


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


def pre_crop(source, rng, enabled, area_loss=None):
    h, w = source.shape[:2]
    if not enabled: return source, [0, 0, w, h]
    if area_loss is not None:
        low, high = area_loss
        if not 0 <= low < high < 1: raise ValueError('Invalid pre-crop area range')
        keep = 1.-float(rng.uniform(low, high))
        # Randomly divide the removed area between axes without stretching.
        fraction = float(rng.random())
        cw = min(w, max(2, int(round(w*keep**fraction))))
        ch = min(h, max(2, int(round(h*keep**(1-fraction)))))
        x, y = int(rng.integers(w-cw+1)), int(rng.integers(h-ch+1))
        return source[y:y+ch, x:x+cw], [x, y, x+cw, y+ch]
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


def _frame_step2000(base, rng, occupancy_range):
    """Reframe old physical scene to the two new end bins without losing film.

    The 30--90% bins use the original physical camera untouched. Near-full
    photos crop *outside* all four corners; distant photos add only exterior
    canvas. Resizing distant frames transforms pixels, masks and labels alike.
    """
    photo, mask, screen = base.clean, base.film_mask, base.screen_mask
    h, w = photo.shape[:2]
    quad = np.asarray(base.quad, np.float64)
    transform = np.asarray(base.info['source_to_photo'], np.float64)
    lo, hi = occupancy_range
    if lo >= .90:
        target = float(rng.uniform(.915, .965))
        mins, maxs = quad.min(axis=0), quad.max(axis=0)
        box = maxs-mins
        quad_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
        scale = np.sqrt((quad_area/target)/float(np.prod(box)))
        nw, nh = int(np.ceil(box[0]*scale+2)), int(np.ceil(box[1]*scale+2))
        if nw >= w or nh >= h:
            raise RuntimeError('Near-full framing is geometrically infeasible')
        x = int(np.clip(round((mins[0]+maxs[0]-nw)/2), 0, w-nw))
        y = int(np.clip(round((mins[1]+maxs[1]-nh)/2), 0, h-nh))
        if mins[0] < x or maxs[0] > x+nw-1 or mins[1] < y or maxs[1] > y+nh-1:
            raise RuntimeError('Near-full framing would remove source content')
        photo = photo[y:y+nh, x:x+nw].copy()
        mask = mask[y:y+nh, x:x+nw].copy()
        screen = screen[y:y+nh, x:x+nw].copy()
        frame = np.array([[1., 0., -x], [0., 1., -y], [0., 0., 1.]])
    elif hi <= .30:
        target = float(rng.uniform(.12, .27))
        old_occupancy = float(mask.mean())
        scale = np.sqrt(old_occupancy/target)
        nw, nh = int(np.ceil(w*scale)), int(np.ceil(h*scale))
        left, top = (nw-w)//2, (nh-h)//2
        border = np.concatenate((photo[0], photo[-1], photo[:, 0], photo[:, -1]))
        color = np.median(border, axis=0).astype(np.uint8)
        expanded = np.empty((nh, nw, 3), np.uint8)
        expanded[:] = color
        expanded[top:top+h, left:left+w] = photo
        photo = expanded
        mask = cv2.copyMakeBorder(mask, top, nh-h-top, left, nw-w-left,
                                  cv2.BORDER_CONSTANT, value=0)
        screen = cv2.copyMakeBorder(screen, top, nh-h-top, left, nw-w-left,
                                    cv2.BORDER_CONSTANT, value=0)
        frame = np.array([[1., 0., left], [0., 1., top], [0., 0., 1.]])
        if max(nw, nh) > 768:
            s = 768/max(nw, nh)
            resized = (max(2, int(round(nw*s))), max(2, int(round(nh*s))))
            photo = cv2.resize(photo, resized, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resized, interpolation=cv2.INTER_LINEAR)
            screen = cv2.resize(screen, resized, interpolation=cv2.INTER_LINEAR)
            frame = np.diag([resized[0]/nw, resized[1]/nh, 1.]) @ frame
    else:
        frame = np.eye(3)
    q = project(quad, frame)
    info = dict(base.info)
    info.update({'source_to_photo': (frame @ transform).tolist(),
                 'quad_photo': q.tolist(), 'photo_size': [photo.shape[1], photo.shape[0]],
                 'occupancy': float(mask.mean()),
                 'frame_transform': frame.tolist()})
    if not lo <= info['occupancy'] < hi:
        raise RuntimeError('Requested occupancy stratum infeasible after framing')
    return photo, mask, screen, frame @ transform, q, info


def _soft_edge(photo, mask, width_px224):
    """Only soften the observed film/background seam; preserve the interior."""
    if width_px224 <= 0:
        return photo.copy(), 0.
    native_width = float(width_px224*max(photo.shape[:2])/224.)
    interior = (mask >= .5).astype(np.uint8)
    if interior.min() == interior.max():
        return photo.copy(), native_width
    inside = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1-interior, cv2.DIST_L2, 3)
    distance = inside+outside
    ring = np.clip((native_width*1.5-distance)/max(native_width, 1e-3), 0, 1)
    sigma = max(.45, native_width*.65)
    blurred = cv2.GaussianBlur(photo, (0, 0), sigma)
    out = photo.astype(np.float32)*(1-ring[..., None])+blurred.astype(np.float32)*ring[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), native_width


def _sample_step2000(source, source_index, view_index, *, seed, domain, kind,
                     integrity, family, source_crop, background, mode, pure_roll,
                     occupancy_range, occupancy_bin_v2, missing_range, missing_target,
                     source_crop_area_loss, capture_material, capture_severity,
                     capture_seed, edge_blend_px224):
    if kind != 'photo' or domain != 'train':
        raise ValueError('New scene version is train-only and photo-only')
    if integrity not in ('complete', 'light', 'moderate', 'severe') or family not in FAMILIES:
        raise ValueError('Invalid new scene stratum')
    if occupancy_range not in ((.90, .98), (.75, .90), (.50, .75), (.30, .50), (.10, .30)):
        raise ValueError('Invalid new occupancy bin')
    if capture_material not in ('screen', 'film') or capture_severity not in ('clear', 'light', 'medium', 'strong'):
        raise ValueError('Invalid capture appearance')
    if integrity == 'complete' and (missing_range is not None or missing_target is not None):
        raise ValueError('Complete view cannot request missing area')
    if integrity != 'complete' and (missing_range is None or missing_target is None or
                                    not missing_range[0] < missing_target < missing_range[1]):
        raise ValueError('Invalid missing-area request')
    rng = np.random.default_rng(np.random.SeedSequence([seed, source_index, view_index, 913]))
    plane, crop = pre_crop(source, rng, source_crop, source_crop_area_loss)
    ph, pw = plane.shape[:2]
    source_area_loss = 1.-(pw*ph)/(source.shape[1]*source.shape[0])
    if source_crop and not source_crop_area_loss[0] <= source_area_loss <= source_crop_area_loss[1]:
        raise RuntimeError('Actual source pre-crop area outside requested stratum')
    bin_old = (2 if occupancy_range[0] >= .75 else 1 if occupancy_range[0] >= .50 else 0)
    base = complete_sample(plane, source_index, view_index, seed=seed,
                           domain=domain, kind='photo', mode=mode, pure_roll=pure_roll,
                           occupancy_bin=bin_old, level='clear')
    photo, mask, screen, transform, q, base_info = _frame_step2000(base, rng, occupancy_range)
    if family != 'viewer_screen':
        exterior = _background(plane, rng, photo.shape[:2], family, background)
        h, w = photo.shape[:2]
        warped = cv2.warpPerspective(plane, transform, (w, h), flags=cv2.INTER_LINEAR)
        photo = np.clip(warped.astype(np.float32) +
                        exterior.astype(np.float32)*(1-mask[..., None]), 0, 255).astype(np.uint8)
        screen = np.zeros((h, w), np.float32)
    info = {**base_info, 'kind': 'photo', 'family': family, 'integrity': integrity,
            'scene_version': STEP2000_SCENE, 'source_pre_crop': crop,
            'source_pre_crop_area_loss': source_area_loss,
            'source_index': source_index, 'view_index': view_index, 'seed': seed,
            'plane_size': [pw, ph], 'post_photo_crop': [0, 0, photo.shape[1], photo.shape[0]],
            'requested_occupancy_range': list(occupancy_range),
            'occupancy_bin_v2': occupancy_bin_v2, 'requested_missing_range': missing_range,
            'requested_missing_target': missing_target, 'edge_blend_px224': edge_blend_px224,
            'capture_material': capture_material, 'capture_severity': capture_severity,
            'capture_seed': capture_seed}
    info.update(visibility(q, transform, (pw, ph), (photo.shape[1], photo.shape[0])))
    view = PartialView(photo, q, mask, screen, transform, pw/ph, info)
    if integrity != 'complete':
        view = crop_to_missing(view, (pw, ph), missing_target, rng)
        actual = float(view.info['missing_fraction'])
        if not missing_range[0] < actual <= missing_range[1]:
            raise RuntimeError('Actual missing fraction outside requested stratum')
    softened, native_width = _soft_edge(view.photo, view.film_mask, edge_blend_px224)
    view.photo = softened
    from .capture_synthesis import recipe as capture_recipe, render as capture_render
    surface = (view.screen_mask if capture_material == 'screen' and family == 'viewer_screen'
               else view.film_mask)
    capture = capture_recipe(capture_material, capture_severity, capture_seed)
    view.photo, capture_info = capture_render(
        view.photo, surface, view.transform, (pw, ph), capture)
    view.info.update({'capture': capture_info, 'edge_blend_native_px': native_width,
                      'occupancy': float(view.film_mask.mean()),
                      'source_to_photo': view.transform.tolist(),
                      'quad_photo': view.quad.tolist(),
                      'photo_size': [view.photo.shape[1], view.photo.shape[0]]})
    lo, hi = occupancy_range
    if not lo <= view.info['occupancy'] < hi:
        raise RuntimeError('Visible occupancy outside requested stratum')
    return view


def sample(source, source_index, view_index, *, seed=20260925, domain='train',
           kind='photo', integrity='complete', family='viewer_screen', source_crop=False,
           high_occupancy=True, background=None, mode='mild', level='light', pure_roll=False,
           scene_version='partial-v3', occupancy_range=None, occupancy_bin_v2=None,
           missing_range=None, missing_target=None, source_crop_area_loss=None,
           capture_material=None, capture_severity=None, capture_seed=None,
           edge_blend_px224=0.):
    if scene_version == STEP2000_SCENE:
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError('Expected uint8 RGB source')
        if domain != 'train':
            raise ValueError('New scene version is train-only')
        if kind == 'keep':
            if source_crop or integrity != 'complete':
                raise ValueError('Keep must preserve the original source')
            original = sample(source, source_index, view_index, seed=seed,
                              domain=domain, kind='keep', scene_version='partial-v3')
            original.info['scene_version'] = STEP2000_SCENE
            return original
        return _sample_step2000(
            source, source_index, view_index, seed=seed, domain=domain, kind=kind,
            integrity=integrity, family=family, source_crop=source_crop,
            background=background, mode=mode, pure_roll=pure_roll,
            occupancy_range=occupancy_range, occupancy_bin_v2=occupancy_bin_v2,
            missing_range=missing_range, missing_target=missing_target,
            source_crop_area_loss=source_crop_area_loss,
            capture_material=capture_material, capture_severity=capture_severity,
            capture_seed=capture_seed, edge_blend_px224=edge_blend_px224)
    if scene_version != 'partial-v3':
        raise ValueError('Unknown scene version')
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
