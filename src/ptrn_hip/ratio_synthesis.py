"""Frozen ratio-v1 synthesis. All pixel geometry uses center spans (size - 1)."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

SEED = 20260921
MODES = ('identity', 'mild', 'hard')
LEVELS = ('clear', 'light', 'medium', 'strong')
APPEARANCE = {
    'clear': (0., 0., 0., 1., 1., 0., 100, 0.),
    'light': (.30, .02, 1., .9, 1.1, .5, 85, .025),
    'medium': (.50, .05, 2., .8, 1.2, .8, 70, .05),
    'strong': (.60, .08, 3., .75, 1.3, 1.2, 60, .08),
}
UNIT = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])


def stream(source_index, view_index, domain, seed=SEED):
    return np.random.default_rng(np.random.SeedSequence([seed, int(source_index), int(view_index), domain]))


def project(points, h):
    q = np.c_[points, np.ones(len(points))] @ h.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-7):
        raise ValueError('Invalid projective denominator')
    return q[:, :2] / q[:, 2:]


def normalization(width, height):
    if min(width, height) < 2:
        raise ValueError('Center-span geometry requires at least two pixels')
    return np.array([[2/(width-1), 0, -1], [0, 2/(height-1), -1], [0, 0, 1.]])


def pixel_corners(width, height):
    return (UNIT + 1) * np.array([width-1, height-1]) / 2


def letterbox(image, size):
    """Isotropic pixel-center scaling; H exactly matches OpenCV warpAffine."""
    height, width = image.shape[:2]
    scale = (size-1)/max(width-1, height-1)
    h = np.array([[scale, 0, (size-1-scale*(width-1))/2],
                  [0, scale, (size-1-scale*(height-1))/2], [0, 0, 1.]])
    pixels = cv2.warpAffine(image, h[:2], (size, size), flags=cv2.INTER_LINEAR)
    return pixels, h


@dataclass
class View:
    clean: np.ndarray
    noisy: np.ndarray
    quad: np.ndarray
    screen_mask: np.ndarray
    log_ratio: float
    info: dict

    def input(self, size, noisy=True):
        pixels, lb = letterbox(self.noisy if noisy else self.clean, size)
        q = project(self.quad, lb)
        normalized = project(q, normalization(size, size))
        h = cv2.getPerspectiveTransform(UNIT.astype(np.float32), normalized.astype(np.float32))
        return pixels, np.r_[h.reshape(-1)[:8], self.log_ratio].astype(np.float32), q.astype(np.float32)


def appearance(image, screen_mask, rng, level):
    """Pixel changes only. Moire mask is the whole viewer, never the film quad."""
    probability, amp, noise, low, high, blur, quality, light = APPEARANCE[level]
    if level == 'clear':
        return image.copy(), {'moire': False, 'camera': 'none'}
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x, y = xx/max(width-1, 1)-.5, yy/max(height-1, 1)-.5
    a = image.astype(np.float32)/255
    mask = screen_mask[..., None]
    gamma = rng.uniform(low, high)
    a += (np.power(a, gamma)-a)*mask
    gradient = rng.uniform(-light, light)*x + rng.uniform(-light, light)*y
    a += gradient[..., None]*mask
    moire = rng.random() < probability
    if moire:
        texture = np.zeros((height, width), np.float32)
        for period_range in ((2., 8.), (8., 40.), (40., 160.)):
            period = rng.uniform(*period_range)
            angle, phase = rng.uniform(0, 2*np.pi, 2)
            coordinate = xx*np.cos(angle) + yy*np.sin(angle)
            perpendicular = -xx*np.sin(angle) + yy*np.cos(angle)
            texture += (np.sin(coordinate*(2*np.pi/period)+phase) +
                        np.sin(perpendicular*(2*np.pi/rng.uniform(*period_range))+phase))/6
        a += texture[..., None]*rng.uniform(0, amp)*mask
    camera = str(rng.choice(('none', 'blur', 'jpeg', 'noise'), p=(.4, .2, .2, .2)))
    if camera == 'noise':
        a += rng.normal(0, rng.uniform(0, noise)/255, a.shape).astype(np.float32)
    pixels = np.clip(a*255, 0, 255).astype(np.uint8)
    if camera == 'blur':
        sigma = rng.uniform(.01, blur)
        pixels = cv2.GaussianBlur(pixels, (0, 0), sigma)
    elif camera == 'jpeg':
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(quality, 101))])
        if not ok:
            raise RuntimeError('JPEG augmentation failed')
        pixels = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return pixels, {'moire': moire, 'camera': camera, 'gamma': float(gamma)}


def sample(source, source_index, view_index, *, mode=None, level=None, seed=SEED):
    geometry = stream(source_index, view_index, 0, seed)
    background = stream(source_index, view_index, 1, seed)
    exterior = stream(source_index, view_index, 2, seed)
    mode = mode or str(geometry.choice(MODES, p=(.3, .6, .1)))
    requested = level or str(exterior.choice(LEVELS, p=(.45, .35, .15, .05)))
    if mode not in MODES or requested not in LEVELS:
        raise ValueError('Unknown synthesis stratum')
    actual = 'light' if mode == 'hard' and requested == 'strong' else requested
    h0, w0 = source.shape[:2]
    # Continuous film boundaries are +/- W/2H and +/- 1/2, with center vertices inset .5/H.
    plane = np.array([[1/h0, 0, -(w0-1)/(2*h0)], [0, 1/h0, -(h0-1)/(2*h0)], [0, 0, 1.]])
    ratio = w0/h0
    margins = background.uniform(.04, .22, 4)
    left, top, right, bottom = [-ratio/2-margins[0], -.5-margins[1], ratio/2+margins[2], .5+margins[3]]
    screen = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    photo_ratio = geometry.uniform(.8, 1.4)
    width, height = (768, round(768/photo_ratio)) if photo_ratio >= 1 else (round(768*photo_ratio), 768)
    bound = 0 if mode == 'identity' else (15 if mode == 'mild' else 40)
    roll_bound = 0 if mode == 'identity' else (12 if mode == 'mild' else 45)
    yaw, pitch = np.deg2rad(geometry.uniform(-bound, bound, 2))
    roll = np.deg2rad(geometry.uniform(-roll_bound, roll_bound))
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rotation = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1.]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    focal = geometry.uniform(900, 1400)
    k = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0,0,1.]])
    fill = geometry.uniform(.74, .95)
    z = max((right-left)*focal/width, (bottom-top)*focal/height)/fill
    tx, ty = geometry.uniform(-.035, .035, 2)*z
    for attempt in range(100):
        camera = k @ np.column_stack((rotation[:, 0], rotation[:, 1], [tx,ty,z]))
        projected_screen = project(screen, camera)
        depths = np.c_[screen, np.zeros(4)] @ rotation[2] + z
        if (depths > .1).all() and (projected_screen >= 4).all() and (projected_screen <= [width-5,height-5]).all():
            break
        z *= 1.035
    else:
        raise RuntimeError('Camera could not keep the screen visible')
    transform = camera @ plane
    quad = project(pixel_corners(w0,h0), transform)
    if not cv2.isContourConvex(quad.astype(np.float32)) or cv2.contourArea(quad.astype(np.float32), oriented=True) <= 0:
        raise ValueError('Mirrored or degenerate generated film')
    # Procedural background has no source/patient relationship and cannot cross-leak a split.
    small = background.integers(15, 180, (5,5,3), dtype=np.uint8)
    canvas = cv2.resize(small, (width,height), interpolation=cv2.INTER_CUBIC)
    screen_mask = np.zeros((height,width), np.float32)
    cv2.fillConvexPoly(screen_mask, np.rint(projected_screen).astype(np.int32), 1.)
    screen_color = int(background.choice([18,38,70,160,220]))
    cv2.fillConvexPoly(canvas, np.rint(projected_screen).astype(np.int32), (screen_color,)*3)
    toolbar = project(np.array([[left,top],[right,top],[right,-.5],[left,-.5]]), camera)
    cv2.fillConvexPoly(canvas, np.rint(toolbar).astype(np.int32), (48,57,68))
    for xpos in np.linspace(left+.04, right-.12, 8):
        button = project(np.array([[xpos,top+.01],[xpos+.04,top+.01],[xpos+.04,top+margins[1]*.7],[xpos,top+margins[1]*.7]]), camera)
        cv2.fillConvexPoly(canvas, np.rint(button).astype(np.int32), (160,170,180))
    warped = cv2.warpPerspective(source, transform, (width,height), flags=cv2.INTER_LINEAR)
    film_mask = cv2.warpPerspective(np.ones((h0,w0),np.float32), transform, (width,height), flags=cv2.INTER_LINEAR)
    clean = np.clip(warped.astype(np.float32)+canvas*(1-film_mask[...,None]),0,255).astype(np.uint8)
    saved_quad = quad.copy()
    noisy, appearance_info = appearance(clean,screen_mask,exterior,actual)
    assert np.array_equal(quad,saved_quad)
    return View(clean,noisy,quad,screen_mask,float(np.log(ratio)),
                {'mode':mode,'requested_appearance':requested,'appearance':actual,
                 'yaw':float(np.rad2deg(yaw)),'pitch':float(np.rad2deg(pitch)), 'roll':float(np.rad2deg(roll)),
                 'camera_attempts':attempt+1,'common_size':[width,height],**appearance_info})
