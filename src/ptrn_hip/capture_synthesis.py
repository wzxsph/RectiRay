"""Opt-in capture appearance v1; never changes geometry or the frozen renderer.

Procedural artifact approximations for human recipe review, not a calibrated
camera/optical simulator. All randomness is local and recorded by effect seed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .ratio_synthesis import letterbox

VERSION = "ptrn-capture-appearance-v1"
SCREEN = ("moire_gray", "moire_color", "bands")
SURFACE = SCREEN + ("lightbox", "haze", "window", "lamp", "highlight")
CAMERA = ("defocus", "motion", "noise", "exposure", "white_balance", "jpeg")
LABELS = {
    "moire_gray": "灰度摩尔纹", "moire_color": "彩色摩尔纹", "bands": "明暗条带",
    "lightbox": "观片灯亮度不均", "haze": "雾状反射", "window": "窗户倒影",
    "lamp": "灯条倒影", "highlight": "局部高光／过曝", "defocus": "失焦",
    "motion": "手抖拖影", "noise": "传感器噪声", "exposure": "曝光",
    "white_balance": "白平衡", "jpeg": "JPEG 压缩",
}


def pixel_delta(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {"mae_255": float(d.mean()), "changed_fraction": float(np.any(d > 0, axis=2).mean()),
            "max_255": float(d.max())}


def recipe(material, severity, seed, effects=None):
    if material not in ("keep", "screen", "film") or severity not in ("clear", "light", "medium", "strong"):
        raise ValueError("Unknown material/severity")
    allowed = (SURFACE if material == "screen" else tuple(x for x in SURFACE if x not in SCREEN)) + CAMERA
    if material == "keep" and severity != "clear":
        raise ValueError("Originals must remain clear")
    rng = np.random.default_rng(seed)
    limit = 0 if severity == "clear" else (2 if severity == "light" else 3)
    if effects is None:
        effects = [] if not limit else list(rng.choice(allowed, size=int(rng.integers(1, limit + 1)), replace=False))
    if len(effects) > limit or len(set(effects)) != len(effects) or any(e not in allowed for e in effects):
        raise ValueError("Invalid effect combination")
    if severity != "clear" and not effects:
        raise ValueError("Non-clear recipe needs an effect")
    if material == "keep" and effects:
        raise ValueError("Originals cannot receive effects")
    # Surface illumination first, then reflections/texture, then camera processing.
    order = ["lightbox", "moire_gray", "moire_color", "bands", "haze", "window", "lamp", "highlight", *CAMERA]
    effects = sorted(effects, key=order.index)
    return {"version": VERSION, "material": material, "severity": severity, "seed": int(seed),
            "effects": [{"name": str(e), "seed": int(rng.integers(0, 2**32))} for e in effects]}


def plane_coordinates(shape, transform, plane_size):
    """Photo pixel centers mapped to normalized reference-plane coordinates."""
    y, x = np.indices(shape, dtype=np.float32)
    inv = np.linalg.inv(transform)
    den = inv[2, 0]*x + inv[2, 1]*y + inv[2, 2]
    den = np.where(np.abs(den) < 1e-8, 1e-8, den)
    u = (inv[0, 0]*x + inv[0, 1]*y + inv[0, 2])/den/max(1, plane_size[0]-1)
    v = (inv[1, 0]*x + inv[1, 1]*y + inv[1, 2])/den/max(1, plane_size[1]-1)
    return u.astype(np.float32), v.astype(np.float32)


def _effect(image, surface, uv, name, severity, seed):
    rng = np.random.default_rng(seed)
    strength = {"light": .45, "medium": .8, "strong": 1.25}[severity]
    a = image.astype(np.float32)/255
    u, v = uv
    h, w = image.shape[:2]
    p = {"strength": strength}
    scope = "surface" if name in SURFACE else "whole_photo"
    if name in SURFACE:
        angle = float(rng.uniform(-np.pi, np.pi))
        phase = float(rng.uniform(0, 2*np.pi))
        cx, cy = map(float, rng.uniform(.25, .75, 2))
        x = (u-cx)*np.cos(angle) + (v-cy)*np.sin(angle)
        y = -(u-cx)*np.sin(angle) + (v-cy)*np.cos(angle)
        p.update(angle_rad=angle, phase_rad=phase, center_plane=[cx, cy])
        if name.startswith("moire"):
            frequencies = [float(rng.uniform(3, 8)), float(rng.uniform(12, 30)), float(rng.uniform(70, 125))]
            # A beat between slightly differing grids plus explicit broad bands.
            # Unlike white noise, this has coherent structure at several scales.
            color_phase = np.array([0, 2.1, 4.2]) if name == "moire_color" else np.zeros(3)
            wave = sum(weight*np.sin(2*np.pi*f*(x[..., None]+.12*y[..., None]**2)+phase+color_phase)
                       for weight, f in zip((.50, .30, .20), frequencies))
            beat = np.sin(2*np.pi*91*x)*np.sin(2*np.pi*95*(x+.018*y))
            wave = .8*wave+.2*beat[..., None]
            a = a*(1+.12*strength*wave)+.035*strength*wave
            p.update(frequencies_per_plane=frequencies, beat_frequencies=[91, 95], color_phase=color_phase.tolist())
        elif name == "bands":
            f = float(rng.uniform(2, 7)); p["cycles_per_plane"] = f
            a *= 1+.19*strength*np.sin(2*np.pi*f*y+phase)[..., None]
        elif name == "lightbox":
            gain = 1+strength*(.35*x-.30*((u-.5)**2+(v-.5)**2)+.12*np.cos(np.pi*v))
            a *= gain[..., None]
        else:
            if name == "window":
                width, height = map(float, rng.uniform(.28, .60, 2))
                edge = .004 if rng.random() < .5 else .025
                inside = np.clip((width/2-np.abs(x))/edge, 0, 1)*np.clip((height/2-np.abs(y))/edge, 0, 1)
                bars = 1-.90*np.maximum(np.exp(-(x/.012)**2), np.exp(-(y/.012)**2))
                layer = inside*bars
                p.update(window_size_plane=[width, height], edge_softness_plane=edge)
            elif name == "lamp":
                width = float(rng.uniform(.012, .035)); length = float(rng.uniform(.3, .6))
                layer = np.exp(-.5*(x/width)**2)*np.exp(-.5*(y/length)**8)
                p.update(width_plane=width, length_plane=length)
            elif name == "haze":
                layer = np.exp(-.5*((x/.42)**2+(y/.33)**2))
            else:
                layer = np.exp(-.5*((x/.11)**2+(y/.09)**2))
            # Linear-light addition and clipping allow real saturation rather
            # than only a translucent white overlay. No anatomy is synthesized.
            amplitude = {"window": .42, "lamp": .8, "haze": .22, "highlight": 1.9}[name]*strength
            tint = np.array([1., .96, .87], np.float32)
            a = np.power(np.clip(a, 0, 1)**2.2+amplitude*layer[..., None]*tint, 1/2.2)
            p.update(reflection_linear_amplitude=amplitude, reflection_tint=tint.tolist())
        # Apply exact support once, including soft boundary coverage.
        a = image.astype(np.float32)/255*(1-surface[..., None])+a*surface[..., None]
    elif name == "defocus":
        sigma = float((.7+strength)*max(h, w)/224); p["sigma_native_px"] = sigma
        a = cv2.GaussianBlur(a, (0, 0), sigma)
    elif name == "motion":
        length = max(3, int(round((1.5+3*strength)*max(h, w)/224)) | 1)
        angle = float(rng.uniform(0, np.pi)); kernel = np.zeros((length, length), np.float32)
        r = (length-1)/2; dx, dy = r*np.cos(angle), r*np.sin(angle)
        cv2.line(kernel, (round(r-dx), round(r-dy)), (round(r+dx), round(r+dy)), 1, 1)
        kernel /= kernel.sum(); a = cv2.filter2D(a, -1, kernel)
        p.update(kernel_length_px=length, angle_rad=angle)
    elif name == "noise":
        sigma = (2+5*strength)/255
        a += rng.normal(0, sigma, a.shape).astype(np.float32)
        p["sigma_255"] = sigma*255
    elif name == "exposure":
        ev = float(rng.choice([-1, 1])*(.12+.5*strength)); a *= 2**ev; p["ev"] = ev
    elif name == "white_balance":
        gains = [1+.12*strength, 1., 1-.12*strength]
        if rng.random() < .5: gains.reverse()
        a *= np.array(gains); p["rgb_gains"] = gains
    elif name == "jpeg":
        quality = int(round(94-25*strength)); p["quality"] = quality
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok: raise RuntimeError("JPEG encoding failed")
        a = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)/255
    else:
        raise ValueError(name)
    out = np.rint(np.clip(a, 0, 1)*255).astype(np.uint8)
    if scope == "surface": out[surface == 0] = image[surface == 0]
    return out, {"name": name, "seed": int(seed), "scope": scope, "parameters": p}


def render(photo, surface, transform, plane_size, spec):
    if spec["version"] != VERSION:
        raise ValueError("Unknown renderer version")
    # Revalidate externally supplied recipes, including the original bypass.
    recipe(spec["material"], spec["severity"], spec["seed"], [e["name"] for e in spec["effects"]])
    if photo.dtype != np.uint8 or photo.ndim != 3 or photo.shape[2] != 3:
        raise ValueError("Expected uint8 RGB")
    if surface.shape != photo.shape[:2] or not np.isfinite(surface).all() or np.any((surface < 0) | (surface > 1)):
        raise ValueError("Invalid surface mask")
    out = photo.copy()
    uv = plane_coordinates(photo.shape[:2], transform, plane_size) if spec["effects"] else None
    details = []
    for effect in spec["effects"]:
        before = out
        out, detail = _effect(out, surface, uv, effect["name"], spec["severity"], effect["seed"])
        detail["native_delta"] = pixel_delta(before, out)
        detail["input224_delta"] = pixel_delta(letterbox(before, 224)[0], letterbox(out, 224)[0])
        # This checks implementation activity, not perceptual realism or quality.
        if detail["native_delta"]["max_255"] == 0 or detail["input224_delta"]["max_255"] == 0:
            raise RuntimeError(f"Inactive capture effect: {effect['name']}")
        details.append(detail)
    return out, {**spec, "effects": details, "native_delta": pixel_delta(photo, out),
                 "input224_delta": pixel_delta(letterbox(photo, 224)[0], letterbox(out, 224)[0])}
