"""Independent implementation of PTRN-style labeled photo synthesis."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .data import read_rgb
from .geometry import corners, pixel_matrix, project


class Synthesizer:
    def __init__(self, backgrounds: Path, rotation_degrees: float = 45, screen_probability: float = .3):
        self.rotation = rotation_degrees
        self.screen_probability = screen_probability
        self.backgrounds = [cv2.resize(read_rgb(p), (256, 256), interpolation=cv2.INTER_AREA) for p in sorted(backgrounds.glob("*.jpg"))]
        if not self.backgrounds:
            raise ValueError("Natural background images are required")

    def matrix(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(100):
            sx, sy = rng.uniform(.22, .82, 2)
            if abs(sx - sy) > .2:
                continue
            angle = np.deg2rad(rng.uniform(-self.rotation, self.rotation))
            r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            shear = np.array([[1, rng.uniform(-.1, .1)], [rng.uniform(-.1, .1), 1]])
            m = np.eye(3)
            m[:2, :2] = r @ shear @ np.diag([sx, sy])
            m[:2, 2] = rng.normal(0, .18, 2)
            m[2, :2] = rng.normal(0, .065, 2)
            q = corners(m)
            if np.max(np.abs(q)) < .97 and cv2.contourArea(q.astype(np.float32)) > .22:
                return m
        raise RuntimeError("Could not sample valid visible geometry")

    def render(self, source: np.ndarray, matrix: np.ndarray, background: np.ndarray, margins=None) -> np.ndarray:
        size = source.shape[0]
        canvas = cv2.resize(background, (size, size)).copy()
        if margins is not None:
            left, top, right, bottom = margins
            frame = np.array([[-1 - left, -1 - top], [1 + right, -1 - top], [1 + right, 1 + bottom], [-1 - left, 1 + bottom]])
            polygon = (project(frame, matrix) + 1) * size / 2
            cv2.fillConvexPoly(canvas, polygon.astype(np.int32), (12, 12, 12))
        transform = pixel_matrix(matrix, (size, size), (size, size))
        mask = cv2.warpPerspective(np.ones((size, size), np.float32), transform, (size, size), flags=cv2.INTER_LINEAR)[..., None]
        warped = cv2.warpPerspective(source, transform, (size, size), flags=cv2.INTER_LINEAR)
        return np.clip(warped.astype(np.float32) + canvas * (1 - mask), 0, 255).astype(np.uint8)

    def sample(self, source: np.ndarray, seed: int, augment: bool = True) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        matrix = self.matrix(rng)
        bg = self.backgrounds[rng.integers(len(self.backgrounds))]
        y, x = rng.integers(0, 33, 2)
        bg = bg[y:y + 224, x:x + 224]
        margins = rng.uniform(.05, .55, 4) if rng.random() < self.screen_probability else None
        image = self.render(source, matrix, bg, margins)
        if augment:
            a = image.astype(np.float32)
            a = a * rng.uniform(.7, 1.3) * rng.uniform(.85, 1.15, (1, 1, 3)) + rng.uniform(-30, 30)
            if rng.random() < .3:
                a += rng.normal(0, rng.uniform(1, 12), a.shape)
            image = np.clip(a, 0, 255).astype(np.uint8)
            if rng.random() < .15:
                image = cv2.GaussianBlur(image, (3, 3), .6)
            if rng.random() < .5:
                success, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 99))])
                if not success:
                    raise ValueError("JPEG augmentation failed")
                image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return image, matrix.reshape(-1)[:8].astype(np.float32)

    def batch(self, images: np.ndarray, indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = [self.sample(images[int(i)], seed + j * 1009) for j, i in enumerate(indices)]
        return np.stack([x for x, _ in pairs]), np.stack([y for _, y in pairs])
