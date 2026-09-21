"""Identity / mild / legacy-hard inputs with exact generated homographies."""
from __future__ import annotations

from collections import Counter
import cv2
import numpy as np

from .geometry import corners, project
from .synthesis import Synthesizer


class MixedSynthesizer(Synthesizer):
    MODES = ("identity", "mild", "hard")

    def __init__(self, backgrounds, rotation_degrees=45, screen_probability=.3, mixture=None):
        super().__init__(backgrounds, rotation_degrees, screen_probability)
        weights = mixture or {"identity": .3, "mild": .6, "hard": .1}
        if set(weights) != set(self.MODES):
            raise ValueError("Mixture must specify identity, mild, hard")
        self.probabilities = np.array([weights[k] for k in self.MODES], dtype=float)
        if (self.probabilities < 0).any() or not np.isclose(self.probabilities.sum(), 1):
            raise ValueError("Mixture probabilities must be nonnegative and sum to one")

    def mild_matrix(self, rng):
        for attempt in range(1, 1001):
            sx, sy = rng.uniform(.65, .98, 2)
            if abs(sx - sy) > .08:
                continue
            angle = rng.uniform(-12, 12)
            a = np.deg2rad(angle)
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            shear = np.array([[1, rng.uniform(-.025, .025)], [rng.uniform(-.025, .025), 1]])
            matrix = np.eye(3)
            matrix[:2, :2] = rotation @ shear @ np.diag([sx, sy])
            matrix[:2, 2] = rng.uniform(-.10, .10, 2)
            matrix[2, :2] = np.clip(rng.normal(0, .01, 2), -.03, .03)
            q = corners(matrix).astype(np.float32)
            if np.max(np.abs(q)) <= .995 and cv2.isContourConvex(q) and cv2.contourArea(q, oriented=True) > .22:
                return matrix, {"angle_degrees": float(angle), "sx": float(sx), "sy": float(sy), "sampling_attempts": attempt}
        raise RuntimeError("Could not sample a fully visible mild photo")

    def window_background(self, background, matrix, rng):
        """Paint the viewer around the target, using the same plane transform."""
        result = cv2.resize(background, (224, 224)).copy()
        left, top, right, bottom = rng.uniform(.07, .3, 4)

        def rectangle(x0, y0, x1, y1, color):
            p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            p = (project(p, matrix) + 1) * 112
            cv2.fillConvexPoly(result, np.rint(p).astype(np.int32), color)

        gray = int(rng.choice([28, 70, 180, 235]))
        rectangle(-1-left, -1-top, 1+right, 1+bottom, (gray, gray, gray))
        rectangle(-1-left, -1-top, 1+right, -1, (48, 57, 68))
        # All decorations stay strictly outside the target source canvas.
        for x in np.linspace(-.9, .8, 8):
            rectangle(x, -1-top*.78, x+.07, -1-top*.3, (190, 195, 200))
        if right > .12:
            for y in np.linspace(-.8, .7, 6):
                rectangle(1+right*.2, y, 1+right*.8, y+.08, (120, 125, 130))
        return result

    def sample_record(self, source, seed, augment=True, mode=None):
        rng = np.random.default_rng(seed)
        mode = mode or str(rng.choice(self.MODES, p=self.probabilities))
        if mode not in self.MODES:
            raise ValueError(mode)
        info = {"mode": mode, "window": False}
        if mode == "identity":
            image, matrix = source.copy(), np.eye(3)
            info.update(angle_degrees=0., sampling_attempts=1)
        elif mode == "hard":
            image, label = super().sample(source, seed, augment)
            matrix = np.r_[label, 1.].reshape(3, 3)
        else:
            matrix, details = self.mild_matrix(rng)
            info.update(details)
            bg = self.backgrounds[rng.integers(len(self.backgrounds))]
            info["window"] = bool(rng.random() < .5)
            if info["window"]:
                bg = self.window_background(bg, matrix, rng)
            image = self.render(source, matrix, bg)
            if augment:
                a = image.astype(np.float32) * rng.uniform(.85, 1.15) + rng.uniform(-15, 15)
                if rng.random() < .2:
                    a += rng.normal(0, rng.uniform(1, 4), a.shape)
                image = np.clip(a, 0, 255).astype(np.uint8)
                if rng.random() < .1:
                    image = cv2.GaussianBlur(image, (3, 3), .4)
                if rng.random() < .3:
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(85, 99))])
                    if not ok:
                        raise ValueError("JPEG augmentation failed")
                    image = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        q = corners(matrix).astype(np.float32)
        info["area_fraction"] = float(cv2.contourArea(q) / 4)
        return image, matrix.reshape(-1)[:8].astype(np.float32), info

    def sample(self, source, seed, augment=True, mode=None):
        image, label, _ = self.sample_record(source, seed, augment, mode)
        return image, label

    def batch_records(self, images, indices, seed):
        rows = [self.sample_record(images[int(i)], seed+j*1009) for j, i in enumerate(indices)]
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), [r[2] for r in rows]


def summarize_samples(records):
    result = {"count": len(records), "modes": dict(Counter(r["mode"] for r in records))}
    mild = [r for r in records if r["mode"] == "mild"]
    if mild:
        attempts = sum(r["sampling_attempts"] for r in mild)
        result["mild"] = {"window_count": sum(r["window"] for r in mild),
                          "sampling_rejection_fraction": 1-len(mild)/attempts}
        for key in ("angle_degrees", "area_fraction"):
            values = [r[key] for r in mild]
            result["mild"][key] = {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}
    return result
