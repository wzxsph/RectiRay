"""Projective geometry in PTRN's normalized [-1, 1] coordinates."""
from __future__ import annotations

import cv2
import numpy as np

UNIT_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)


def as_matrix(parameters: np.ndarray) -> np.ndarray:
    a = np.asarray(parameters, np.float64)
    if a.size != 8 or not np.isfinite(a).all():
        raise ValueError("Expected eight finite projective parameters")
    return np.r_[a.reshape(-1), 1.0].reshape(3, 3)


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64)
    q = np.c_[p, np.ones(len(p))] @ np.asarray(matrix).T
    if not np.isfinite(q).all() or (np.abs(q[:, 2]) < 1e-7).any():
        raise ValueError("Degenerate projective denominator")
    return q[:, :2] / q[:, 2:]


def corners(matrix: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    q = project(UNIT_CORNERS, matrix)
    return (q + 1) * np.array(size) / 2 if size else q


def pixel_matrix(matrix: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> np.ndarray:
    def normalization(size):
        w, h = size
        return np.array([[2 / w, 0, -1], [0, 2 / h, -1], [0, 0, 1]], np.float64)
    return np.linalg.inv(normalization(target_size)) @ matrix @ normalization(source_size)


def validate_quad(q: np.ndarray, size: tuple[int, int]) -> list[str]:
    q = np.asarray(q, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        return ["invalid_coordinates"]
    w, h = size
    reasons = []
    if not cv2.isContourConvex(q) or cv2.contourArea(q, oriented=True) <= 0:
        reasons.append("nonconvex_or_mirrored")
    if abs(cv2.contourArea(q)) < w * h * 0.05:
        reasons.append("region_too_small")
    if (q < -np.array(size) * .02).any() or (q > np.array(size) * 1.02).any():
        reasons.append("outside_photo")
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    if sides.min() < 30:
        reasons.append("short_edge")
    if any(min(sides[i], sides[i + 2]) / max(sides[i], sides[i + 2], 1) < .25 for i in (0, 1)):
        reasons.append("extreme_perspective")
    return reasons


def rectify(rgb: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    q = corners(matrix, (w, h)).astype(np.float32)
    reasons = validate_quad(q, (w, h))
    if reasons:
        raise ValueError(",".join(reasons))
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    ow, oh = max(32, round(max(sides[0], sides[2])) + 1), max(32, round(max(sides[1], sides[3])) + 1)
    if ow * oh > 24_000_000:
        raise ValueError("unreasonable_output_size")
    dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(rgb, transform, (ow, oh), flags=cv2.INTER_CUBIC), transform


def geometry_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    errors, overlaps, invalid = [], [], 0
    for y, p in zip(truth, prediction, strict=True):
        a = corners(as_matrix(y)).astype(np.float32)
        try:
            b = corners(as_matrix(p)).astype(np.float32)
            error = np.linalg.norm(a - b, axis=1).mean() / (2 * np.sqrt(2))
            if not cv2.isContourConvex(b) or cv2.contourArea(b, oriented=True) <= 0:
                raise ValueError("invalid_quad")
            intersection, _ = cv2.intersectConvexConvex(a, b)
            union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
            overlaps.append(float(max(0, intersection) / max(union, 1e-8)))
            errors.append(float(error))
        except (ValueError, cv2.error):
            invalid += 1
            overlaps.append(0.0)
            errors.append(1.0)
    return {"corner_nme_mean": float(np.mean(errors)), "corner_nme_p90": float(np.percentile(errors, 90)), "quad_iou_mean": float(np.mean(overlaps)), "invalid": invalid, "count": len(errors)}
