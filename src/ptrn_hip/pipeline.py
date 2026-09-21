"""Explicit source routing before decoding, windowing, inference or resampling."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import cv2
from PIL import Image

from .data import read_rgb
from .geometry import as_matrix, rectify


def process_image(source, output=None, source_kind="unknown", predictor=None):
    source = Path(source)
    if source_kind not in {"digital", "phone", "unknown"}:
        raise ValueError("source_kind must be digital, phone or unknown")
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output) if output is not None else None
    if output is not None and (output.exists() or output.resolve() == source.resolve()):
        raise FileExistsError("Output must be a new file; originals and previous results are never overwritten")
    if source_kind in {"digital", "unknown"}:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, output.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        digest = hashlib.sha256((output or source).read_bytes()).hexdigest()
        return {"action": "digital_passthrough" if source_kind == "digital" else "unknown_preserved",
                "model_used": False, "decoded": False, "sha256": digest}
    if predictor is None or output is None or output.suffix.lower() != ".png":
        raise ValueError("Phone processing requires a predictor and a new .png output")
    photo = read_rgb(source)
    small = cv2.resize(photo, (224, 224), interpolation=cv2.INTER_AREA)
    prediction = predictor(small)
    flat, transform = rectify(photo, as_matrix(prediction))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        Image.fromarray(flat).save(stream, format="PNG")
    return {"action": "phone_candidate", "model_used": True, "decoded": True,
            "output_size": [flat.shape[1], flat.shape[0]], "photo_to_flat_matrix": transform.tolist()}
