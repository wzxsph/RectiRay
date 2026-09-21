"""Read-only source import; private metadata and cache live under .local/."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode.startswith("I") or im.mode == "F":
            a = np.asarray(im, np.float32)
            lo, hi = np.percentile(a[np.isfinite(a)], (.5, 99.5))
            a = np.clip((a - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
            return np.repeat(a[..., None], 3, axis=-1)
        return np.asarray(im.convert("RGB"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare(sources: dict[str, Path], output: Path, size: int = 224) -> dict:
    if (output / "manifest.local.json").exists():
        raise FileExistsError("Existing data cache: choose a new output instead of overwriting")
    for source in sources.values():
        if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
            raise ValueError("Cache must be outside read-only input directories")
    entries = [(name, p) for name, root in sources.items() for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not entries:
        raise ValueError("No image files")
    output.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype=np.uint8, shape=(len(entries), size, size, 3))
    records, failed, group_counts = [], [], Counter()
    for index, (name, path) in enumerate(entries):
        sid = f"s{index + 1:05d}"
        try:
            image = read_rgb(path)
            if min(image.shape[:2]) < 16:
                raise ValueError("Image too small")
            resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            cache[index] = resized
            # Retain every file. Only use exact decoded-pixel equality to keep
            # obvious duplicate copies together; no perceptual deduplication.
            group = hashlib.sha256(resized.tobytes()).digest()
            split = "val" if int.from_bytes(group[:4], "little") % 10 == 0 else "train"
            group_counts[group] += 1
            records.append({"id": sid, "cache_index": index, "source_name": name, "source_path": str(path.resolve()), "split": split, "original_size": [image.shape[1], image.shape[0]], "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except (OSError, ValueError) as error:
            cache[index] = 0
            failed.append({"id": sid, "reason": type(error).__name__})
        if (index + 1) % 500 == 0:
            print(f"Read {index + 1}/{len(entries)}", flush=True)
    cache.flush()
    summary = {"total_files": len(entries), "readable": len(records), "excluded_unreadable": len(failed), "sources": dict(Counter(v[0] for v in entries)), "splits": dict(Counter(v["split"] for v in records)), "exact_duplicate_copies_retained": sum(n - 1 for n in group_counts.values()), "patient_mapping_available": False, "split_scope": "source_image_development_only", "source_modified": False}
    write_json(output / "manifest.local.json", {"records": records, "failed": failed, "summary": summary})
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def load_cache(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((root / "manifest.local.json").read_text())
    indices = np.array([r["cache_index"] for r in metadata["records"] if r["split"] == split], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"Empty {split} split")
    return np.load(root / "images.npy", mmap_mode="r"), indices
