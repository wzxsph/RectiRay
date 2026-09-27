"""Read-only validation for local, pre-split RectiRay film/background assets.

This module deliberately imports no model or training runtime. Reports contain
aggregate counts and SHA bindings, never source filenames or absolute paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat

import numpy as np
from PIL import Image, ImageOps


FILM_SPLITS = ("train", "dev", "quarantine")
BACKGROUND_SPLITS = ("train", "dev")


class AssetError(ValueError):
    """An asset, binding, or safety boundary is invalid."""


def _regular_nofollow(path: Path):
    """Open a regular file without following a last-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise AssetError("Asset is not a regular file")
    return os.fdopen(fd, "rb")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_nofollow(Path(path)) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(source: Path, target: Path) -> str:
    """Byte-copy a regular source and verify it stayed unchanged during copy."""
    source, target = Path(source), Path(target)
    before = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    copied = hashlib.sha256()
    with _regular_nofollow(source) as reader, target.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)
            copied.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    if before != copied.hexdigest() or before != sha256_file(source):
        raise AssetError("Input file changed during byte copy")
    return before


def decoded_rgb(path: Path) -> tuple[int, int, str]:
    """Mirror PartialDataset's EXIF/intensity conversion without importing it."""
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode.startswith("I") or image.mode == "F":
                values = np.asarray(image, np.float32)
                low, high = np.percentile(values[np.isfinite(values)], (.5, 99.5))
                values = np.clip((values - low) * 255 / max(high - low, 1),
                                 0, 255).astype(np.uint8)
                rgb = np.repeat(values[..., None], 3, axis=-1)
            else:
                rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError, TypeError, IndexError) as error:
        raise AssetError("Asset does not decode as an RGB image") from error
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype.name != "uint8"
            or min(rgb.shape[:2]) < 16):
        raise AssetError("Decoded RGB image has invalid size or format")
    height, width = rgb.shape[:2]
    return width, height, hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def _root(root: Path) -> Path:
    root = Path(root).expanduser().absolute()
    if any(part.is_symlink() for part in (root, *root.parents)) or not root.is_dir():
        raise AssetError("Asset root is absent or a symlink")
    return root


def safe_asset_path(root: Path, relative: str) -> Path:
    """Resolve one manifest asset while refusing traversal and any symlink."""
    root = _root(root)
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or relative.startswith("/") or any(part in ("", ".", "..")
                                               for part in relative.split("/"))):
        raise AssetError("Unsafe asset relative path")
    pieces = PurePosixPath(relative).parts
    path = root.joinpath(*pieces)
    cursor = root
    for piece in pieces:
        cursor = cursor / piece
        if cursor.is_symlink():
            raise AssetError("Asset path contains a symlink")
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise AssetError("Asset path escapes its root")
    if not path.is_file():
        raise AssetError("Manifest asset is missing or not a regular file")
    return path


def split_files(root: Path, split: str, *, required: bool) -> list[Path]:
    """Enumerate a user-provided split in stable path order, rejecting links."""
    root = _root(root)
    branch = root / split
    if branch.is_symlink():
        raise AssetError("Input split directory is a symlink")
    if not branch.exists() and not required:
        return []
    if not branch.is_dir():
        raise AssetError("Required split directory is absent or a symlink")
    found = []
    for current, dirs, files in os.walk(branch, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            if (current_path / name).is_symlink():
                raise AssetError("Input tree contains a symlink directory")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise AssetError("Input tree contains a symlink or non-file asset")
            found.append(path)
    found.sort(key=lambda path: path.relative_to(branch).as_posix())
    if required and not found:
        raise AssetError("Required input split is empty")
    return found


def _json_file(root: Path, relative: str):
    try:
        return json.loads(safe_asset_path(root, relative).read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssetError("Manifest is not valid UTF-8 JSON") from error


def _duplicate_report(entries: list[tuple[str, str]], *, quarantine: bool) -> dict:
    groups: dict[str, Counter] = defaultdict(Counter)
    for split, digest in entries:
        groups[digest][split] += 1
    cross = 0
    retained = 0
    quarantine_overlap = 0
    for counts in groups.values():
        if counts["train"] and counts["dev"]:
            cross += 1
        retained += sum(max(value - 1, 0) for value in counts.values())
        if quarantine and counts["quarantine"] and (counts["train"] or counts["dev"]):
            quarantine_overlap += 1
    return {"cross_train_dev_groups": cross,
            "same_split_extra_copies": retained,
            "quarantine_overlap_groups": quarantine_overlap}


def validate_bundle(bundle: Path, backgrounds: Path) -> dict:
    """Verify every manifest row, including quarantine and legacy viewer rows.

    The function accepts authorized read roots anywhere, but refuses manifest
    traversal and symlink assets. Its return value is safe aggregate metadata.
    """
    bundle, backgrounds = _root(bundle), _root(backgrounds)
    film_manifest = safe_asset_path(bundle, "data/manifest.json")
    split_manifest = safe_asset_path(bundle, "data/split.json")
    background_manifest = safe_asset_path(bundle, "data/backgrounds-manifest.json")
    films = _json_file(bundle, "data/manifest.json")
    splits = _json_file(bundle, "data/split.json")
    background_data = _json_file(bundle, "data/backgrounds-manifest.json")
    records = films.get("records") if isinstance(films, dict) else None
    if not isinstance(records, list) or not records:
        raise AssetError("Film manifest has no records")
    if (not isinstance(splits, dict) or set(splits) != set(FILM_SPLITS)
            or not all(isinstance(splits.get(k), list) for k in FILM_SPLITS)):
        raise AssetError("Film split manifest is incomplete")
    flat = [index for split in FILM_SPLITS for index in splits[split]]
    if (len(flat) != len(records) or any(type(index) is not int for index in flat)
            or sorted(flat) != list(range(len(records)))):
        raise AssetError("Film split indices overlap, omit, or exceed records")
    if not splits["train"] or not splits["dev"]:
        raise AssetError("Train and development film splits must be nonempty")
    members = {index: split for split in FILM_SPLITS for index in splits[split]}
    film_ids = set()
    film_bytes = []
    film_pixels = []
    for index, row in enumerate(records):
        if not isinstance(row, dict) or row.get("split") != members[index]:
            raise AssetError("Film record and split membership differ")
        identifier = row.get("id")
        if identifier is not None:
            if not isinstance(identifier, str) or not identifier or identifier in film_ids:
                raise AssetError("Film record IDs are invalid or duplicated")
            film_ids.add(identifier)
        path = safe_asset_path(bundle, row.get("source_path"))
        byte_hash = sha256_file(path)
        if byte_hash != row.get("source_sha256"):
            raise AssetError("Film asset SHA256 mismatch")
        width, height, pixel_hash = decoded_rgb(path)
        if "original_size" in row and row["original_size"] != [width, height]:
            raise AssetError("Film dimensions differ from manifest")
        if row.get("pixel_sha256") and row["pixel_sha256"] != pixel_hash:
            raise AssetError("Film decoded-pixel SHA256 mismatch")
        film_bytes.append((row["split"], byte_hash))
        film_pixels.append((row["split"], pixel_hash))
    rows = background_data.get("rows") if isinstance(background_data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise AssetError("Background manifest has no rows")
    background_ids = set()
    bg_bytes = []
    bg_pixels = []
    bg_counts = Counter()
    real_counts = Counter()
    for row in rows:
        if not isinstance(row, dict) or row.get("split") not in BACKGROUND_SPLITS:
            raise AssetError("Background row has an invalid split")
        kind = row.get("source_kind")
        if kind not in ("real_photo", "procedural_interface"):
            raise AssetError("Background row has an unknown source kind")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in background_ids:
            raise AssetError("Background IDs are invalid or duplicated")
        background_ids.add(identifier)
        path = safe_asset_path(backgrounds, row.get("path"))
        byte_hash = sha256_file(path)
        if byte_hash != row.get("sha256"):
            raise AssetError("Background asset SHA256 mismatch")
        width, height, pixel_hash = decoded_rgb(path)
        if (("width" in row and row["width"] != width)
                or ("height" in row and row["height"] != height)):
            raise AssetError("Background dimensions differ from manifest")
        if row.get("pixel_sha256") and row["pixel_sha256"] != pixel_hash:
            raise AssetError("Background decoded-pixel SHA256 mismatch")
        bg_bytes.append((row["split"], byte_hash))
        bg_pixels.append((row["split"], pixel_hash))
        bg_counts[row["split"]] += 1
        if kind == "real_photo":
            real_counts[row["split"]] += 1
    if not real_counts["train"] or not real_counts["dev"]:
        raise AssetError("Train and development need real-photo backgrounds")
    duplicates = {
        "films_file": _duplicate_report(film_bytes, quarantine=True),
        "films_pixel": _duplicate_report(film_pixels, quarantine=True),
        "backgrounds_file": _duplicate_report(bg_bytes, quarantine=False),
        "backgrounds_pixel": _duplicate_report(bg_pixels, quarantine=False),
    }
    if any(item["cross_train_dev_groups"] for item in duplicates.values()):
        raise AssetError("Exact or decoded-pixel duplicate crosses train and development")
    return {
        "manifest_sha256": sha256_file(film_manifest),
        "split_sha256": sha256_file(split_manifest),
        "background_manifest_sha256": sha256_file(background_manifest),
        "split_counts": {name: len(splits[name]) for name in FILM_SPLITS},
        "background_counts": {name: bg_counts[name] for name in BACKGROUND_SPLITS},
        "background_real_photo_counts": {name: real_counts[name] for name in BACKGROUND_SPLITS},
        "duplicates": duplicates,
        "all_film_records_checked": len(records),
        "all_background_rows_checked": len(rows),
    }
