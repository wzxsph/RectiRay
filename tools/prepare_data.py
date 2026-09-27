"""Package already-split local films and real-photo backgrounds for RectiRay.

Input layout (recursive images in each split; no patient/group inference)::

    SOURCES/train/  SOURCES/dev/  SOURCES/quarantine/  # last may be absent/empty
    BACKGROUNDS/train/  BACKGROUNDS/dev/

The output is a new ignored directory under this repository's `.local/`::

    OUTPUT/data/manifest.json
    OUTPUT/data/split.json
    OUTPUT/data/backgrounds-manifest.json
    OUTPUT/data/films/<split>/<anonymous-number>.<original-extension>
    OUTPUT/backgrounds/<split>/<anonymous-number>.<original-extension>

Use `OUTPUT` as the PartialDataset bundle and `OUTPUT/backgrounds` as its
background root. Source bytes are copied unchanged; no anonymization, license
decision, splitting, patient mapping, or clinical review is performed here.

`--provenance` optionally accepts a local JSON object with `sources` and
`backgrounds` arrays of records. Each record needs `identifier` and `license`;
`name`, `url`, and `scope` are optional strings. This record stays under `.local`
and is not a claim that rights or de-identification have been established.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.data_assets import (AssetError, BACKGROUND_SPLITS, FILM_SPLITS,
                               copy_verified, decoded_rgb, sha256_file,
                               split_files, validate_bundle)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _output_path(value: Path) -> Path:
    local = REPO / ".local"
    if local.is_symlink() or not local.is_dir():
        raise AssetError("Repository .local directory is absent or a symlink")
    output = Path(value).expanduser().absolute()
    if (output == local or not output.is_relative_to(local)
            or not output.resolve(strict=False).is_relative_to(local.resolve())):
        raise AssetError("Output must be a new directory inside repository .local")
    cursor = output.parent
    while cursor != local:
        if cursor.is_symlink():
            raise AssetError("Output parent contains a symlink")
        cursor = cursor.parent
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists; choose a new local path")
    relative = output.relative_to(REPO).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", relative],
                             cwd=REPO, check=False)
    if ignored.returncode != 0:
        raise AssetError("Output path is not Git-ignored")
    return output


def _input_root(value: Path) -> Path:
    path = Path(value).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_dir():
        raise AssetError("Input root is absent or a symlink")
    return path


def _provenance(path: Path | None) -> dict:
    if path is None:
        return {"status": "unknown", "sources": [], "backgrounds": []}
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise AssetError("Provenance file is absent or a symlink")
    if any(parent.is_symlink() for parent in path.parents):
        raise AssetError("Provenance path contains a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssetError("Provenance must be UTF-8 JSON") from error
    if not isinstance(raw, dict):
        raise AssetError("Provenance must be a JSON object")
    result = {"status": "provided"}
    for field in ("sources", "backgrounds"):
        values = raw.get(field)
        if not isinstance(values, list):
            raise AssetError("Provenance needs source and background record arrays")
        clean = []
        for row in values:
            if not isinstance(row, dict) or not isinstance(row.get("identifier"), str) \
                    or not row["identifier"].strip() or not isinstance(row.get("license"), str) \
                    or not row["license"].strip():
                raise AssetError("Provenance record needs identifier and license")
            if any(key not in ("identifier", "name", "url", "license", "scope")
                   or not isinstance(value, str) for key, value in row.items()):
                raise AssetError("Provenance record has an unsupported field")
            clean.append(dict(row))
        result[field] = clean
    return result


def _destination_name(prefix: str, ordinal: int, source: Path) -> str:
    suffix = source.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        raise AssetError("Input image extension is unsupported by the data contract")
    return f"{prefix}{ordinal:06d}{suffix}"


def _publish_new_directory(stage: Path, output: Path) -> None:
    """Linux atomic rename that refuses to replace an arriving output."""
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "Atomic no-replace directory rename unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(stage), at_fdcwd,
              os.fsencode(output), rename_noreplace) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError("Output appeared during atomic publication")
        raise OSError(code, os.strerror(code))


def prepare_data(sources: Path, backgrounds: Path, output: Path,
                 provenance: Path | None = None) -> dict:
    """Build and validate a bundle, then publish it with one directory rename."""
    output = _output_path(output)
    source_root, background_root = _input_root(sources), _input_root(backgrounds)
    for root in (source_root, background_root):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise AssetError("Output and input trees must be disjoint")
    if (source_root == background_root or source_root.is_relative_to(background_root)
            or background_root.is_relative_to(source_root)):
        raise AssetError("Film and background input trees must be disjoint")
    source_files = {name: split_files(source_root, name, required=name != "quarantine")
                    for name in FILM_SPLITS}
    background_files = {name: split_files(background_root, name, required=True)
                        for name in BACKGROUND_SPLITS}
    provenance_record = _provenance(provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    original_bindings: list[tuple[Path, str]] = []
    try:
        rows = []
        split_indices = {name: [] for name in FILM_SPLITS}
        for split in FILM_SPLITS:
            for path in source_files[split]:
                index = len(rows)
                relative = (Path("data") / "films" / split /
                            _destination_name("f", index + 1, path))
                target = stage / relative
                digest = copy_verified(path, target)
                original_bindings.append((path, digest))
                width, height, pixel_sha = decoded_rgb(target)
                rows.append({"id": f"f{index + 1:06d}", "cache_index": index,
                             "source_path": relative.as_posix(), "split": split,
                             "original_size": [width, height],
                             "source_sha256": digest, "pixel_sha256": pixel_sha})
                split_indices[split].append(index)
        background_rows = []
        for split in BACKGROUND_SPLITS:
            for path in background_files[split]:
                index = len(background_rows)
                relative = (Path(split) / _destination_name("b", index + 1, path))
                target = stage / "backgrounds" / relative
                digest = copy_verified(path, target)
                original_bindings.append((path, digest))
                width, height, pixel_sha = decoded_rgb(target)
                background_rows.append({"id": f"b{index + 1:06d}",
                                        "path": relative.as_posix(),
                                        "split": split, "source_kind": "real_photo",
                                        "sha256": digest, "pixel_sha256": pixel_sha,
                                        "width": width, "height": height,
                                        "license": "unknown"})
        film_counts = {name: len(split_indices[name]) for name in FILM_SPLITS}
        bg_counts = {name: sum(row["split"] == name for row in background_rows)
                     for name in BACKGROUND_SPLITS}
        _write_json(stage / "data/manifest.json", {
            "records": rows,
            "summary": {"counts": film_counts, "scope": "user_pre_split_no_patient_inference",
                        "source_modified": False, "provenance_status": provenance_record["status"]},
            "failed": [],
        })
        _write_json(stage / "data/split.json", split_indices)
        _write_json(stage / "data/backgrounds-manifest.json", {
            "version": "rectiray-local-presplit-v1", "rows": background_rows,
            "counts": bg_counts, "provenance_status": provenance_record["status"],
        })
        _write_json(stage / "data/provenance.local.json", provenance_record)
        report = validate_bundle(stage, stage / "backgrounds")
        for path, expected in original_bindings:
            if sha256_file(path) != expected:
                raise AssetError("An input file changed before atomic publication")
        report.update({"source_modified": False,
                       "provenance_status": provenance_record["status"],
                       "scope": "user-supplied pre-split files; no patient grouping inferred"})
        _write_json(stage / "data/audit.local.json", report)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output appeared during preparation")
        _publish_new_directory(stage, output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    args = parser.parse_args()
    result = prepare_data(args.sources, args.backgrounds, args.output, args.provenance)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
