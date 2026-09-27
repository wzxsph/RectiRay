"""Synthetic-only checks for the local pre-split asset packager."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

from PIL import Image, ImageDraw
import numpy as np
import pytest

from tools import data_assets, prepare_data


def picture(path: Path, color: tuple[int, int, int], *, fmt: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (40, 36), color)
    ImageDraw.Draw(image).rectangle((3, 4, 17, 13), fill=tuple(255 - v for v in color))
    image.save(path, format=fmt)


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    sources = tmp_path / "raw-films"
    backgrounds = tmp_path / "raw-backgrounds"
    picture(sources / "train" / "sub" / "a.png", (35, 67, 91))
    picture(sources / "dev" / "b.png", (103, 47, 159))
    (sources / "quarantine").mkdir(parents=True)
    picture(backgrounds / "train" / "a.png", (11, 177, 91))
    picture(backgrounds / "dev" / "b.png", (209, 73, 30))
    return sources, backgrounds


def test_data_only_decoder_matches_training_decoder_without_path_mutation(
        tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    picture(rgb_path, (35, 67, 91))
    integer_path = tmp_path / "integer.tiff"
    Image.fromarray(np.arange(40 * 36, dtype=np.uint16).reshape(36, 40)).save(integer_path)
    rotated_path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.open(rgb_path).save(rotated_path, exif=exif)
    sys.path.insert(0, str(prepare_data.REPO / "src"))
    from ptrn_hip.data import read_rgb
    for path in (rgb_path, integer_path, rotated_path):
        before = list(sys.path)
        width, height, pixel_sha = data_assets.decoded_rgb(path)
        assert sys.path == before
        reference = read_rgb(path)
        assert (width, height) == (reference.shape[1], reference.shape[0])
        assert pixel_sha == hashlib.sha256(reference.tobytes()).hexdigest()


@pytest.fixture
def output_parent():
    local = prepare_data.REPO / ".local"
    local.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-data-test-", dir=local) as temporary:
        yield Path(temporary)


def test_packages_all_splits_loads_partial_dataset_and_preserves_bytes(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    picture(sources / "quarantine" / "c.png", (160, 100, 20))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "sources": [{"identifier": "test-films", "license": "synthetic fixture"}],
        "backgrounds": [{"identifier": "test-bg", "license": "synthetic fixture"}],
    }))
    output = output_parent / "bundle"
    report = prepare_data.prepare_data(sources, backgrounds, output, provenance)
    assert report["split_counts"] == {"train": 1, "dev": 1, "quarantine": 1}
    assert report["background_counts"] == {"train": 1, "dev": 1}
    assert report["all_film_records_checked"] == 3
    assert report["all_background_rows_checked"] == 2
    assert report["provenance_status"] == "provided"
    manifest = json.loads((output / "data/manifest.json").read_text())
    split = json.loads((output / "data/split.json").read_text())
    backgrounds_manifest = json.loads((output / "data/backgrounds-manifest.json").read_text())
    assert split == {"train": [0], "dev": [1], "quarantine": [2]}
    assert all("a.png" not in row["source_path"] and "b.png" not in row["source_path"]
               for row in manifest["records"])
    for original, row in zip([sources / "train/sub/a.png", sources / "dev/b.png",
                              sources / "quarantine/c.png"], manifest["records"], strict=True):
        assert original.read_bytes() == (output / row["source_path"]).read_bytes()
    for original, row in zip([backgrounds / "train/a.png", backgrounds / "dev/b.png"],
                             backgrounds_manifest["rows"], strict=True):
        assert original.read_bytes() == (output / "backgrounds" / row["path"]).read_bytes()
    validated = data_assets.validate_bundle(output, output / "backgrounds")
    assert all(report[key] == value for key, value in validated.items())
    sys.path.insert(0, str(prepare_data.REPO / "src"))
    from ptrn_hip.partial_data import PartialDataset
    for split_name in ("train", "dev"):
        dataset = PartialDataset(output, output / "backgrounds", split=split_name,
                                 views=1, verify=True)
        assert len(dataset) == 1
    # PartialDataset reads only train/dev assets; the separate verifier must
    # still catch damage in an otherwise ignored quarantine film.
    quarantine_asset = output / manifest["records"][2]["source_path"]
    quarantine_asset.write_bytes(b"damaged quarantine")
    with pytest.raises(data_assets.AssetError, match="Film asset SHA256 mismatch"):
        data_assets.validate_bundle(output, output / "backgrounds")


def test_rejects_cross_split_byte_and_pixel_duplicates_without_partial_output(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    output = output_parent / "byte-duplicate"
    (sources / "dev/b.png").write_bytes((sources / "train/sub/a.png").read_bytes())
    with pytest.raises(data_assets.AssetError, match="duplicate crosses"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert not output.exists()
    assert not list(output_parent.glob(".byte-duplicate.staging-*"))
    picture(sources / "dev/b.png", (103, 47, 159))
    picture(sources / "dev/b.bmp", (35, 67, 91), fmt="BMP")
    output = output_parent / "pixel-duplicate"
    with pytest.raises(data_assets.AssetError, match="duplicate crosses"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert not output.exists()
    (sources / "dev/b.bmp").unlink()
    picture(backgrounds / "dev/b.bmp", (11, 177, 91), fmt="BMP")
    output = output_parent / "background-pixel-duplicate"
    with pytest.raises(data_assets.AssetError, match="duplicate crosses"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert not output.exists()


def test_quarantine_and_same_split_duplicates_are_retained_and_reported(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    original = (sources / "train/sub/a.png").read_bytes()
    (sources / "train/copy.png").write_bytes(original)
    (sources / "quarantine/copy.png").write_bytes(original)
    output = output_parent / "bundle"
    report = prepare_data.prepare_data(sources, backgrounds, output)
    assert report["split_counts"] == {"train": 2, "dev": 1, "quarantine": 1}
    for name in ("films_file", "films_pixel"):
        assert report["duplicates"][name]["same_split_extra_copies"] == 1
        assert report["duplicates"][name]["quarantine_overlap_groups"] == 1
    assert report["provenance_status"] == "unknown"


def test_repeated_preparation_is_deterministic_and_does_not_modify_inputs(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    picture(sources / "train" / "z.png", (21, 42, 84))
    originals = {path: path.read_bytes()
                 for root in (sources, backgrounds)
                 for path in root.rglob("*") if path.is_file()}
    first, second = output_parent / "first", output_parent / "second"
    first_report = prepare_data.prepare_data(sources, backgrounds, first)
    second_report = prepare_data.prepare_data(sources, backgrounds, second)
    assert first_report == second_report
    first_files = {path.relative_to(first): path.read_bytes()
                   for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes()
                    for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_invalid_image_symlink_output_and_traversal_are_rejected(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    bad = sources / "train" / "broken.png"
    bad.write_bytes(b"not an image")
    output = output_parent / "invalid"
    with pytest.raises(data_assets.AssetError, match="decode"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert not output.exists()
    assert not list(output_parent.glob(".invalid.staging-*"))
    bad.unlink()
    (sources / "train" / "linked.png").symlink_to(sources / "dev/b.png")
    with pytest.raises(data_assets.AssetError, match="symlink"):
        prepare_data.prepare_data(sources, backgrounds, output_parent / "linked")
    (sources / "train" / "linked.png").unlink()
    alias = tmp_path / "source-alias"
    alias.symlink_to(sources, target_is_directory=True)
    with pytest.raises(data_assets.AssetError, match="symlink"):
        prepare_data.prepare_data(alias, backgrounds, output_parent / "source-alias")
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(data_assets.AssetError, match="symlink"):
        prepare_data.prepare_data(parent_alias / sources.name, backgrounds,
                                  output_parent / "ancestor-alias")
    quarantine = sources / "quarantine"
    quarantine.rmdir()
    quarantine.symlink_to(tmp_path / "missing-quarantine", target_is_directory=True)
    with pytest.raises(data_assets.AssetError, match="symlink"):
        prepare_data.prepare_data(sources, backgrounds,
                                  output_parent / "dangling-quarantine")
    quarantine.unlink()
    quarantine.mkdir()
    provenance_dir = tmp_path / "provenance-source"
    provenance_dir.mkdir()
    (provenance_dir / "origin.json").write_text(json.dumps({
        "sources": [], "backgrounds": [],
    }))
    provenance_alias = tmp_path / "provenance-alias"
    provenance_alias.symlink_to(provenance_dir, target_is_directory=True)
    with pytest.raises(data_assets.AssetError, match="symlink"):
        prepare_data.prepare_data(sources, backgrounds,
                                  output_parent / "provenance-alias",
                                  provenance_alias / "origin.json")
    with pytest.raises(data_assets.AssetError, match=".local"):
        prepare_data.prepare_data(sources, backgrounds, tmp_path / "outside")
    existing = output_parent / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        prepare_data.prepare_data(sources, backgrounds, existing)
    output = output_parent / "bundle"
    prepare_data.prepare_data(sources, backgrounds, output)
    manifest_path = output / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["source_path"] = "../escape.png"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(data_assets.AssetError, match="Unsafe"):
        data_assets.validate_bundle(output, output / "backgrounds")


def test_validator_rejects_extra_split_and_missing_or_duplicate_background_ids(
        tmp_path: Path, output_parent: Path) -> None:
    sources, backgrounds = inputs(tmp_path)
    output = output_parent / "bundle"
    prepare_data.prepare_data(sources, backgrounds, output)
    split_path = output / "data/split.json"
    original_splits = json.loads(split_path.read_text())
    split_path.write_text(json.dumps({**original_splits, "unused": []}))
    with pytest.raises(data_assets.AssetError, match="split manifest"):
        data_assets.validate_bundle(output, output / "backgrounds")
    split_path.write_text(json.dumps(original_splits))

    bg_path = output / "data/backgrounds-manifest.json"
    original_backgrounds = json.loads(bg_path.read_text())
    without_id = json.loads(json.dumps(original_backgrounds))
    del without_id["rows"][1]["id"]
    bg_path.write_text(json.dumps(without_id))
    with pytest.raises(data_assets.AssetError, match="Background IDs"):
        data_assets.validate_bundle(output, output / "backgrounds")
    duplicate_id = json.loads(json.dumps(original_backgrounds))
    duplicate_id["rows"][1]["id"] = duplicate_id["rows"][0]["id"]
    bg_path.write_text(json.dumps(duplicate_id))
    with pytest.raises(data_assets.AssetError, match="Background IDs"):
        data_assets.validate_bundle(output, output / "backgrounds")


def test_atomic_publish_refuses_destination_that_appears_during_preparation(
        tmp_path: Path, output_parent: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources, backgrounds = inputs(tmp_path)
    output = output_parent / "arrived"
    original_publish = prepare_data._publish_new_directory

    def create_racing_destination(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("another writer")
        original_publish(stage, destination)

    monkeypatch.setattr(prepare_data, "_publish_new_directory", create_racing_destination)
    with pytest.raises(FileExistsError, match="appeared"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert (output / "owner.txt").read_text() == "another writer"
    assert not list(output_parent.glob(".arrived.staging-*"))


def test_source_mutation_before_publish_and_legacy_background_row(
        tmp_path: Path, output_parent: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources, backgrounds = inputs(tmp_path)
    original_validate = prepare_data.validate_bundle

    def mutate_after_staging(bundle: Path, background_root: Path):
        result = original_validate(bundle, background_root)
        picture(sources / "train/sub/a.png", (10, 20, 30))
        return result

    monkeypatch.setattr(prepare_data, "validate_bundle", mutate_after_staging)
    output = output_parent / "mutated"
    with pytest.raises(data_assets.AssetError, match="changed"):
        prepare_data.prepare_data(sources, backgrounds, output)
    assert not output.exists()
    monkeypatch.setattr(prepare_data, "validate_bundle", original_validate)
    output = output_parent / "bundle"
    prepare_data.prepare_data(sources, backgrounds, output)
    legacy_file = output / "backgrounds" / "train" / "viewer.png"
    picture(legacy_file, (13, 14, 15))
    bg_manifest = output / "data/backgrounds-manifest.json"
    raw = json.loads(bg_manifest.read_text())
    raw["rows"].append({"id": "viewer-legacy", "path": "train/viewer.png",
                        "split": "train", "source_kind": "procedural_interface",
                        "sha256": data_assets.sha256_file(legacy_file),
                        "width": 40, "height": 36})
    raw.pop("counts", None)  # historical manifests need not have aggregate counts
    bg_manifest.write_text(json.dumps(raw))
    report = data_assets.validate_bundle(output, output / "backgrounds")
    assert report["all_background_rows_checked"] == 3
    assert report["background_counts"] == {"train": 2, "dev": 1}
    legacy_file.write_bytes(b"damaged")
    with pytest.raises(data_assets.AssetError, match="SHA256 mismatch"):
        data_assets.validate_bundle(output, output / "backgrounds")
