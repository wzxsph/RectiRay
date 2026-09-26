"""Build six fixed, licensed-source PTRN research panels.

Run with the local AV-DDH v2 and Fraiwan v2 image roots, the audited private
bundle, and the two exact checkpoints. This tool never writes source filenames
or source-file metadata into the figure. Keep the original dataset trees local.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from compare_real_photos import (
    sha256, validate_asset_binding, validate_code_binding,
)
from ptrn_hip.data import read_rgb
from ptrn_hip.partial_geometry import CONTRACT, rectify_visible
from ptrn_hip.partial_synthesis import STEP2000_SCENE, sample
from ptrn_hip.ratio_synthesis import letterbox, project
from ptrn_pipeline.evaluation import inference, load_model


PARENT_SHA256 = "016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535"
BRANCH_SHA256 = "35f0101260b19ee1484022de3a4ed40fd3ce2f4bdb0ef73f379c4fd458a53b3b"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
REPO = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass(frozen=True)
class Case:
    slug: str
    title: str
    dataset: str
    ordinal: int
    seed: int
    options: dict


# These source ordinals are in relative-path order within the stated versioned
# source tree. Original names stay in the local dataset, never in this repository.
CASES = (
    Case("01-near-full", "Near-full frame", "av", 1228, 20260926,
         dict(kind="photo", integrity="complete", family="viewer_screen",
              mode="mild", occupancy_range=(.90, .98),
              capture_material="film", capture_severity="light")),
    Case("02-mild-perspective", "Mild perspective", "fr", 135, 20260927,
         dict(kind="photo", integrity="complete", family="mixed_color",
              mode="mild", occupancy_range=(.75, .90),
              capture_material="film", capture_severity="clear")),
    Case("03-missing-corner", "Missing film corner", "av", 2022, 20260928,
         dict(kind="photo", integrity="light", family="viewer_screen",
              mode="mild", occupancy_range=(.50, .75),
              missing_range=(.02, .10), missing_target=.06,
              capture_material="film", capture_severity="light")),
    Case("04-near-color", "Near-color boundary", "fr", 244, 20260929,
         dict(kind="photo", integrity="complete", family="near_color",
              mode="mild", occupancy_range=(.50, .75),
              capture_material="film", capture_severity="light",
              edge_blend_px224=1.5)),
    Case("05-screen", "Screen interference", "av", 2346, 20260930,
         dict(kind="photo", integrity="complete", family="viewer_screen",
              mode="mild", occupancy_range=(.75, .90),
              capture_material="screen", capture_severity="strong")),
    Case("06-original", "Original retained", "fr", 276, 20261001,
         dict(kind="keep", integrity="complete")),
)


def source_paths(root: Path, expected_count: int) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError("Versioned public source tree is missing")
    paths = sorted((p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: p.relative_to(root).as_posix())
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} source images; found {len(paths)}")
    return paths


def safe_source(path: Path, dataset: str) -> np.ndarray:
    # Source files contain no EXIF in this audited local dataset. Refuse a
    # changed file with metadata rather than accidentally carrying it forward.
    with Image.open(path) as original:
        if original.getexif():
            raise ValueError("Selected source contains EXIF")
    rgb = read_rgb(path).copy()
    if dataset == "av":
        # The three selected AV images have only generic viewer footer labels.
        # Remove those two small corner strips before synthesis so no footer
        # text travels into the published panels. Anatomy is left untouched.
        rgb[-31:, :190] = 0
        rgb[-31:, -140:] = 0
    return rgb


def preflight(bundle: Path, checkpoints: tuple[Path, Path]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The audited model inference path requires CUDA")
    for path, expected in zip(checkpoints, (PARENT_SHA256, BRANCH_SHA256), strict=True):
        if sha256(path) != expected:
            raise ValueError("Checkpoint SHA256 differs from the fixed research comparison")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("contract") != CONTRACT:
            raise ValueError("Checkpoint contract differs")
        # The renderer-only update and two known parent training-source updates
        # are explicitly checked against the reviewed byte hashes in the
        # existing local comparison tool.
        validate_code_binding(checkpoint, REPO, allow_renderer_update=True,
                              allow_parent_source_updates=True)
        validate_asset_binding(checkpoint, bundle)


def checkerboard(width: int, height: int, size: int = 20) -> Image.Image:
    grid = Image.new("RGB", (width, height), "#d7dfe5")
    draw = ImageDraw.Draw(grid)
    for y in range(0, height, size):
        for x in range(0, width, size):
            if (x // size + y // size) % 2:
                draw.rectangle((x, y, x + size - 1, y + size - 1), fill="#aebec9")
    return grid


def fit_image(rgb: np.ndarray, width: int, height: int, alpha: np.ndarray | None = None) -> Image.Image:
    if alpha is None:
        image = Image.fromarray(rgb, "RGB")
    else:
        image = Image.fromarray(np.dstack((rgb, alpha)).astype(np.uint8), "RGBA")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    frame = checkerboard(width, height)
    left = (width - image.width) // 2
    top = (height - image.height) // 2
    if image.mode == "RGBA":
        frame.paste(image, (left, top), image)
    else:
        frame.paste(image, (left, top))
    return frame


def overlay_corners(rgb: np.ndarray, quads: dict[str, np.ndarray]) -> np.ndarray:
    picture = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(picture)
    colors = {"Ground truth": "#ffffff", "Parent 2000": "#38d4f2",
              "Branch +2250": "#ffad56"}
    for label, quad in quads.items():
        points = [(float(x), float(y)) for x, y in quad]
        draw.line(points + [points[0]], fill=colors[label], width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x-5, y-5, x+5, y+5), fill=colors[label],
                         outline="#122132", width=2)
    return np.asarray(picture)


def predict_and_render(model, view, *, trusted_original: bool = False) -> tuple[np.ndarray, float, object | None, str | None]:
    pixels, letterbox_matrix = letterbox(view.photo, 224)
    with torch.no_grad():
        pred = inference(model, [{"pixels": pixels}])
    quad = project(pred["quad"][0], np.linalg.inv(letterbox_matrix))
    ratio = float(pred["ratio"][0])
    if trusted_original:
        # Provenance is supplied by the case recipe. Predicted corners remain
        # visible for diagnosis but cannot affect the trusted keep decision.
        rendered = rectify_visible(view.photo, quad, ratio, max_side=768,
                                   trusted_original=True)
        if rendered.info.get("action") != "keep" or not rendered.info.get("pixel_exact"):
            raise ValueError("Trusted-original contract did not return a keep action")
        if not np.array_equal(rendered.pixels, view.photo) or not np.all(rendered.valid_mask == 255):
            raise ValueError("Trusted-original output differs from input pixels")
        return quad, ratio, rendered, None
    if not np.isfinite(quad).all() or not np.isfinite(ratio):
        return quad, ratio, None, "Non-finite prediction"
    try:
        rendered = rectify_visible(view.photo, quad, ratio, max_side=768)
        return quad, ratio, rendered, None
    except (ValueError, FloatingPointError, OverflowError) as error:
        # Keep the case and visibly report the failure.
        return quad, ratio, None, str(error)


def font(size: int):
    return ImageFont.truetype(FONT, size)


def panel(case: Case, view, predictions: dict) -> Image.Image:
    canvas = Image.new("RGB", (1700, 720), "#101b29")
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 26), f"{case.slug[:2]}  {case.title}", font=font(29), fill="#f2f6f9")
    draw.text((35, 70), "Fixed synthetic example · same input for both checkpoints",
              font=font(17), fill="#adc4d2")
    positions = (35, 585, 1135)
    labels = ("Input + predicted corners", "Parent · 2000 steps", "Branch · +2250 steps")
    for x, label in zip(positions, labels, strict=True):
        draw.rounded_rectangle((x, 110, x + 530, 680), radius=14,
                               fill="#1c2d3d", outline="#5e7588", width=2)
        draw.text((x + 18, 130), label, font=font(19), fill="#f0f6f8")
    input_rgb = overlay_corners(view.photo,
                                {"Ground truth": view.quad,
                                 "Parent 2000": predictions["parent"][0],
                                 "Branch +2250": predictions["branch"][0]})
    canvas.paste(fit_image(input_rgb, 492, 465), (54, 170))
    for x, key in ((604, "parent"), (1154, "branch")):
        quad, ratio, rendered, reason = predictions[key]
        if rendered is None:
            placeholder = checkerboard(492, 465)
            pd = ImageDraw.Draw(placeholder)
            pd.rounded_rectangle((25, 195, 467, 290), radius=9,
                                 fill="#fff2e0", outline="#ad6720", width=2)
            pd.text((42, 225), "Rectification unavailable", font=font(20), fill="#713d0a")
            canvas.paste(placeholder, (x, 170))
            status = reason or "Unknown geometry failure"
        else:
            canvas.paste(fit_image(rendered.pixels, 492, 465, rendered.valid_mask),
                         (x, 170))
            status = ("Trusted provenance · pixel-exact keep" if view.info["kind"] == "keep"
                      else f"Predicted full-plane W/H = {ratio:.3f}")
        draw.text((x + 4, 649), status[:46], font=font(15), fill="#d8e9f1")
    draw.text((56, 649), "GT white   ·   parent cyan   ·   branch orange",
              font=font(15), fill="#d8e9f1")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-root", required=True, type=Path,
                        help="Local AV-DDH v2 raw image root")
    parser.add_argument("--fraiwan-root", required=True, type=Path,
                        help="Local Fraiwan DDH X-ray v2 image root")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--branch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Example output directory must be empty")
    bundle = args.bundle.resolve()
    checkpoints = (args.parent.resolve(), args.branch.resolve())
    preflight(bundle, checkpoints)
    paths = {"av": source_paths(args.av_root, 2417),
             "fr": source_paths(args.fraiwan_root, 354)}
    views = []
    # Lock all six scenes before looking at any model result. A rendering
    # failure below cannot alter the chosen sources, seeds or scene options.
    for index, case in enumerate(CASES):
        source = safe_source(paths[case.dataset][case.ordinal], case.dataset)
        view = sample(source, case.ordinal, index, seed=case.seed, domain="train",
                      scene_version=STEP2000_SCENE, capture_seed=case.seed + 1000,
                      occupancy_bin_v2=0, **case.options)
        views.append(view)
    model, _ = load_model(bundle, checkpoints[0])
    model.eval()
    predictions = [{} for _ in CASES]
    for label, checkpoint in (("parent", checkpoints[0]), ("branch", checkpoints[1])):
        if label == "branch":
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
        for i, view in enumerate(views):
            predictions[i][label] = predict_and_render(
                model, view, trusted_original=CASES[i].options["kind"] == "keep")
    args.output.mkdir(parents=True)
    for case, view, result in zip(CASES, views, predictions, strict=True):
        output = args.output / f"{case.slug}.png"
        figure = panel(case, view, result)
        figure.save(output, format="PNG", compress_level=9)
        with Image.open(output) as verify:
            if verify.getexif() or "exif" in verify.info or verify.size != (1700, 720):
                raise ValueError("Output metadata or dimensions differ")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"{case.slug}: {digest} occupancy={view.info['occupancy']:.4f} "
              f"missing={view.info['missing_fraction']:.4f} "
              f"parent_rendered={result['parent'][2] is not None} "
              f"branch_rendered={result['branch'][2] is not None}")
    # Recompose the picture regions at an 850 px reading width. The overview
    # uses new native-size labels so they stay readable when embedded in README.
    row_height = 355
    overview = Image.new("RGB", (850, 68 + row_height * len(CASES)), "#101b29")
    overview_draw = ImageDraw.Draw(overview)
    overview_draw.text((14, 12), "PTRN V3 · six fixed synthetic views",
                       font=font(23), fill="#f2f6f9")
    overview_draw.text((14, 43), "Public CC BY 4.0 sources · illustrative research examples",
                       font=font(13), fill="#adc4d2")
    for index, case in enumerate(CASES):
        top = 68 + index * row_height
        overview_draw.text((14, top + 2), f"{case.slug[:2]}  {case.title}",
                           font=font(20), fill="#f2f6f9")
        overview_draw.line((14, top + 32, 836, top + 32), fill="#5e7588", width=1)
        with Image.open(args.output / f"{case.slug}.png") as image:
            for x, source_x, label, key in ((14, 54, "Input + corners", None),
                                           (292, 604, "Parent · 2000", "parent"),
                                           (570, 1154, "Branch · +2250", "branch")):
                overview_draw.rounded_rectangle((x, top + 42, x + 266, top + 344),
                                                 radius=8, fill="#1c2d3d",
                                                 outline="#5e7588", width=1)
                overview_draw.text((x + 9, top + 51), label,
                                   font=font(14), fill="#f0f6f8")
                picture = image.crop((source_x, 170, source_x + 492, 635))
                picture = picture.resize((250, 236), Image.Resampling.LANCZOS)
                overview.paste(picture, (x + 8, top + 76))
                if key is None:
                    status = "GT white · parent cyan · branch orange"
                elif case.options["kind"] == "keep":
                    status = "Trusted · pixel-exact keep"
                elif predictions[index][key][2] is None:
                    status = "Rectification unavailable"
                else:
                    status = f"Predicted W/H {predictions[index][key][1]:.3f}"
                overview_draw.text((x + 9, top + 320), status,
                                   font=font(12), fill="#d8e9f1")
    overview_path = args.output / "overview.webp"
    overview.save(overview_path, format="WEBP", lossless=True, method=6)
    with Image.open(overview_path) as verify:
        if verify.size != overview.size or verify.getexif():
            raise ValueError("Overview dimensions or EXIF differ")
    # A single VP8L chunk means there is no EXIF, XMP, ICC, animation or text
    # chunk. Pillow reports default loop/background values even for this static
    # one-chunk file; inspect the container bytes rather than that dictionary.
    encoded = overview_path.read_bytes()
    if encoded[:4] != b"RIFF" or encoded[8:16] != b"WEBPVP8L":
        raise ValueError("Overview is not a simple lossless WebP")
    chunk_bytes = int.from_bytes(encoded[16:20], "little")
    if 20 + chunk_bytes + (chunk_bytes & 1) != len(encoded):
        raise ValueError("Overview contains extra metadata chunks")
    print(f"overview.webp: {sha256(overview_path)}")


if __name__ == "__main__":
    main()
