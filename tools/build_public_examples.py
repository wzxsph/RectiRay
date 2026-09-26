"""Build or localize six fixed, licensed-source PTRN research panels.

Run with the local AV-DDH v2 and Fraiwan v2 image roots, the audited private
bundle, and the two exact checkpoints. This tool never writes source filenames
or source-file metadata into the figure. Keep the original dataset trees local.
For --locale zh-CN, localize the six reviewed English panels already in Git;
that path needs only Pillow and a Noto Sans CJK font, not the source or GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("KERAS_BACKEND", "torch")

from PIL import Image, ImageDraw, ImageFont


PARENT_SHA256 = "016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535"
BRANCH_SHA256 = "35f0101260b19ee1484022de3a4ed40fd3ce2f4bdb0ef73f379c4fd458a53b3b"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
REPO = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_ZH_FONT: tuple[Path, int] | None = None

ZH_TITLES = {
    "01-near-full": "近满幅照片",
    "02-mild-perspective": "轻度透视",
    "03-missing-corner": "胶片缺角",
    "04-near-color": "近色融边",
    "05-screen": "屏幕干扰",
    "06-original": "原片直通",
}
ENGLISH_PANEL_SHA256 = {
    "01-near-full": "ea0f8f1d3257a4b60e723226fc872ee61126870922726867b70e2a5b4aa7f22a",
    "02-mild-perspective": "2b06eac1ca46c9db53723a8e8750aed731087de136750dcffe3069a5a2bf0a17",
    "03-missing-corner": "7dbca46fbd13d3c3538f372168bba63c97e60e798a8ecdc95efbf05f607dcb8c",
    "04-near-color": "8ef3e627ef7d277288e27185d7710ddc1b850208ea82b1c9d2ae2a95fd3fa2f1",
    "05-screen": "7796a34924d2683dc3eaf995435df0e83d22e3a349523b43e1c9dbab242717cb",
    "06-original": "0ee6da1a3227d2f9b7fd97e547e69fb30ad686415a204a096fe0a314e9190393",
}
ENGLISH_RATIO_LABELS = {
    "01-near-full": ("1.087", "1.110"),
    "02-mild-perspective": ("0.958", "0.981"),
    "03-missing-corner": ("1.193", "1.303"),
    "04-near-color": ("0.910", "0.978"),
    "05-screen": ("1.138", "1.166"),
}


def import_training_dependencies() -> None:
    """Keep the Chinese-only localization usable without CUDA or training assets."""
    global np, torch, sha256, validate_asset_binding, validate_code_binding
    global read_rgb, CONTRACT, rectify_visible, STEP2000_SCENE, sample
    global letterbox, project, inference, load_model
    import numpy as np
    import torch
    from compare_real_photos import sha256, validate_asset_binding, validate_code_binding
    from ptrn_hip.data import read_rgb
    from ptrn_hip.partial_geometry import CONTRACT, rectify_visible
    from ptrn_hip.partial_synthesis import STEP2000_SCENE, sample
    from ptrn_hip.ratio_synthesis import letterbox, project
    from ptrn_pipeline.evaluation import inference, load_model


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


def resolve_zh_font(explicit: Path | None, explicit_index: int | None) -> tuple[Path, int]:
    """Find a Noto Simplified Chinese face without binding the tool to one PC."""
    selected = explicit or (Path(os.environ["PTRN_CJK_FONT"])
                            if "PTRN_CJK_FONT" in os.environ else None)
    if selected is not None:
        if not selected.is_file():
            raise FileNotFoundError(f"Chinese font does not exist: {selected}")
        index = (2 if "NotoSansCJK" in selected.name and selected.suffix == ".ttc"
                 else 0) if explicit_index is None else explicit_index
        return selected, index
    if shutil.which("fc-match"):
        match = subprocess.run(
            ["fc-match", "-f", "%{file}\n%{index}\n",
             "Noto Sans CJK SC:lang=zh-cn"],
            capture_output=True, text=True, check=False)
        lines = match.stdout.splitlines() if match.returncode == 0 else []
        if len(lines) >= 2:
            path = Path(lines[0])
            if path.is_file() and ("NotoSansCJK" in path.name or "NotoSansSC" in path.name):
                return path, int(lines[1])
    for candidate, index in (
        (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), 2),
        (Path("/usr/local/share/fonts/NotoSansCJK-Regular.ttc"), 2),
        (Path.home() / "Library/Fonts/NotoSansCJK-Regular.ttc", 2),
        (Path("C:/Windows/Fonts/NotoSansSC-Regular.otf"), 0),
    ):
        if candidate.is_file():
            return candidate, index
    raise FileNotFoundError(
        "Chinese figures require Noto Sans CJK SC; pass --zh-font PATH "
        "(and --zh-font-index for a TTC face) or set PTRN_CJK_FONT")


def zh_font(size: int):
    if _ZH_FONT is None:
        raise RuntimeError("Chinese font was not initialized")
    return ImageFont.truetype(_ZH_FONT[0], size, index=_ZH_FONT[1])


def zh_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
            size: int, fill: str, max_width: int) -> None:
    face = zh_font(size)
    bounds = draw.textbbox((0, 0), value, font=face)
    if bounds[2] - bounds[0] > max_width:
        raise ValueError(f"Chinese figure label exceeds {max_width} px: {value}")
    draw.text(xy, value, font=face, fill=fill)


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


def panel_zh(case: Case, english: Image.Image) -> Image.Image:
    """Replace captions while copying all three reviewed picture areas exactly."""
    canvas = Image.new("RGB", (1700, 720), "#101b29")
    draw = ImageDraw.Draw(canvas)
    zh_text(draw, (35, 26), f"{case.slug[:2]}  {ZH_TITLES[case.slug]}",
            29, "#f2f6f9", 1630)
    zh_text(draw, (35, 70), "固定种子合成示例 · 两个权重使用同一输入",
            17, "#adc4d2", 1630)
    positions = (35, 585, 1135)
    labels = ("输入图及预测四角", "父模型 · 第 2000 步", "分支 · 新增 2250 步")
    for x, label in zip(positions, labels, strict=True):
        draw.rounded_rectangle((x, 110, x + 530, 680), radius=14,
                               fill="#1c2d3d", outline="#5e7588", width=2)
        zh_text(draw, (x + 18, 130), label, 19, "#f0f6f8", 494)
    for x in (54, 604, 1154):
        area = (x, 170, x + 492, 635)
        canvas.paste(english.crop(area), (x, 170))
        if canvas.crop(area).tobytes() != english.crop(area).tobytes():
            raise ValueError("Chinese panel changed a reviewed picture pixel")
    if case.options["kind"] == "keep":
        statuses = ("可信来源 · 逐像素原样保留",) * 2
    else:
        statuses = tuple(f"预测整片宽高比 = {ratio}"
                         for ratio in ENGLISH_RATIO_LABELS[case.slug])
    for x, status in zip((604, 1154), statuses, strict=True):
        zh_text(draw, (x + 4, 649), status, 15, "#d8e9f1", 502)
    zh_text(draw, (56, 649), "真值 白色 · 父模型 青色 · 分支 橙色",
            15, "#d8e9f1", 490)
    return canvas


def overview_zh(output: Path) -> None:
    """Lay out the six Chinese panels at the same README reading width."""
    row_height = 355
    overview = Image.new("RGB", (850, 68 + row_height * len(CASES)), "#101b29")
    draw = ImageDraw.Draw(overview)
    zh_text(draw, (14, 12), "PTRN V3 · 六例固定合成图", 23, "#f2f6f9", 822)
    zh_text(draw, (14, 43), "公开 CC BY 4.0 数据 · 研究示例", 13, "#adc4d2", 822)
    for index, case in enumerate(CASES):
        top = 68 + index * row_height
        zh_text(draw, (14, top + 2), f"{case.slug[:2]}  {ZH_TITLES[case.slug]}",
                20, "#f2f6f9", 822)
        draw.line((14, top + 32, 836, top + 32), fill="#5e7588", width=1)
        with Image.open(output / f"{case.slug}.png") as image:
            for x, source_x, label, key in ((14, 54, "输入图 + 四角", None),
                                           (292, 604, "父模型 · 2000 步", "parent"),
                                           (570, 1154, "分支 · +2250 步", "branch")):
                draw.rounded_rectangle((x, top + 42, x + 266, top + 344),
                                       radius=8, fill="#1c2d3d",
                                       outline="#5e7588", width=1)
                zh_text(draw, (x + 9, top + 51), label, 14, "#f0f6f8", 248)
                picture = image.crop((source_x, 170, source_x + 492, 635))
                picture = picture.resize((250, 236), Image.Resampling.LANCZOS)
                overview.paste(picture, (x + 8, top + 76))
                if key is None:
                    status = "真值白 · 父模型青 · 分支橙"
                elif case.options["kind"] == "keep":
                    status = "可信来源 · 逐像素保留"
                else:
                    ratio = ENGLISH_RATIO_LABELS[case.slug][0 if key == "parent" else 1]
                    status = f"预测宽高比 {ratio}"
                zh_text(draw, (x + 9, top + 320), status, 12, "#d8e9f1", 248)
    overview_path = output / "overview.webp"
    overview.save(overview_path, format="WEBP", lossless=True, method=6)
    with Image.open(overview_path) as verify:
        if verify.size != overview.size or verify.getexif():
            raise ValueError("Chinese overview dimensions or EXIF differ")
    encoded = overview_path.read_bytes()
    if encoded[:4] != b"RIFF" or encoded[8:16] != b"WEBPVP8L":
        raise ValueError("Chinese overview is not a simple lossless WebP")
    chunk_bytes = int.from_bytes(encoded[16:20], "little")
    if 20 + chunk_bytes + (chunk_bytes & 1) != len(encoded):
        raise ValueError("Chinese overview contains extra metadata chunks")
    print(f"overview.webp: {hashlib.sha256(encoded).hexdigest()}")


def localize_english_panels(reference_dir: Path, output: Path) -> None:
    """Preserve already reviewed public example pixels without fresh inference."""
    output.mkdir(parents=True)
    for case in CASES:
        reference = reference_dir / f"{case.slug}.png"
        if hashlib.sha256(reference.read_bytes()).hexdigest() != ENGLISH_PANEL_SHA256[case.slug]:
            raise ValueError(f"Reviewed English example changed: {case.slug}")
        with Image.open(reference) as english:
            if english.mode != "RGB" or english.size != (1700, 720) or english.info:
                raise ValueError(f"Reviewed English example metadata differs: {case.slug}")
            localized = panel_zh(case, english)
        result = output / f"{case.slug}.png"
        localized.save(result, format="PNG", compress_level=9)
        with Image.open(result) as verify:
            if verify.mode != "RGB" or verify.size != (1700, 720) or verify.info:
                raise ValueError(f"Chinese example metadata differs: {case.slug}")
        print(f"{case.slug}: {hashlib.sha256(result.read_bytes()).hexdigest()}")
    overview_zh(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-root", type=Path,
                        help="Local AV-DDH v2 raw image root")
    parser.add_argument("--fraiwan-root", type=Path,
                        help="Local Fraiwan DDH X-ray v2 image root")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--branch", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--locale", choices=("en", "zh-CN"), default="en",
                        help="Figure-label language; default keeps the original English output")
    parser.add_argument("--reference-dir", type=Path,
                        default=REPO / "docs/figures/examples",
                        help="Reviewed English panels used only for Chinese localization")
    parser.add_argument("--zh-font", type=Path,
                        help="Optional Noto Sans CJK SC font for Chinese figures")
    parser.add_argument("--zh-font-index", type=int,
                        help="Face index when --zh-font points to a TTC collection")
    args = parser.parse_args()
    if args.locale == "zh-CN":
        global _ZH_FONT
        _ZH_FONT = resolve_zh_font(args.zh_font, args.zh_font_index)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Example output directory must be empty")
    if args.locale == "zh-CN":
        localize_english_panels(args.reference_dir.resolve(), args.output)
        return
    missing = [name for name in ("av_root", "fraiwan_root", "bundle", "parent", "branch")
               if getattr(args, name) is None]
    if missing:
        parser.error("English generation requires " + ", ".join("--" + name.replace("_", "-")
                                                          for name in missing))
    import_training_dependencies()
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
