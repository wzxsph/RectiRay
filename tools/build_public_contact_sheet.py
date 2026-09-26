"""Build bilingual 100-view contact sheets from the six audited public films.

The six source ordinals, privacy strips and license audit are defined in
build_public_examples.py and docs/figures/examples/CREDITS.md. Source paths,
source filenames, weight files and the detailed audit remain local. Every
numbered slot is fixed before inference; a failed slot is shown, not replaced.

Run with KERAS_BACKEND=torch and PYTHONPATH=src on a CUDA machine::

    python tools/build_public_contact_sheet.py \
      --av-root "$AV_V2_RAW" --fraiwan-root "$FRAIWAN_V2_IMAGES" \
      --bundle "$LOCAL_BUNDLE" --branch "$BRANCH_PLUS2250" \
      --output-en docs/figures/examples/contact-sheet-100.png \
      --output-zh docs/figures/examples/zh-CN/contact-sheet-100.png \
      --local-report .local/contact-sheet100/audit.local.json

No private phone photograph, case annotation or COCO photo is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_public_examples as public


REPO = Path(__file__).resolve().parents[1]
SCENE_SEED_BASE = 2026092700
SCENE_RETRY_STRIDE = 104729
SCENE_MAX_ATTEMPTS = 12
GEOMETRY_RETRY_ERRORS = frozenset({
    "Unable to satisfy requested missing-area stratum",
    "Near-full framing is geometrically infeasible",
    "Near-full framing would remove source content",
    "Requested occupancy stratum infeasible after framing",
    "Actual missing fraction outside requested stratum",
    "Visible occupancy outside requested stratum",
})
SIZE = (3440, 1920)
BACKGROUND = "#faf9f6"
INK = "#172b35"
TEAL = "#087e8b"
CORAL = "#d56a46"
TILE_WIDTH = 158
TILE_HEIGHT = 158
TILE_GAP_X = 6
TILE_GAP_Y = 5
GRID_TOP = 220
GRID_WIDTH = 10 * TILE_WIDTH + 9 * TILE_GAP_X


@dataclass(frozen=True)
class Category:
    key: str
    count: int
    mark: str
    options: dict


CATEGORIES = (
    Category("keep", 20, "K", dict(kind="keep", integrity="complete")),
    Category("near_full", 16, "F", dict(
        kind="photo", integrity="complete", family="viewer_screen",
        mode="mild", occupancy_range=(.90, .98),
        capture_material="film", capture_severity="light")),
    Category("mild_perspective", 16, "P", dict(
        kind="photo", integrity="complete", family="mixed_color",
        mode="mild", occupancy_range=(.75, .90),
        capture_material="film", capture_severity="clear")),
    Category("missing_corner", 16, "C", dict(
        kind="photo", integrity="light", family="viewer_screen",
        mode="mild", occupancy_range=(.50, .75),
        missing_range=(.02, .10), missing_target=.06,
        capture_material="film", capture_severity="light")),
    Category("near_color", 16, "B", dict(
        kind="photo", integrity="complete", family="near_color",
        mode="mild", occupancy_range=(.50, .75),
        capture_material="film", capture_severity="light",
        edge_blend_px224=1.5)),
    Category("screen", 16, "S", dict(
        kind="photo", integrity="complete", family="viewer_screen",
        mode="mild", occupancy_range=(.75, .90),
        capture_material="screen", capture_severity="strong")),
)


@dataclass(frozen=True)
class Scene:
    number: int
    category: Category
    source_slot: int
    seed: int


def scenes() -> tuple[Scene, ...]:
    result = []
    for category in CATEGORIES:
        for category_index in range(category.count):
            number = len(result) + 1
            # The same six previously reviewed public radiographs rotate through
            # the strata. Near-full screen framing is infeasible for the three
            # square Fraiwan films, so that stratum uses the audited AV films.
            # Keep views intentionally repeat trusted originals.
            source_slot = ((0, 2, 4)[category_index % 3] if category.key == "near_full"
                           else (number - 1) % len(public.CASES))
            result.append(Scene(number, category, source_slot,
                                SCENE_SEED_BASE + number))
    assert len(result) == 100
    return tuple(result)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pixel_sha(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes(order="C")).hexdigest()


def preflight(bundle: Path, branch: Path) -> None:
    if not public.torch.cuda.is_available():
        raise RuntimeError("The fixed +2250 comparison requires CUDA inference")
    if sha256(branch) != public.BRANCH_SHA256:
        raise ValueError("Branch checkpoint SHA256 differs from the reviewed +2250 weight")
    renderer = REPO / "src/ptrn_hip/partial_geometry.py"
    renderer_sha = "a23bb919dd5cea1437477b0cb6478247e6bf9735599962787d7b48e42ef92549"
    if sha256(renderer) != renderer_sha:
        raise ValueError("Reviewed V3 renderer bytes changed")
    checkpoint = public.torch.load(branch, map_location="cpu", weights_only=True)
    if checkpoint.get("contract") != public.CONTRACT:
        raise ValueError("Checkpoint geometry contract changed")
    public.validate_code_binding(checkpoint, REPO, allow_renderer_update=True,
                                 allow_parent_source_updates=True)
    public.validate_asset_binding(checkpoint, bundle)


def load_sources(av_root: Path, fraiwan_root: Path):
    paths = {"av": public.source_paths(av_root, 2417),
             "fr": public.source_paths(fraiwan_root, 354)}
    rgb_sources = []
    audit = []
    for source in public.CASES:
        file = paths[source.dataset][source.ordinal]
        rgb = public.safe_source(file, source.dataset)
        rgb_sources.append(rgb)
        audit.append({"dataset": "AV-DDH v2" if source.dataset == "av" else "Fraiwan v2",
                      "ordinal": source.ordinal,
                      "source_file_sha256": sha256(file),
                      "sanitized_rgb_sha256": pixel_sha(rgb),
                      "sanitized_size": [rgb.shape[1], rgb.shape[0]]})
    return tuple(rgb_sources), audit


def synthesize(scene: Scene, sources: tuple[np.ndarray, ...]):
    """Resolve geometry deterministically before any checkpoint prediction.

    This only retries a scene-generation RuntimeError. The first feasible
    sample is retained irrespective of what the model will later predict.
    """
    source_case = public.CASES[scene.source_slot]
    last_error = None
    for attempt in range(SCENE_MAX_ATTEMPTS):
        seed = scene.seed + attempt * SCENE_RETRY_STRIDE
        options = dict(scene.category.options)
        if options["kind"] == "photo":
            options["capture_seed"] = seed + 1000
            options["occupancy_bin_v2"] = 0
        try:
            view = public.sample(sources[scene.source_slot], source_case.ordinal,
                                 scene.number - 1, seed=seed, domain="train",
                                 scene_version=public.STEP2000_SCENE, **options)
            return view, attempt, None
        except RuntimeError as error:
            if str(error) not in GEOMETRY_RETRY_ERRORS:
                raise
            last_error = error
    return None, SCENE_MAX_ATTEMPTS, last_error


def checkerboard(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "#e9eeed")
    draw = ImageDraw.Draw(image)
    step = 12
    for y in range(0, height, step):
        for x in range(0, width, step):
            if (x // step + y // step) % 2:
                draw.rectangle((x, y, min(x + step - 1, width - 1),
                                min(y + step - 1, height - 1)), fill="#d5e2e1")
    return image


def thumbnail(rgb: np.ndarray | None, mask: np.ndarray | None = None,
              *, failure: str | None = None) -> Image.Image:
    width, height = 150, 130
    card = checkerboard(width, height)
    if rgb is None:
        draw = ImageDraw.Draw(card)
        draw.rounded_rectangle((9, 41, width - 9, 89), radius=6,
                               fill="#fff1eb", outline=CORAL, width=2)
        draw.text((18, 56), failure or "Unavailable", font=font(12), fill=INK)
        return card
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    if mask is not None and (mask.shape != rgb.shape[:2] or mask.dtype != np.uint8):
        raise ValueError("Output mask shape or dtype differs")
    image = Image.fromarray(rgb, "RGB")
    if mask is not None:
        image.putalpha(Image.fromarray(mask, "L"))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    at = ((width - image.width) // 2, (height - image.height) // 2)
    if image.mode == "RGBA":
        card.paste(image, at, image)
    else:
        card.paste(image, at)
    return card


def font(size: int, *, chinese: bool = False) -> ImageFont.FreeTypeFont:
    if chinese:
        if public._ZH_FONT is None:
            raise RuntimeError("Chinese font was not resolved")
        return ImageFont.truetype(public._ZH_FONT[0], size,
                                  index=public._ZH_FONT[1])
    return ImageFont.truetype(public.FONT, size)


def fit_caption(draw: ImageDraw.ImageDraw, xy, value: str,
                *, size: int, max_width: int, chinese: bool, fill=INK) -> None:
    face = font(size, chinese=chinese)
    box = draw.textbbox((0, 0), value, font=face)
    if box[2] - box[0] > max_width:
        raise ValueError(f"Caption exceeds {max_width} pixels")
    draw.text(xy, value, font=face, fill=fill)


def sheet(inputs: list[Image.Image], outputs: list[Image.Image],
          fixed: tuple[Scene, ...], *, chinese: bool) -> Image.Image:
    if len(inputs) != 100 or len(outputs) != 100 or len(fixed) != 100:
        raise ValueError("All 100 input/output slots are required")
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    fit_caption(draw, (32, 21), "RectiRay", size=42, max_width=800,
                chinese=False, fill=INK)
    subtitle = ("六张公开髋部 X 光片 · 100 个固定场景" if chinese else
                "100 fixed scenes from six public hip radiographs")
    fit_caption(draw, (32, 79), subtitle, size=25, max_width=1700,
                chinese=chinese, fill=INK)
    context = ("20 次可信原片直通（底片重复） · 80 次模拟拍摄 · 分支 +2250" if chinese else
               "20 trusted-original references (repeated) · 80 simulated captures · branch +2250")
    fit_caption(draw, (32, 119), context, size=18, max_width=2400,
                chinese=chinese, fill="#50636a")
    draw.rounded_rectangle((32, 163, 1704, 1859), radius=14,
                           fill="#f2f5f3", outline="#c8d7d5", width=2)
    draw.rounded_rectangle((1736, 163, 3408, 1859), radius=14,
                           fill="#f2f5f3", outline="#c8d7d5", width=2)
    left_label = "输入 · 原片或模拟拍摄照片" if chinese else "INPUT · source or simulated photograph"
    right_label = ("输出 · +2250 展平／可信原片旁路" if chinese else
                   "OUTPUT · +2250 rectification / trusted keep")
    fit_caption(draw, (50, 178), left_label, size=19, max_width=1600,
                chinese=chinese, fill=TEAL)
    fit_caption(draw, (1754, 178), right_label, size=19, max_width=1600,
                chinese=chinese, fill=TEAL)
    for panel_x, pictures in ((32, inputs), (1736, outputs)):
        grid_x = panel_x + (1672 - GRID_WIDTH) // 2
        for scene, picture in zip(fixed, pictures, strict=True):
            slot = scene.number - 1
            x = grid_x + (slot % 10) * (TILE_WIDTH + TILE_GAP_X)
            y = GRID_TOP + (slot // 10) * (TILE_HEIGHT + TILE_GAP_Y)
            color = CORAL if scene.category.key == "missing_corner" else TEAL
            draw.rounded_rectangle((x, y, x + TILE_WIDTH - 1, y + TILE_HEIGHT - 1),
                                   radius=6, fill="#ffffff", outline="#cddbd8", width=1)
            fit_caption(draw, (x + 6, y + 2), f"{scene.number:03d}", size=11,
                        max_width=39, chinese=False, fill=INK)
            fit_caption(draw, (x + 137, y + 2), scene.category.mark, size=11,
                        max_width=17, chinese=False, fill=color)
            if picture.size != (150, 130):
                raise ValueError("Thumbnail size differs")
            canvas.paste(picture, (x + 4, y + 23))
    footer = ("K 原片直通 20 · F 近满幅 16 · P 轻度透视 16 · C 缺角 16 · B 近色融边 16 · S 屏幕干扰 16" if chinese else
              "K keep 20 · F near-full 16 · P mild perspective 16 · C missing corner 16 · B near-color 16 · S screen 16")
    fit_caption(draw, (32, 1871), footer, size=16, max_width=2600,
                chinese=chinese, fill="#50636a")
    caveat = ("非独立实拍验证" if chinese else "Not independent real-photo validation")
    face = font(16, chinese=chinese)
    width = draw.textbbox((0, 0), caveat, font=face)[2]
    fit_caption(draw, (SIZE[0] - width - 32, 1871), caveat, size=16,
                max_width=760, chinese=chinese, fill=CORAL)
    return canvas


def save_plain_png(image: Image.Image, output: Path) -> str:
    if output.exists():
        raise FileExistsError("Contact sheet output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".tmp")
    image.save(pending, format="PNG", compress_level=9)
    with Image.open(pending) as check:
        if check.mode != "RGB" or check.size != SIZE or check.info or check.getexif():
            raise ValueError("Contact sheet dimensions, color mode or metadata differ")
    os.replace(pending, output)
    return sha256(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-root", type=Path, required=True)
    parser.add_argument("--fraiwan-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--branch", type=Path, required=True)
    parser.add_argument("--output-en", type=Path, required=True)
    parser.add_argument("--output-zh", type=Path, required=True)
    parser.add_argument("--local-report", type=Path, required=True)
    parser.add_argument("--zh-font", type=Path)
    parser.add_argument("--zh-font-index", type=int)
    args = parser.parse_args()
    if args.output_en.resolve() == args.output_zh.resolve():
        parser.error("English and Chinese outputs must differ")
    if not args.local_report.resolve().is_relative_to(REPO / ".local"):
        parser.error("Audit report must stay under the ignored .local directory")
    for output in (args.output_en, args.output_zh, args.local_report):
        if output.exists():
            raise FileExistsError("Output already exists; preserve the existing audit")
    public.import_training_dependencies()
    public._ZH_FONT = public.resolve_zh_font(args.zh_font, args.zh_font_index)
    preflight(args.bundle.resolve(), args.branch.resolve())
    sources, source_audit = load_sources(args.av_root.resolve(),
                                         args.fraiwan_root.resolve())
    fixed = scenes()  # Selection is locked before the first model prediction.
    public.torch.set_num_threads(4)
    input_thumbs = []
    output_thumbs = []
    status = Counter()
    retry_histogram = Counter()
    realized_occupancy = []
    realized_missing = []
    failure_slots = []
    views = []
    # Freeze all 100 realized views before the first model output. Feasibility
    # retries cannot be influenced by a checkpoint result.
    for scene in fixed:
        try:
            view, retries, error = synthesize(scene, sources)
            if view is None:
                raise error or RuntimeError("No feasible synthetic view")
        except Exception as error:
            # The frozen recipe cannot replace a difficult scene. Keep its
            # numbered slot visible in both panels and record the failure.
            input_thumbs.append(thumbnail(None, failure="Scene failed"))
            views.append(None)
            status["scene_failed"] += 1
            failure_slots.append({"slot": scene.number, "stage": "scene",
                                  "error_type": type(error).__name__})
            print(f"{scene.number:03d}/100 scene failed: {type(error).__name__}", flush=True)
            continue
        retry_histogram[retries] += 1
        views.append(view)
        input_thumbs.append(thumbnail(view.photo))
        realized_occupancy.append(float(view.info["occupancy"]))
        realized_missing.append(float(view.info.get("missing_fraction", 0.)))
        print(f"prepared {scene.number:03d}/100 {scene.category.key} retries={retries}", flush=True)
    model, _ = public.load_model(args.bundle.resolve(), args.branch.resolve())
    model.eval()
    for scene, view in zip(fixed, views, strict=True):
        if view is None:
            output_thumbs.append(thumbnail(None, failure="Scene failed"))
            continue
        try:
            _, _, result, reason = public.predict_and_render(
                model, view, trusted_original=scene.category.key == "keep")
            if result is None:
                raise ValueError(reason or "No rectified result")
            output_thumbs.append(thumbnail(result.pixels, result.valid_mask))
            status["trusted_keep" if scene.category.key == "keep" else "rectified"] += 1
        except Exception as error:
            output_thumbs.append(thumbnail(None, failure="Output failed"))
            status["output_failed"] += 1
            failure_slots.append({"slot": scene.number, "stage": "output",
                                  "error_type": type(error).__name__})
        print(f"inferred {scene.number:03d}/100 {scene.category.key}", flush=True)
    en = sheet(input_thumbs, output_thumbs, fixed, chinese=False)
    zh = sheet(input_thumbs, output_thumbs, fixed, chinese=True)
    # Captions differ, but every numbered picture rectangle is identical.
    for x in (32, 1736):
        grid_x = x + (1672 - GRID_WIDTH) // 2
        for number in range(100):
            xx = grid_x + (number % 10) * (TILE_WIDTH + TILE_GAP_X) + 4
            yy = GRID_TOP + (number // 10) * (TILE_HEIGHT + TILE_GAP_Y) + 23
            box = (xx, yy, xx + 150, yy + 130)
            if en.crop(box).tobytes() != zh.crop(box).tobytes():
                raise ValueError("Localization changed a picture pixel")
    digest_en = save_plain_png(en, args.output_en)
    digest_zh = save_plain_png(zh, args.output_zh)
    report = {
        "scope": "public six-source, fixed 100-scene illustration; not independent validation",
        "branch_sha256": public.BRANCH_SHA256,
        "renderer_sha256": sha256(REPO / "src/ptrn_hip/partial_geometry.py"),
        "source_audit": source_audit,
        "seed_base": SCENE_SEED_BASE,
        "retry_stride": SCENE_RETRY_STRIDE,
        "max_scene_attempts": SCENE_MAX_ATTEMPTS,
        "retry_histogram": dict(sorted(retry_histogram.items())),
        "source_rotation": "six audited CASES; near-full uses AV slots 0/2/4",
        "categories": dict((item.key, item.count) for item in CATEGORIES),
        "status": dict(status),
        "failure_slots": failure_slots,
        "occupancy_min": min(realized_occupancy) if realized_occupancy else None,
        "occupancy_max": max(realized_occupancy) if realized_occupancy else None,
        "missing_max": max(realized_missing) if realized_missing else None,
        "en_sha256": digest_en,
        "zh_sha256": digest_zh,
        "picture_rectangles_identical": True,
        "export_size": list(SIZE),
        "export_metadata_empty": True,
    }
    args.local_report.parent.mkdir(parents=True, exist_ok=True)
    args.local_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: value for k, value in report.items()
                      if k not in ("source_audit", "failure_slots")}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
