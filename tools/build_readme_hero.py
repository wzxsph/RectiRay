"""Build the English and Chinese README teaser from reviewed public panels.

Only already-published synthetic example pixels are copied. No model, source
radiograph, private data bundle, or GPU is required. Set --zh-font on systems
without Noto Sans CJK SC installed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs/figures/examples"
SOURCE_SHA256 = {
    "en": "7dbca46fbd13d3c3538f372168bba63c97e60e798a8ecdc95efbf05f607dcb8c",
    "zh-CN": "1bb0d6faa354bf4c81a42784817e9486e35c7782569f3f1282b0c77cb6a2c152",
}
SOURCE_BOXES = ((54, 170, 546, 635), (1154, 170, 1646, 635))
DEST_X = (34, 574)
DEST_Y = 150
CANVAS_SIZE = (1100, 670)
LABELS = {
    "en": (
        "From phone photo to rectified radiograph",
        "Fixed synthetic example · missing image content is not generated",
        "Input photograph",
        "Rectified output (+2250)",
        "White: reference · cyan: parent · orange: +2250",
        "Checkerboard: area outside the photograph",
    ),
    "zh-CN": (
        "从手机照片提取摆正的 X 光片",
        "固定合成示例 · 未拍到的影像内容不会被补造",
        "输入照片",
        "展平输出（+2250）",
        "白：参考角点 · 青：父模型 · 橙：+2250",
        "棋盘格：原照片未拍到的区域",
    ),
}


def resolve_zh_font(explicit: Path | None, face_index: int | None) -> tuple[Path, int]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        default_index = 2 if explicit.suffix.lower() == ".ttc" and "NotoSansCJK" in explicit.name else 0
        return explicit, default_index if face_index is None else face_index
    if shutil.which("fc-match"):
        match = subprocess.run(
            ["fc-match", "-f", "%{file}\n%{index}\n", "Noto Sans CJK SC:lang=zh-cn"],
            text=True, capture_output=True, check=False,
        )
        lines = match.stdout.splitlines()
        if match.returncode == 0 and len(lines) >= 2:
            path = Path(lines[0])
            if path.is_file() and "NotoSans" in path.name:
                return path, int(lines[1]) if face_index is None else face_index
    raise FileNotFoundError("Noto Sans CJK SC not found; pass --zh-font and --zh-font-index")


def font(path: Path, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def fit_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
             face: ImageFont.FreeTypeFont, max_width: int, color: str) -> None:
    bbox = draw.textbbox(xy, value, font=face)
    if bbox[2] - bbox[0] > max_width:
        raise ValueError(f"Label exceeds {max_width} pixels: {value}")
    draw.text(xy, value, font=face, fill=color)


def build(locale: str, zh_font: tuple[Path, int] | None, output: Path) -> str:
    source = EXAMPLES / ("zh-CN/03-missing-corner.png" if locale == "zh-CN"
                         else "03-missing-corner.png")
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256[locale]:
        raise ValueError(f"Reviewed input changed: {source}")
    face_path, face_index = zh_font if locale == "zh-CN" else (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 0)
    if not face_path.is_file():
        raise FileNotFoundError(face_path)
    heading, subheading, left_label, right_label, left_note, right_note = LABELS[locale]
    canvas = Image.new("RGB", CANVAS_SIZE, "#faf9f6")
    draw = ImageDraw.Draw(canvas)
    fit_text(draw, (24, 17), heading, font(face_path, face_index, 30), 1052, "#172b35")
    fit_text(draw, (24, 62), subheading, font(face_path, face_index, 17), 1052, "#657781")
    with Image.open(source) as reviewed:
        if reviewed.mode != "RGB" or reviewed.size != (1700, 720) or reviewed.info:
            raise ValueError("Reviewed source dimensions, mode or metadata changed")
        for idx, (box, x) in enumerate(zip(SOURCE_BOXES, DEST_X, strict=True)):
            draw.line((x - 10, 100, x + 502, 100), fill="#dbe1df", width=2)
            title = left_label if idx == 0 else right_label
            fit_text(draw, (x + 4, 111), title, font(face_path, face_index, 20),
                     482, "#172b35")
            pixels = reviewed.crop(box)
            canvas.paste(pixels, (x, DEST_Y))
            if canvas.crop((x, DEST_Y, x + 492, DEST_Y + 465)).tobytes() != pixels.tobytes():
                raise ValueError("Hero changed an audited picture pixel")
            note = left_note if idx == 0 else right_note
            fit_text(draw, (x + 4, 623), note, font(face_path, face_index, 14),
                     482, "#657781")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=9)
    with Image.open(output) as check:
        if check.size != CANVAS_SIZE or check.mode != "RGB" or check.info:
            raise ValueError("Export has unexpected dimensions, mode or metadata")
    return sha256(output.read_bytes()).hexdigest()



def build_mobile(locale: str, zh_font: tuple[Path, int] | None, output: Path) -> str:
    """Stack the same unscaled picture rectangles for narrow README viewports."""
    source = EXAMPLES / ("zh-CN/03-missing-corner.png" if locale == "zh-CN"
                         else "03-missing-corner.png")
    if sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256[locale]:
        raise ValueError(f"Reviewed input changed: {source}")
    face_path, face_index = zh_font if locale == "zh-CN" else (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 0)
    title = ("从手机照片取出摆正的 X 光片" if locale == "zh-CN"
             else "Phone photo → rectified radiograph")
    subtitle = ("固定合成示例 · 不补造未拍到的影像" if locale == "zh-CN"
                else "Fixed synthetic example · no missing pixels generated")
    _, _, left_label, right_label, left_note, right_note = LABELS[locale]
    canvas = Image.new("RGB", (540, 1240), "#faf9f6")
    draw = ImageDraw.Draw(canvas)
    fit_text(draw, (14, 15), title, font(face_path, face_index, 26), 512, "#172b35")
    fit_text(draw, (14, 56), subtitle, font(face_path, face_index, 16), 512, "#657781")
    with Image.open(source) as reviewed:
        if reviewed.mode != "RGB" or reviewed.size != (1700, 720) or reviewed.info:
            raise ValueError("Reviewed source dimensions, mode or metadata changed")
        for idx, box in enumerate(SOURCE_BOXES):
            top = 100 if idx == 0 else 665
            draw.line((14, top, 526, top), fill="#dbe1df", width=2)
            fit_text(draw, (28, top + 11), left_label if idx == 0 else right_label,
                     font(face_path, face_index, 20), 482, "#172b35")
            pixels = reviewed.crop(box)
            image_y = top + 50
            canvas.paste(pixels, (24, image_y))
            if canvas.crop((24, image_y, 516, image_y + 465)).tobytes() != pixels.tobytes():
                raise ValueError("Mobile hero changed an audited picture pixel")
            fit_text(draw, (28, top + 522), left_note if idx == 0 else right_note,
                     font(face_path, face_index, 14), 482, "#657781")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", compress_level=9)
    with Image.open(output) as check:
        if check.size != (540, 1240) or check.mode != "RGB" or check.info:
            raise ValueError("Mobile export has unexpected dimensions, mode or metadata")
    return sha256(output.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-en", type=Path,
                        default=EXAMPLES / "hero-missing-corner.png")
    parser.add_argument("--output-zh", type=Path,
                        default=EXAMPLES / "zh-CN/hero-missing-corner.png")
    parser.add_argument("--output-en-mobile", type=Path,
                        default=EXAMPLES / "hero-missing-corner-mobile.png")
    parser.add_argument("--output-zh-mobile", type=Path,
                        default=EXAMPLES / "zh-CN/hero-missing-corner-mobile.png")
    parser.add_argument("--zh-font", type=Path)
    parser.add_argument("--zh-font-index", type=int)
    args = parser.parse_args()
    zh_font = resolve_zh_font(args.zh_font, args.zh_font_index)
    for locale, path, mobile_path in (
        ("en", args.output_en, args.output_en_mobile),
        ("zh-CN", args.output_zh, args.output_zh_mobile),
    ):
        print(f"{path}: {build(locale, zh_font, path)}")
        print(f"{mobile_path}: {build_mobile(locale, zh_font, mobile_path)}")


if __name__ == "__main__":
    main()
