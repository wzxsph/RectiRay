#!/usr/bin/env python3
"""Build the bilingual, source-audited RectiRay architecture illustration.

This is a schematic of PartialModel.forward, SpatialModel._spatial_logits,
RatioResidual.forward, and partial_geometry.rectify_visible. It never opens
medical images or model weights. The SVG bytes are deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1500, 950
INK, MUTED, TEAL, CORAL = "#172b35", "#657781", "#087e8b", "#d56a46"
PAPER, RULE, PALE = "#faf9f6", "#dbe1df", "#eaf3f2"

COPY = {
    "en": {
        "title": "From spatial features to a rectified photograph",
        "subtitle": "RectiRay · V3 research pipeline  /  geometry on a 224-pixel view; pixels from the original photograph",
        "accessible_title": "RectiRay model and rendering architecture",
        "accessible_desc": "Schematic, not real feature activations. An original-resolution photograph is letterboxed and normalized to a 224-pixel input. DenseNet201 produces 56 by 56 by 256, 28 by 28 by 512, 14 by 14 by 1792, and 7 by 7 by 1920 feature maps. One-by-one projections to 64 channels and top-down upsampling, addition, and 3-by-3 smoothing produce a shared four-channel spatial map. A flattened direct linear eight readout predicts ordered, unbounded corners. A separate one-by-one mask head is training-only. Pooled conv5 features and detached corners feed a ratio residual on an edge-log baseline. Four corners and full-plane ratio guide a single geometric resampling of the original photograph, yielding observed output and validity. A trusted original with verified provenance follows a separate pixel-exact keep route.",
        "capture": "ORIGINAL PHOTO", "input": "NETWORK VIEW", "encoder": "ENCODER", "fusion": "SPATIAL DECODER", "heads": "PREDICTIONS",
        "original": "Original RGB", "schematic": "full resolution · schematic", "letterbox": "letterbox + normalize", "input_size": "224 × 224 RGB",
        "densenet": "DenseNet201", "frozen": "head stage: encoder + BN frozen",
        "fuse": "1×1→64 · ↑2×+add · 3×3", "conv2": "conv2", "conv3": "conv3", "conv4": "conv4", "conv5": "conv5",
        "shared": "Shared map", "shared_size": "4 ch · 56²",
        "quad": "Ordered corners", "quad_ops": "56² pool · flat · Linear 8", "quad_note": "TL  TR  BR  BL · unbounded", "not_heatmap": "direct coordinate readout",
        "mask": "Visible-film mask", "mask_ops": "1 × 1 convolution", "mask_note": "train-only auxiliary; no crop",
        "ratio": "FULL-PLANE RATIO", "ratio_sources": "GAP conv5 (1920) + detached corners", "ratio_ops": "edge-log baseline + MLP δ → log(W/H)", "ratio_note": "exp → W/H · not visible-crop ratio",
        "render": "RENDER AT ORIGINAL RESOLUTION", "geo": "quad + full-plane W / H", "sampler": "Geometry sampler", "sample_note": "one photo-to-output mapping", "output": "Observed output", "validity": "+ pixel-validity mask", "unseen_a": "Unphotographed pixels", "unseen_b": "remain absent",
        "trust": "Verified provenance", "keep": "Pixel-exact original keep path", "legend": "Schematic tensors, not measured activations",
    },
    "zh-CN": {
        "title": "从空间特征到摆正后的照片",
        "subtitle": "RectiRay · V3 研究架构  /  在 224 像素视图上预测几何；从原分辨率照片读取像素",
        "accessible_title": "RectiRay 模型与取图架构",
        "accessible_desc": "示意图，特征图并非真实激活。原分辨率照片等比例补边并归一化至 224 像素。DenseNet201 输出 56 乘 56 乘 256、28 乘 28 乘 512、14 乘 14 乘 1792、7 乘 7 乘 1920 四级特征。各级经一乘一投影至 64 通道，逐级向上采样、相加和三乘三平滑，得到共享四通道空间图。展平后直接线性八维读出有序、可越界四角。独立的一乘一掩码头仅辅助训练。conv5 汇总特征和停止梯度的四角进入比例残差，与边长对数基线得到完整平面比例。四角及比例指导原照片的一次几何采样，得到实际拍到的输出和有效掩码。来源可信的数字原片走独立的逐像素保留路径。",
        "capture": "原始照片", "input": "网络视图", "encoder": "编码器", "fusion": "空间解码器", "heads": "预测结果",
        "original": "原始 RGB", "schematic": "原分辨率 · 照片示意", "letterbox": "等比例补边＋归一化", "input_size": "224 × 224 RGB",
        "densenet": "DenseNet201", "frozen": "head 阶段：冻结编码器及 BN",
        "fuse": "1 × 1 → 64 · ↑ 上采样相加 · 3 × 3", "conv2": "conv2", "conv3": "conv3", "conv4": "conv4", "conv5": "conv5",
        "shared": "共享空间图", "shared_size": "4 通道 · 56²",
        "quad": "有序四角", "quad_ops": "56² 池化 · 展平 · 线性 8 维", "quad_note": "左上  右上  右下  左下 · 可越界", "not_heatmap": "直接坐标读出",
        "mask": "可见片区掩码", "mask_ops": "1 × 1 卷积", "mask_note": "仅辅助训练；不负责裁剪",
        "ratio": "完整平面宽高比", "ratio_sources": "conv5 全局均值 (1920)＋停止梯度四角", "ratio_ops": "边长对数基线＋MLP 残差 δ → log(宽／高)", "ratio_note": "取指数 → 完整宽／高；非可见裁片比例",
        "render": "在原分辨率取图", "geo": "四角＋完整平面宽／高", "sampler": "几何取图器", "sample_note": "照片到输出只映射采样一次", "output": "实际拍到的输出", "validity": "＋像素有效掩码", "unseen_a": "未拍到的影像", "unseen_b": "不会被补造",
        "trust": "可信数字原片", "keep": "逐像素保留原片路径", "legend": "张量为示意，并非真实激活图",
    },
}


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def text(self, x: float, y: float, value: str, style: str, *, anchor: str | None = None) -> None:
        a = f' text-anchor="{anchor}"' if anchor else ""
        self.add(f'<text class="{style}" x="{x:g}" y="{y:g}"{a}>{escape(value)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = RULE,
             width: float = 2, dash: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{width:g}"{d}/>')

    def path(self, d: str, *, color: str = MUTED, width: float = 2.2,
             marker: str | None = None, dash: str | None = None) -> None:
        m = f' marker-end="url(#{marker})"' if marker else ""
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width:g}" stroke-linecap="round" stroke-linejoin="round"{m}{dd}/>')

    def rect(self, x: float, y: float, w: float, h: float, *, fill: str = "#fff",
             stroke: str | None = None, radius: float = 0, sw: float = 1.5) -> None:
        s = f' stroke="{stroke}" stroke-width="{sw:g}"' if stroke else ""
        self.add(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{radius:g}" fill="{fill}"{s}/>')


def tensor(s: SVG, x: float, y: float, size: float, *, color: str = TEAL, channels: int = 3) -> None:
    """A stack of gridded squares: a conceptual map, not an activation image."""
    for layer in range(channels - 1, -1, -1):
        offset = 5 * layer
        s.rect(x + offset, y - offset, size, size, fill="#f8fcfb" if layer == 0 else "#dceceb",
               stroke=color if layer == 0 else "#b8d4d4", radius=2, sw=1.5)
    for n in (1, 2):
        p = size * n / 3
        s.line(x + p, y, x + p, y + size, color="#a4c4c4", width=1)
        s.line(x, y + p, x + size, y + p, color="#a4c4c4", width=1)


def photo(s: SVG, x: float, y: float, w: float, h: float) -> None:
    """Generic vector geometry, deliberately not a patient radiograph."""
    s.rect(x, y, w, h, fill="#e7e9e6", stroke=RULE, radius=5)
    s.rect(x + 7, y + 8, w - 14, h - 16, fill="#273b44", radius=2)
    s.add(f'<path d="M{x+26:g} {y+36:g} L{x+w-24:g} {y+25:g} L{x+w-18:g} {y+h-28:g} L{x+22:g} {y+h-18:g}Z" fill="#8ca5a9" stroke="#f6faf8" stroke-width="2.5"/>')
    for yy in (y + h * .42, y + h * .56, y + h * .70):
        s.path(f'M{x+42:g} {yy:g} Q{x+w/2:g} {yy-20:g} {x+w-35:g} {yy+4:g}', color="#d8e2dd", width=4)
    for xx, yy in ((x + 26, y + 36), (x + w - 24, y + 25),
                   (x + w - 18, y + h - 28), (x + 22, y + h - 18)):
        s.add(f'<circle cx="{xx:g}" cy="{yy:g}" r="4.2" fill="{CORAL}"/>')


def plot(locale: str) -> str:
    t = COPY[locale]
    fonts = '"Noto Sans CJK SC","DejaVu Sans",sans-serif' if locale == "zh-CN" else '"DejaVu Sans","Noto Sans CJK SC",sans-serif'
    s = SVG()
    s.add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc" xml:lang="{locale}">')
    s.add(f'<title id="title">{escape(t["accessible_title"])}</title>')
    s.add(f'<desc id="desc">{escape(t["accessible_desc"])} Source contract: src/ptrn_hip/partial_model.py PartialModel.forward; src/ptrn_hip/spatial_model.py SpatialModel._spatial_logits; src/ptrn_hip/spatial_objectives.py RatioResidual.forward; src/ptrn_hip/partial_geometry.py rectify_visible.</desc>')
    s.add(f'''<defs>
      <marker id="arrow-teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{TEAL}"/></marker>
      <marker id="arrow-ink" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{INK}"/></marker>
      <marker id="arrow-coral" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{CORAL}"/></marker>
      <style>
        text{{font-family:{fonts};fill:{INK}}}
        .title{{font-size:36px;font-weight:700;letter-spacing:-.7px}}
        .subtitle{{font-size:18px;fill:{MUTED}}}
        .kicker{{font-size:15px;fill:{TEAL};font-weight:700;letter-spacing:1.3px}}
        .head{{font-size:24px;font-weight:700}}
        .label{{font-size:20px;font-weight:650}}
        .body{{font-size:18px}}
        .small{{font-size:17px;fill:{MUTED}}}
        .micro{{font-size:15px;fill:{MUTED}}}
        .teal{{fill:{TEAL}}}.coral{{fill:{CORAL}}}.muted{{fill:{MUTED}}}
      </style>
    </defs>''')
    s.rect(0, 0, WIDTH, HEIGHT, fill=PAPER)
    s.text(55, 57, t["title"], "title")
    s.text(55, 87, t["subtitle"], "subtitle")
    s.line(55, 109, 1445, 109, width=1.5)

    # Five aligned regions; the glyphs carry the architectural distinction.
    for x, key in ((60, "capture"), (265, "input"), (460, "encoder"), (735, "fusion"), (1204, "heads")):
        s.text(x, 145, t[key], "kicker")
    s.line(55, 158, 1445, 158, width=1)

    photo(s, 69, 194, 146, 225)
    s.text(60, 453, t["original"], "body")
    s.text(60, 476, t["schematic"], "small")
    s.path("M216 306H252", color=INK, marker="arrow-ink")

    s.rect(270, 244, 145, 145, fill="#edf0ed", stroke=RULE, radius=3)
    # Draw exactly the same portrait photograph under a uniform scale, with
    # horizontal letterbox padding. The network view must not imply stretching.
    scale = 145 / 225
    left = 270 + (145 - 146 * scale) / 2
    s.add(f'<g transform="translate({left:g} 244) scale({scale:g})">')
    photo(s, 0, 0, 146, 225)
    s.add('</g>')
    s.text(270, 428, t["input_size"], "label")
    s.text(270, 453, t["letterbox"], "small")
    s.path("M416 306H447", color=INK, marker="arrow-ink")

    s.text(460, 186, t["densenet"], "head")
    s.text(460, 212, t["frozen"], "small")
    # Conv2 to conv5, top to bottom. Real channel counts from DenseNet201 taps.
    levels = [("conv2", 56, 256, 258, 52), ("conv3", 28, 512, 334, 43),
              ("conv4", 14, 1792, 410, 34), ("conv5", 7, 1920, 486, 25)]
    for name, side, channels, cy, icon_size in levels:
        tensor(s, 468 + (52 - icon_size) / 2, cy - icon_size / 2, icon_size)
        s.text(541, cy - 2, t[name], "label")
        s.text(541, cy + 20, f"{side}² × {channels}", "small")
        s.path(f"M660 {cy:g}H721", color=TEAL, marker="arrow-teal")

    s.text(735, 186, t["fuse"], "small")
    # One lateral projection per scale, then explicit add and 3x3 smooth.
    # The smaller map glyphs visually encode decreasing spatial extent.
    for _, side, _, cy, _ in levels:
        icon_size = {56: 48, 28: 39, 14: 30, 7: 22}[side]
        s.text(740, cy + 5, "1 × 1", "body")
        tensor(s, 811 + (48 - icon_size) / 2, cy - icon_size / 2, icon_size, channels=2)
        s.text(797, cy - 34, f"{side}² × 64", "small")
        s.path(f"M871 {cy:g}H904", color=TEAL, marker="arrow-teal")
        if side != 7:  # The first conv5 smoothing has no upsampled summand.
            s.add(f'<circle cx="920" cy="{cy}" r="12" fill="{PAPER}" stroke="{TEAL}" stroke-width="2"/>')
            s.text(920, cy + 6, "+", "body teal", anchor="middle")
            s.path(f"M932 {cy:g}H949", color=TEAL, marker="arrow-teal")
        else:
            s.path(f"M904 {cy:g}H949", color=TEAL, marker="arrow-teal")
        s.rect(950, cy - 17, 46, 34, fill="#fff", stroke=TEAL, sw=1.5)
        s.text(973, cy + 6, "3×3", "small teal", anchor="middle")
        s.path(f"M996 {cy:g}H1005", color=TEAL, marker="arrow-teal")
        smooth_size = {56: 34, 28: 29, 14: 24, 7: 19}[side]
        tensor(s, 1010, cy - smooth_size / 2, smooth_size, channels=2)
    # Smoothed output from each lower scale is upsampled into the next add node.
    for lower, upper in ((486, 410), (410, 334), (334, 258)):
        s.path(f"M1047 {lower}H1060V{upper+32}H920V{upper+13}",
               color=TEAL, width=2.2, marker="arrow-teal")
        s.text(1070, (lower + upper) / 2 + 5, "↑", "body teal")
    s.path("M1047 258H1080", color=TEAL, marker="arrow-teal")
    s.text(1046, 214, "1 × 1 → 4", "small teal")
    tensor(s, 1093, 230, 52, channels=4)
    s.text(1065, 187, t["shared"], "label")
    s.text(1065, 322, t["shared_size"], "small")

    # Two readouts from the same four-channel map: direct corners and mask.
    s.path("M1164 257H1192", color=CORAL, marker="arrow-coral")
    s.text(1204, 223, t["quad"], "head coral")
    s.text(1204, 252, t["quad_ops"], "small")
    s.text(1204, 278, t["quad_note"], "body")
    s.add(f'<path d="M1234 302L1295 296L1302 329L1230 333Z" fill="none" stroke="{CORAL}" stroke-width="2.2"/>')
    for x, y in ((1234, 302), (1295, 296), (1302, 329), (1230, 333)):
        s.add(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{CORAL}"/>')
    s.text(1204, 355, t["not_heatmap"], "small coral")

    s.path("M1164 278H1190V390H1198", color=MUTED, marker="arrow-ink")
    s.text(1204, 386, t["mask"], "label")
    s.text(1204, 410, t["mask_ops"], "small")
    s.text(1204, 434, t["mask_note"], "small teal")
    s.rect(1204, 447, 41, 29, fill=PALE, stroke="#8dbabb", radius=2)
    s.rect(1227, 452, 13, 18, fill="#9fcdca", radius=1)

    # Ratio is not read from the spatial logits. Dashes denote detached inputs.
    s.line(1057, 502, 1445, 502, color=RULE, width=1.5)
    s.text(1062, 529, t["ratio"], "kicker")
    s.text(1062, 554, t["ratio_sources"], "body")
    s.text(1062, 580, t["ratio_ops"], "body coral")
    s.text(1062, 604, t["ratio_note"], "small")
    s.path("M647 486V592H1045", color=MUTED, width=2, marker="arrow-ink", dash="6 5")
    s.path("M1321 332H1458V551H1439", color=MUTED, width=1.8, marker="arrow-ink", dash="6 5")

    # Original RGB bypasses the inference path: only geometry is predicted.
    s.line(55, 645, 1445, 645, width=1.5)
    s.text(60, 675, t["render"], "kicker")
    s.text(692, 702, t["geo"], "body coral")
    s.path("M1260 611V689H925V715", color=CORAL, width=2.4, marker="arrow-coral")
    s.path("M69 305H45V771H635", color=INK, width=2.4, marker="arrow-ink")
    s.rect(653, 716, 307, 127, fill="#fff", stroke="#b9cfcd", radius=6, sw=1.6)
    s.text(678, 752, t["sampler"], "head")
    s.text(678, 782, t["sample_note"], "body")
    s.path("M691 813H918", color=TEAL, width=2.1, marker="arrow-teal")
    s.path("M960 778H1017", color=CORAL, marker="arrow-coral")
    s.rect(1033, 721, 192, 117, fill="#e8eceb", stroke=RULE, radius=3)
    s.rect(1050, 733, 158, 92, fill="#a2b8b9")
    s.add('<path d="M1060 744L1198 739L1198 811L1061 814Z" fill="#405862"/>')
    s.path("M1075 774Q1123 749 1187 779", color="#d3e1dc", width=4)
    s.text(1034, 863, t["output"], "label")
    s.text(1034, 886, t["validity"], "small")
    s.text(1239, 762, t["unseen_a"], "small")
    s.text(1239, 785, t["unseen_b"], "small")

    # Separate trusted-origin action. Its dashed route cannot be mistaken for
    # a learned classifier or a second sampling pass.
    s.path("M45 775V919H636", color=MUTED, width=1.8, marker="arrow-ink", dash="6 5")
    s.text(653, 919, t["trust"], "small")
    s.path("M856 913H945", color=TEAL, marker="arrow-teal", dash="6 5")
    s.text(962, 919, t["keep"], "body teal")
    s.text(60, 941, t["legend"], "micro")
    s.add('</svg>')
    return "\n".join(s.parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated bytes without writing")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root for output paths")
    args = parser.parse_args()
    paths = {
        "en": args.root / "docs/figures/architecture.svg",
        "zh-CN": args.root / "docs/figures/i18n/architecture.zh-CN.svg",
    }
    for locale, path in paths.items():
        rendered = plot(locale).encode("utf-8")
        if args.check:
            if not path.exists() or path.read_bytes() != rendered:
                raise SystemExit(f"out of date: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rendered)
        print(f"{locale}: {path} ({len(rendered)} bytes)")


if __name__ == "__main__":
    main()
