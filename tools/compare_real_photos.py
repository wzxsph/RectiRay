"""Create a local side-by-side PTRN gallery for authorized real photographs.

Example:
  KERAS_BACKEND=torch python tools/compare_real_photos.py \
      --bundle /path/to/kit --photos /path/to/authorized-photos \
      --checkpoint plus2250=/path/to/branch-step-2250.pt \
      --output .local/reviews/plus2250-real

The output contains copies of the photographs. Keep it in an ignored local
directory; neither this tool nor its output is suitable for public CI.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re

os.environ.setdefault("KERAS_BACKEND", "torch")

import cv2
import numpy as np
import torch

from ptrn_hip.data import read_rgb
from ptrn_hip.partial_geometry import CONTRACT, rectify_visible
from ptrn_hip.ratio_synthesis import letterbox, project
from ptrn_pipeline.evaluation import inference, load_model


PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RENDERER_PATH = "src/ptrn_hip/partial_geometry.py"
# Reviewed local V3 area-resampling renderer. The +2250 training checkpoint
# predates this rendering-only change; its network/labels remain unchanged.
RENDERER_SHA256 = "a23bb919dd5cea1437477b0cb6478247e6bf9735599962787d7b48e42ef92549"
PARENT_TRAINING_ONLY_UPDATES = {
    "partial_data.py": (
        "c4631fcda515673f7f442e98f446e90d6b83cb70911744efef905983ef170f97",
        "fc142a85c7935b74f2d7a1903bdbc8e4b11529c6b10136700d80985dfc887906",
    ),
    "partial_synthesis.py": (
        "b4de206c40a1dcd249fdfd1fcad20ab9273f57b6ddf5eb769a40822763fa74cc",
        "5456f2e659131924e3a63ea7663a6f959f22f3a2fe48cdcf4394b1b8eb7853d8",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_arg(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", label) or not path:
        raise argparse.ArgumentTypeError("Use --checkpoint short_label=/path/to/file.pt")
    return label, Path(path)


def validate_code_binding(checkpoint: dict, repo: Path, *, allow_renderer_update: bool,
                          allow_parent_source_updates: bool = False) -> list[dict]:
    """Verify training bytes; separately record the opted-in rendering change."""
    deviations = []
    code = checkpoint.get("binding", {}).get("code")
    if not isinstance(code, dict) or not code:
        raise ValueError("Checkpoint has no source binding")
    repository_relative = any("/" in name for name in code)
    for relative, expected in code.items():
        # The original pipeline runner stored package basenames; the later
        # step-2000 branch stored repository-relative paths.
        source = (repo / (relative if repository_relative else f"src/ptrn_hip/{relative}")).resolve()
        if not source.is_relative_to(repo.resolve()) or not source.is_file():
            raise ValueError(f"Checkpoint source binding differs: {relative}")
        actual = sha256(source)
        if actual != expected:
            renderer = allow_renderer_update and source == (repo / RENDERER_PATH).resolve() and actual == RENDERER_SHA256
            old_pair = PARENT_TRAINING_ONLY_UPDATES.get(relative)
            parent = allow_parent_source_updates and old_pair == (expected, actual)
            if not (renderer or parent):
                raise ValueError(f"Checkpoint source binding differs: {relative}")
            deviations.append({"path": relative, "trained_sha256": expected,
                               "current_sha256": actual,
                               "reason": "reviewed V3 area-resampling renderer" if renderer else "later training-only source update"})
    return deviations


def validate_asset_binding(checkpoint: dict, bundle: Path) -> None:
    assets = checkpoint["binding"].get("assets")
    if not assets and isinstance(checkpoint["binding"].get("weights"), dict):
        binding = checkpoint["binding"]
        assets = {f"weights/{name}": digest for name, digest in binding["weights"].items()}
        assets.update({"data/manifest.json": binding["manifest"],
                       "data/backgrounds-manifest.json": binding["background_manifest"],
                       "data/split.json": binding["development"]["split"]})
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Checkpoint has no asset binding")
    for relative, expected in assets.items():
        source = (bundle / relative).resolve()
        if not source.is_relative_to(bundle.resolve()) or not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Checkpoint asset binding differs: {relative}")


def write_png(path: Path, rgba: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write {path}")


def write_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Save the exact oriented inference pixels, without source EXIF metadata."""
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Could not write {path}")


def gallery_html(labels: list[str]) -> str:
    options = "".join(f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in labels)
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PTRN 实拍取图对照</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101820;color:#edf4f7}
body{margin:0}header,main{padding:18px max(15px,3vw)}header{background:#1b2d39}p{color:#c4d6dd;line-height:1.5}
nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px max(15px,3vw);position:sticky;top:0;background:#1b2d39}
button,select{background:#172a35;color:inherit;border:1px solid #8197a2;padding:7px;border-radius:6px;font:inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}figure{margin:0;background:#1b2d39;padding:10px;border-radius:8px}
figcaption{min-height:35px}img{width:100%;height:60vh;object-fit:contain;background:repeating-conic-gradient(#384c56 0% 25%,#223642 0% 50%) 50%/20px 20px}
.photo{position:relative}.photo svg{position:absolute;inset:0;width:100%;height:60vh;pointer-events:none}a{color:#9ed9eb}
</style></head><body><header><h1>PTRN 实拍取图对照</h1>
<p>同一批本地实拍，逐张查看原图、预测四角与实际可见部分取图。没有可靠四角真值时，本页仅供视觉评审，不计算正确率。</p>
<p>影像和逐例记录仅保存在本地；透明处表示输入没有可用像素，没有补造影像内容。</p></header>
<nav><button id="prev">上一张</button><select id="photo"></select><button id="next">下一张</button>
<label>左 <select id="left">""" + options + """</select></label><label>右 <select id="right">""" + options + """</select></label>
<label><input id="quad" type="checkbox" checked>显示四角</label><span id="count"></span></nav>
<main id="content"></main><script src="gallery-data.js"></script><script>
const data=window.PTRN_GALLERY, $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let index=0;data.rows.forEach((row,i)=>{$('photo').add(new Option(row.id,i))});
$('right').selectedIndex=Math.min(1,data.models.length-1);
function figure(row,label){const out=row.outputs[label];if(!out)return '';
return `<figure><figcaption>${esc(label)} · 比例 ${Number.isFinite(out.ratio)?out.ratio.toFixed(3):'—'}</figcaption>`+
(out.rendered?`<a href="${esc(out.image)}" target="_blank"><img loading="lazy" src="${esc(out.image)}" alt="${esc(row.id)} ${esc(label)}"></a>`:
`<p>取图失败：${esc(out.reason)}</p>`)+`</figure>`}
function draw(){const row=data.rows[index],left=$('left').value,right=$('right').value;
const lines=$('quad').checked?[left,right].map((label,i)=>{const out=row.outputs[label];return out.quad?
`<polygon points="${out.quad.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${i?'#ffb466':'#6ac9ff'}" stroke-width="3" vector-effect="non-scaling-stroke"/>`:''}).join(''):'';
$('content').innerHTML=`<h2>${esc(row.id)}</h2><div class="grid"><figure><figcaption>原图 · 蓝色左权重 / 橙色右权重</figcaption><div class="photo"><img src="${esc(row.input)}" alt="${esc(row.id)} 原图"><svg viewBox="0 0 ${row.width} ${row.height}" preserveAspectRatio="xMidYMid meet">${lines}</svg></div></figure>${figure(row,left)}${figure(row,right)}</div>`;
$('photo').value=index;$('count').textContent=`${index+1} / ${data.rows.length}`;
$('prev').disabled=index===0;$('next').disabled=index===data.rows.length-1}
$('prev').onclick=()=>{index--;draw()};$('next').onclick=()=>{index++;draw()};
$('photo').onchange=()=>{index=Number($('photo').value);draw()};
for(const id of ['left','right','quad'])$(id).onchange=draw;draw();
</script></body></html>"""


def build_gallery(bundle: Path, photos: Path, checkpoints: list[tuple[str, Path]], output: Path,
                  max_side: int, *, allow_renderer_update: bool = False,
                  allow_parent_source_updates: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The saved pipeline inference path requires CUDA")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty; existing case images are never overwritten")
    repo = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo.resolve()) and not output.resolve().is_relative_to((repo / ".local").resolve()):
        raise ValueError("Repository-local case galleries must be under ignored .local/")
    if not photos.is_dir() or not bundle.is_dir():
        raise FileNotFoundError("Photos and bundle directories are required")
    # Validate all checkpoints before copying any real photograph into output.
    preflight = []
    for label, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("contract") != CONTRACT:
            raise ValueError(f"{label}: renderer/checkpoint contract differs")
        deviations = validate_code_binding(checkpoint, repo, allow_renderer_update=allow_renderer_update,
                                           allow_parent_source_updates=allow_parent_source_updates)
        validate_asset_binding(checkpoint, bundle)
        preflight.append((label, path, sha256(path), deviations))
        del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir()
    sources = sorted(path for path in photos.iterdir() if path.is_file() and path.suffix.lower() in PHOTO_EXTENSIONS)
    if not sources:
        raise ValueError("No supported photos found")
    prepared = []
    rows = []
    for i, source in enumerate(sources, 1):
        rgb = read_rgb(source)
        pixels, transform = letterbox(rgb, 224)
        target = images / f"photo-{i:04d}.png"
        write_rgb_png(target, rgb)
        digest = sha256(source)
        if sha256(source) != digest or not np.array_equal(read_rgb(target), rgb):
            raise ValueError("Source changed or metadata-free preview pixels differ")
        prepared.append((rgb, {"pixels": pixels}, np.linalg.inv(transform)))
        rows.append({"id": f"photo-{i:04d}", "input": f"images/{target.name}", "source_sha256": digest,
                     "width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "outputs": {}})
    model = None
    models = []
    for label, path, checkpoint_sha, deviations in preflight:
        if model is None:
            model, checkpoint = load_model(bundle, path)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"], strict=True)
        if sha256(path) != checkpoint_sha:
            raise ValueError(f"{label}: checkpoint changed during gallery build")
        model.eval()
        descriptor = {"label": label, "checkpoint_sha256": checkpoint_sha, "step": checkpoint.get("step"),
                      "contract": checkpoint["contract"], "source_deviations": deviations}
        models.append(descriptor)
        del checkpoint
        for row, (rgb, item, inverse) in zip(rows, prepared):
            entry = {"quad": None, "ratio": None, "rendered": False, "image": None}
            try:
                with torch.no_grad():
                    prediction = inference(model, [item])
                quad = project(prediction["quad"][0], inverse)
                ratio = float(prediction["ratio"][0])
                if not np.isfinite(quad).all() or not np.isfinite(ratio):
                    raise ValueError("Non-finite predicted geometry")
                entry.update(quad=quad.tolist(), ratio=ratio)
                rectified = rectify_visible(rgb, quad, ratio, max_side=max_side)
                relative = f"images/{row['id']}-{label}.png"
                rgba = np.dstack((rectified.pixels, rectified.valid_mask))
                write_png(output / relative, rgba)
                entry.update(rendered=True, image=relative, info=rectified.info)
            except (ValueError, FloatingPointError, OverflowError) as error:
                entry["reason"] = str(error)
            row["outputs"][label] = entry
    report = {"scope": "Authorized local real photographs; visual comparison without ground truth",
              "models": models, "rows": rows,
              "runtime": {"prediction_backend": "eager", "batch": 1, "renderer_contract": CONTRACT,
                          "renderer_sha256": sha256(repo / RENDERER_PATH),
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                          "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "max_side": max_side}}
    (output / "report.local.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    safe_json = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (output / "gallery-data.js").write_text("window.PTRN_GALLERY=" + safe_json + ";\n", encoding="utf-8")
    (output / "index.html").write_text(gallery_html([label for label, _ in checkpoints]), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=checkpoint_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--allow-renderer-update", action="store_true",
                        help="Accept only the reviewed V3 renderer SHA when it differs from training")
    parser.add_argument("--allow-parent-source-updates", action="store_true",
                        help="Allow two exact audited, training-only source changes for original pipeline checkpoints")
    args = parser.parse_args()
    if len({label for label, _ in args.checkpoint}) != len(args.checkpoint):
        parser.error("Checkpoint labels must be unique")
    if not 2 <= args.max_side <= 8192:
        parser.error("Invalid --max-side")
    report = build_gallery(args.bundle, args.photos, args.checkpoint, args.output, args.max_side,
                           allow_renderer_update=args.allow_renderer_update,
                           allow_parent_source_updates=args.allow_parent_source_updates)
    print(f"Created local gallery: {len(report['rows'])} photos × {len(report['models'])} checkpoints")


if __name__ == "__main__":
    main()
