"""Local visual previews, byte-preservation audit and raw-regressor comparisons."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .data import read_rgb, write_json
from .geometry import as_matrix, corners, pixel_matrix, validate_quad
from .mixed_synthesis import summarize_samples
from .pipeline import process_image
from .reports import STYLE, contact_sheet, save_rgb


def figures(paths, labels):
    return ''.join(f'<figure><a href="{p.name}" target="_blank"><img src="{p.name}" loading="lazy"></a><figcaption>{html.escape(label)}</figcaption></figure>' for p, label in zip(paths, labels))


def preview_mixed(config):
    from .engine import setup
    images, indices, _, synth = setup(config)
    output = Path(config["run"])/"preview"
    if (output/"index.html").exists():
        raise FileExistsError("Preview exists; choose a new run")
    output.mkdir(parents=True, exist_ok=True)
    selected = indices[np.linspace(0, len(indices)-1, 30).astype(int)]
    rows, articles, records = [], [], []
    for j, index in enumerate(selected):
        mode = "identity" if j % 10 < 3 else ("mild" if j % 10 < 9 else "hard")
        source = images[index]
        photo, label, info = synth.sample_record(source, config["seed"]+j, mode=mode)
        matrix = as_matrix(label)
        overlay = photo.copy()
        cv2.polylines(overlay, [np.rint(corners(matrix, (224, 224))).astype(np.int32)], True, (70, 255, 80), 2)
        recovered = cv2.warpPerspective(photo, np.linalg.inv(pixel_matrix(matrix, (224, 224), (224, 224))), (224, 224))
        sid = f"synth{j+1:02}"
        paths = []
        for name, array in zip(["source", "photo", "label", "recovered"], [source, photo, overlay, recovered]):
            p = output/f"{sid}-{name}.png"
            save_rgb(p, array)
            paths.append(p)
        records.append({"id": sid, "cache_index": int(index), "label": label.tolist(), **info})
        rows.append((sid+" "+mode+" | source / input / label / inverse", paths))
        articles.append(f'<article><h2>{sid} · {mode}</h2><div class="grid four">'+figures(paths, ["数字原片", "训练输入", "自动四角", "真值还原"])+"</div></article>")
    for start in range(0, 30, 5):
        contact_sheet(rows[start:start+5], output/f"contact-{start//5+1:02}.jpg", 224)
    write_json(output/"examples.local.json", records)
    write_json(output/"summary.json", summarize_samples(records))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>B1.1合成检查</title><style>'+STYLE+'</style><h1>B1.1：30张合成标签检查</h1><p>9张原片恒等，18张轻微翻拍，3张较大变形。绿色四角只标原片，不包含软件窗口。</p>'+''.join(articles))
    print(json.dumps(summarize_samples(records)), flush=True)


def audit_digital(config):
    data = Path(config["data"])
    manifest = json.loads((data/"manifest.local.json").read_text())
    records = [r for r in manifest["records"] if r["split"] == "val"]
    root = Path(config["run"])/"digital"
    output = root/"passthrough"
    output.mkdir(parents=True, exist_ok=True)
    checks, modes = [], Counter()
    for r in records:
        source = Path(r["source_path"])
        target = output/(r["id"]+source.suffix.lower())
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if source_hash != r["source_sha256"]:
            raise ValueError("Source no longer matches prepared manifest")
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            result = process_image(source, target, "digital")
            digest = result["sha256"]
        if digest != source_hash:
            raise ValueError("Digital passthrough is not byte-identical")
        with Image.open(source) as im:
            mode, size = im.mode, list(im.size)
            dtype = str(np.asarray(im).dtype)
        modes[mode+"/"+dtype] += 1
        checks.append({"id": r["id"], "cache_index": r["cache_index"], "byte_identical": True, "mode": mode, "dtype": dtype, "size": size})
    summary = {"count": len(records), "byte_identical": len(checks), "model_used": False, "decoded_in_pipeline": False,
               "source_unchanged": True, "pixel_modes": dict(modes), "real_high_bit_depth_available": any("16" in r["dtype"] or "32" in r["dtype"] for r in checks)}
    write_json(root/"passthrough-results.local.json", checks)
    write_json(root/"passthrough-summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def digital_comparison(config):
    run = Path(config["run"])
    root = run/"digital"
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(config["data"])/"manifest.local.json").read_text())
    pools = defaultdict(list)
    for r in manifest["records"]:
        if r["split"] == "val":
            pools[r["source_name"]].append(r)
    selected = [r for pool in pools.values() for r in pool[:3]]
    names = [config["baseline_name"], config["candidate_name"]]
    predictions = {}
    for name in names:
        d = np.load(run/"evaluation"/name/"identity-predictions.local.npz")
        predictions[name] = dict(zip(d["cache_indices"].tolist(), d["predictions"]))
    articles, rows = [], []
    for r in selected:
        sid = r["id"]
        photo = read_rgb(Path(r["source_path"]))
        original = root/f"{sid}-display.png"
        save_rgb(original, photo)
        paths, labels = [original], ["原片显示预览（直通文件保持原字节）"]
        for name in names:
            matrix = as_matrix(predictions[name][r["cache_index"]])
            overlay = photo.copy()
            quad = corners(matrix, (photo.shape[1], photo.shape[0]))
            reasons = validate_quad(quad, (photo.shape[1], photo.shape[0]))
            cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (40, 255, 80), max(2, photo.shape[1]//450))
            path = root/f"{sid}-{name}-quad.png"
            save_rgb(path, overlay)
            paths.append(path)
            labels.append(name.upper()+" 强制经过模型：预测框"+("；几何检查拒绝："+", ".join(reasons) if reasons else "；几何检查通过"))
        rows.append((sid+" | digital / B1 raw / B1.1 raw", paths))
        articles.append(f'<article><h2>{sid}</h2><div class="grid">'+figures(paths, labels)+"</div></article>")
    for start in range(0, len(rows), 4):
        contact_sheet(rows[start:start+4], root/f"contact-{start//4+1:02}.jpg", 350)
    evaluation = json.loads((run/"full-development-evaluation.json").read_text())
    bypass = json.loads((root/"passthrough-summary.json").read_text())
    table = '<table><tr><th>指标</th>'+''.join(f'<th>{n.upper()}</th>' for n in names)+'</tr>'
    for title, metric in [("平均四角位移", "corner_nme_mean"), ("P90四角位移", "corner_nme_p90"), ("平均原画布覆盖", "target_canvas_coverage_mean")]:
        table += '<tr><td>'+title+'</td>'+''.join(f'<td>{evaluation[n]["identity"][metric]*100:.2f}%</td>' for n in names)+'</tr>'
    table += '<tr><td>几何检查拒绝</td>'+''.join(f'<td>{evaluation[n]["identity"]["invalid"]} / 478</td>' for n in names)+'</tr>'
    table += '</table>'
    (root/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>数字原片保护</title><style>'+STYLE+'td,th{padding:8px 22px;text-align:left}</style><h1>数字原片：文件直通与模型恒等分开检查</h1>'+f'<p>实际数字片直通：{bypass["byte_identical"]}/{bypass["count"]} 文件SHA-256完全一致。此路径没有解码、推理、缩放或去纹。</p><p>下表关闭直通，强制全部478张开发原片经过回归器；下方展示12例预测框。画布覆盖率不等于骨骼保留率。绿色预测线伸出画布时会被显示边界截断，具体拒绝原因写在每张图下方。</p>'+table+''.join(articles))
