"""Local-only visual evidence for synthesis and real photo regression."""
from __future__ import annotations

import html
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .data import read_rgb, write_json
from .geometry import as_matrix, corners, pixel_matrix, rectify


STYLE = """body{margin:30px auto;max-width:1600px;background:#10151b;color:#e6edf4;font:16px system-ui;padding:0 20px}a{color:#85cfff}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.four{grid-template-columns:repeat(4,1fr)}img{max-width:100%;height:430px;object-fit:contain;background:#080b0f}figure{margin:0}figcaption{padding:10px 0;color:#aabdcf}article{border-top:1px solid #374352;padding:20px 0}code{color:#9bd5ff}@media(max-width:900px){.grid,.four{grid-template-columns:1fr}img{height:340px}}"""


def save_rgb(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def contact_sheet(rows, path, tile=300):
    result = Image.new("RGB", (len(rows[0][1]) * tile, len(rows) * (tile + 30)), "#15202b")
    draw = ImageDraw.Draw(result)
    for row, (label, paths) in enumerate(rows):
        for col, source in enumerate(paths):
            with Image.open(source) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (tile, tile))
            result.paste(thumb, (col * tile + (tile - thumb.width) // 2, row * (tile + 30) + 30))
        draw.text((8, row * (tile + 30) + 8), label, fill="white")
    result.save(path)


def preview(config, output=Path(".local/preview"), count=20):
    if "mixture" in config:
        from .v2_reports import preview_mixed
        return preview_mixed(config)
    from .engine import setup
    images, train_indices, _, synth = setup(config)
    output.mkdir(parents=True, exist_ok=True)
    # Spread across the full cache so all four source pools are represented.
    selected = train_indices[np.linspace(0, len(train_indices) - 1, count).astype(int)]
    articles, rows = [], []
    for j, index in enumerate(selected):
        source = images[index]
        photo, label = synth.sample(source, config["seed"] + j)
        matrix = as_matrix(label)
        quad = corners(matrix, (224, 224))
        overlay = photo.copy()
        cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (70, 255, 80), 2)
        recovered = cv2.warpPerspective(photo, np.linalg.inv(pixel_matrix(matrix, (224, 224), (224, 224))), (224, 224))
        sid = f"synth{j + 1:02}"
        paths = []
        for name, array in zip(["source", "photo", "label", "recovered"], [source, photo, overlay, recovered]):
            path = output / f"{sid}-{name}.png"
            save_rgb(path, array)
            paths.append(path)
        rows.append((sid + " | source / synthetic / label / inverse", paths))
        figures = "".join(f'<figure><a href="{p.name}"><img src="{p.name}" loading="lazy"></a><figcaption>{title}</figcaption></figure>' for p, title in zip(paths, ["数字原片", "合成翻拍", "自动四角标签", "用真值矩阵还原"]))
        articles.append(f'<article><h2>{sid}</h2><div class="grid four">{figures}</div></article>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>合成标签检查</title><style>' + STYLE + '</style><h1>20张合成示例</h1><p>标签由我们施加的变换精确计算；还原图的模糊来自224像素采样与外观增强。</p>' + "".join(articles))
    for start in range(0, len(rows), 5):
        contact_sheet(rows[start:start + 5], output / f"contact-{start // 5 + 1}.jpg", 224)
    write_json(output / "summary.json", {"count": count, "cache_indices": selected.tolist()})


def compare(config, inputs, checkpoint, output):
    from .model import build, configure, infer, keras
    configure(mixed=False)
    paths = sorted(Path(inputs).glob("p[0-9][0-9][0-9].jpg"))
    if not paths:
        raise ValueError("No anonymous phone images found")
    output = Path(output)
    if (output / "results.local.json").exists():
        raise FileExistsError("Choose a fresh comparison directory")
    output.mkdir(parents=True, exist_ok=True)
    small = np.stack([cv2.resize(read_rgb(p), (224, 224), interpolation=cv2.INTER_AREA) for p in paths])
    predictions = {}
    baseline_name = config.get("baseline_name", "b0")
    candidate_name = config.get("candidate_name", "b1")
    names = [baseline_name, candidate_name]
    for name, weights in [(baseline_name, config["checkpoint"]), (candidate_name, checkpoint)]:
        model, _ = build(Path(weights))
        predictions[name] = infer(model, small, config["batch_size"])
        del model
        keras.backend.clear_session()
    records, articles, rows = [], [], []
    for index, path in enumerate(paths):
        sid = path.stem
        photo = read_rgb(path)
        original = output / f"{sid}-original.jpg"
        shutil.copyfile(path, original)
        group = "IMG" if int(sid[1:]) <= 20 else "weixin"
        record = {"id": sid, "review_group": group}
        display_paths = [original]
        labels = ["原始手机照片"]
        for name in names:
            label = predictions[name][index]
            matrix = as_matrix(label)
            info = {"matrix8": label.tolist()}
            overlay = photo.copy()
            try:
                quad = corners(matrix, (photo.shape[1], photo.shape[0]))
                info["quad"] = quad.tolist()
                cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (50, 255, 90), max(2, photo.shape[1] // 500))
                flat, matrix = rectify(photo, matrix)
                info.update({"geometry_valid": True, "photo_to_flat_matrix": matrix.tolist(), "output_size": [flat.shape[1], flat.shape[0]]})
                target = output / f"{sid}-{name}.png"
                save_rgb(target, flat)
                caption = name.upper() + " · 仅展平"
            except ValueError as error:
                info.update({"geometry_valid": False, "reason": str(error)})
                target = output / f"{sid}-{name}-rejected.jpg"
                save_rgb(target, overlay)
                caption = name.upper() + " · 几何检查拒绝（显示候选框）"
            save_rgb(output / f"{sid}-{name}-quad.jpg", overlay)
            record[name] = info
            display_paths.append(target)
            labels.append(caption)
        records.append(record)
        rows.append((sid + " | original / " + " / ".join(names), display_paths))
        figures = "".join(f'<figure><a href="{p.name}" target="_blank"><img src="{p.name}" loading="lazy"></a><figcaption>{html.escape(title)}</figcaption></figure>' for p, title in zip(display_paths, labels))
        quad_links = ' · '.join(f'<a href="{sid}-{name}-quad.jpg">{name.upper()}四角</a>' for name in names)
        articles.append(f'<article id="{sid}"><h2>{sid} · {group}</h2><div class="grid">{figures}</div><p>{quad_links}</p></article>')
        print(f"compare {sid}", flush=True)
    counts = {name: sum(r[name]["geometry_valid"] for r in records) for name in names}
    summary = {"count": len(records), "geometry_valid": counts, "visual_acceptance": "requires_review", "real_ground_truth_available": False,
               "manual_per_image_correction": False, "demoire_applied": False,
               "groups": {group: {"count": sum(r["review_group"] == group for r in records),
                           "geometry_valid": {name: sum(r["review_group"] == group and r[name]["geometry_valid"] for r in records) for name in names}} for group in ["IMG", "weixin"]}}
    summary["candidate_checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    qualification_note = ""
    evaluation_path = Path(config["run"])/"full-development-evaluation.json"
    if "mixture" in config and evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        if evaluation.get("candidate_sha256") != summary["candidate_checkpoint_sha256"]:
            raise ValueError("Comparison checkpoint does not match the evaluated candidate")
        summary["numeric_selection_gates"] = evaluation["selection_gates"]
        if not all(evaluation["selection_gates"].values()):
            qualification_note = '<p style="color:#ffc47c">该候选未通过运行前设定的全部联合数值门槛；这是原协议的检查记录，手机展平效果需另行评审。背景杂乱不扣分，重点看视角、残余透视、比例与内容完整性。候选数量不代表视角正确率。</p>'
    write_json(output / "results.local.json", records)
    write_json(output / "summary.json", summary)
    navigation = ' · '.join(f'<a href="#{p.stem}">{p.stem}</a>' for p in paths)
    introduction = f'<h1>髋部 PTRN：{len(paths)}张真实照片对照</h1><p>{baseline_name.upper()}基线 / {candidate_name.upper()}候选。相同裁剪、尺寸估计和几何检查。未逐张修正方向，未去摩尔纹。</p><p>几何检查通过 {baseline_name.upper()} {counts[baseline_name]}/{len(paths)}，{candidate_name.upper()} {counts[candidate_name]}/{len(paths)}。这只是数学有效性，不能当作视觉成功率。</p><p>IMG 20张优先；weixin 27张单列。这47张是已查看过的开发照片；没有人工真值四角，不是独立精度测试。</p><nav>{navigation}</nav>'
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>PTRN 髋部微调对照</title><style>' + STYLE + '</style>' + introduction + qualification_note + "".join(articles))
    for start in range(0, len(rows), 6):
        contact_sheet(rows[start:start + 6], output / f"contact-{start // 6 + 1:02}.jpg", 320)
    print(json.dumps(summary), flush=True)
