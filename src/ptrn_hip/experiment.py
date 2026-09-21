"""B1.1 grouped development evaluation and bounded fine-tuning."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .data import write_json
from .geometry import as_matrix, corners, validate_quad
from .mixed_synthesis import summarize_samples


def validation_groups(config, images, indices, synth):
    """One fixed example per source per group, preserving original identity sizes."""
    selected = np.random.default_rng(config["seed"]+1).permutation(indices)
    manifest = json.loads((Path(config["data"])/"manifest.local.json").read_text())
    records = {r["cache_index"]: r for r in manifest["records"]}
    groups = {}
    for number, mode in enumerate(synth.MODES):
        rows = [synth.sample_record(images[int(i)], config["seed"]+2+number*1000003+j*1009, mode=mode) for j, i in enumerate(selected)]
        groups[mode] = {"images": np.stack([r[0] for r in rows]), "labels": np.stack([r[1] for r in rows]),
                        "sizes": np.array([records[int(i)]["original_size"] if mode == "identity" else [224, 224] for i in selected]),
                        "cache_indices": selected, "sampling": summarize_samples([r[2] for r in rows])}
    return groups


def pixel_metrics(truth, prediction, sizes):
    """All finite corner errors remain in the denominator, including bad quads."""
    errors, overlaps, coverage, invalid, rows = [], [], [], 0, []
    for y, p, size in zip(truth, prediction, sizes, strict=True):
        width, height = map(int, size)
        error, overlap, cover, reasons = 1., 0., 0., []
        try:
            a = corners(as_matrix(y), (width, height)).astype(np.float32)
            b = corners(as_matrix(p), (width, height)).astype(np.float32)
            error = float(np.linalg.norm(a-b, axis=1).mean()/np.hypot(width, height))
            reasons = validate_quad(b, (width, height))
            if cv2.isContourConvex(b) and cv2.contourArea(b, oriented=True) > 0:
                intersection, _ = cv2.intersectConvexConvex(a, b)
                area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
                overlap = max(0., float(intersection))/max(area_a+area_b-intersection, 1e-8)
                cover = max(0., float(intersection))/max(area_a, 1e-8)
        except (ValueError, cv2.error):
            reasons = ["nonfinite_or_degenerate"]
        errors.append(error)
        overlaps.append(overlap)
        coverage.append(cover)
        invalid += bool(reasons)
        rows.append({"corner_nme": error, "quad_iou": overlap, "target_canvas_coverage": cover, "reasons": reasons})
    return {"count": len(rows), "corner_nme_mean": float(np.mean(errors)), "corner_nme_median": float(np.median(errors)),
            "corner_nme_p90": float(np.quantile(errors, .9)), "quad_iou_mean": float(np.mean(overlaps)),
            "target_canvas_coverage_mean": float(np.mean(coverage)), "invalid": int(invalid),
            "within_1_percent": sum(e <= .01 for e in errors), "within_2_percent": sum(e <= .02 for e in errors)}, rows


def evaluate_groups(model, groups, batch_size, output=None):
    from .model import infer
    result = {}
    for name, group in groups.items():
        prediction = infer(model, group["images"], batch_size)
        metrics, rows = pixel_metrics(group["labels"], prediction, group["sizes"])
        result[name] = {"matrix_mse": float(np.mean((prediction-group["labels"])**2)), **metrics}
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output/f"{name}-predictions.local.npz", predictions=prediction, labels=group["labels"], sizes=group["sizes"], cache_indices=group["cache_indices"])
            write_json(output/f"{name}-results.local.json", rows)
    return result


def selection_gates(scores, baseline):
    identity, mild, hard = (scores[k] for k in ("identity", "mild", "hard"))
    return {"identity": identity["corner_nme_mean"] <= .01 and identity["corner_nme_p90"] <= .02 and identity["invalid"] == 0,
            "mild": mild["corner_nme_mean"] < baseline["mild"]["corner_nme_mean"] and mild["invalid"] <= baseline["mild"]["invalid"],
            "hard": hard["corner_nme_mean"] <= baseline["hard"]["corner_nme_mean"]*1.1 and hard["invalid"] <= baseline["hard"]["invalid"]}


def decision_gate(scores, baseline):
    return (scores["identity"]["corner_nme_mean"] <= baseline["identity"]["corner_nme_mean"]*.5
            and scores["mild"]["corner_nme_mean"] <= baseline["mild"]["corner_nme_mean"])


def grouped_batches(images, indices, synth, batch_size, seed):
    rng = np.random.default_rng(seed)
    step = 0
    while True:
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            selected = order[start:start+batch_size]
            if len(selected) < batch_size:
                selected = np.resize(np.concatenate([selected, order]), batch_size)
            yield synth.batch_records(images, selected, seed+step*100003)
            step += 1


def train_groups(config):
    from .engine import setup, gpu_memory, save_weights
    from .model import build, configure, preprocess, set_stage
    environment = configure(mixed=True)
    run = Path(config["run"])
    if (run/"history.jsonl").exists():
        raise FileExistsError("Choose a fresh run; existing training is never overwritten")
    run.mkdir(parents=True, exist_ok=True)
    images, train_indices, val_indices, synth = setup(config)
    groups = validation_groups(config, images, val_indices, synth)
    for name, group in groups.items():
        np.savez_compressed(run/f"validation-{name}.local.npz", **{k: v for k, v in group.items() if k != "sampling"})
    write_json(run/"validation-sampling.json", {k: v["sampling"] for k, v in groups.items()})
    model, backbone = build(Path(config["checkpoint"]))
    metadata = {"config": config, "environment": environment, "parameters": model.count_params(),
                "initial_checkpoint_sha256": hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((Path(config["data"])/"manifest.local.json").read_bytes()).hexdigest(),
                "train_sources": len(train_indices), "validation_sources": len(val_indices),
                "validation_scope": "fixed_source_image_development_not_patient_independent", "stage_trainable_parameters": {}}
    baseline = evaluate_groups(model, groups, config["batch_size"])
    write_json(run/"baseline.json", baseline)
    print("B1 grouped baseline " + json.dumps(baseline), flush=True)
    best, best_step = float("inf"), None
    diagnostic_best, diagnostic_step = float("inf"), None
    step, counts = 0, Counter()
    sampling_records = []
    started = time.perf_counter()
    stream = grouped_batches(images, train_indices, synth, config["batch_size"], config["seed"])
    total_steps = config["warmup_steps"]+config["finetune_steps"]
    stop_reason = "completed_budget"
    with (run/"history.jsonl").open("x") as log, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, stream)
        for stage, count, lr in [("head", config["warmup_steps"], config["warmup_lr"]), ("last_block", config["finetune_steps"], config["finetune_lr"])]:
            metadata["stage_trainable_parameters"][stage] = set_stage(model, backbone, stage, lr)
            write_json(run/"metadata.local.json", metadata)
            for _ in range(count):
                x, y, sampled = pending.result()
                pending = pool.submit(next, stream)
                model.reset_metrics()
                loss = float(model.train_on_batch(preprocess(x), y))
                if not np.isfinite(loss):
                    write_json(run/"failure.json", {"step": step, "reason": "nonfinite_loss"})
                    raise FloatingPointError("Non-finite loss")
                step += 1
                sampling_records.extend(sampled)
                counts.update(r["mode"] for r in sampled)
                item = {"step": step, "stage": stage, "train_loss": loss, "elapsed_seconds": time.perf_counter()-started,
                        "sample_counts": dict(counts)}
                if step % config["validate_every"] == 0 or step == total_steps or step == config["decision_step"]:
                    scores = evaluate_groups(model, groups, config["batch_size"])
                    gates = selection_gates(scores, baseline)
                    item.update(validation=scores, selection_gates=gates)
                    if all(gates.values()) and scores["mild"]["corner_nme_mean"] < best:
                        best, best_step = scores["mild"]["corner_nme_mean"], step
                        save_weights(model, run/"best.weights.h5")
                        write_json(run/"best.json", {"step": step, "validation": scores, "selection_gates": gates})
                    diagnostic_score = scores["identity"]["corner_nme_mean"]+scores["mild"]["corner_nme_mean"]
                    if diagnostic_score < diagnostic_best:
                        diagnostic_best, diagnostic_step = diagnostic_score, step
                        save_weights(model, run/"diagnostic.weights.h5")
                        write_json(run/"diagnostic.json", {"step": step, "validation": scores, "selection_gates": gates, "qualified": all(gates.values())})
                    save_weights(model, run/"last.weights.h5")
                    if step == config["decision_step"]:
                        item["continue_gate"] = decision_gate(scores, baseline)
                        if not item["continue_gate"]:
                            stop_reason = "decision_gate_failed"
                    print("validation " + json.dumps(item), flush=True)
                elif step % 50 == 0:
                    print(f"{stage} {step}/{total_steps} loss={loss:.6f} elapsed={item['elapsed_seconds']:.1f}s", flush=True)
                log.write(json.dumps(item)+"\n")
                log.flush()
                if step % 50 == 0:
                    write_json(run/"status.json", {**item, **gpu_memory(), "planned_steps": total_steps})
                if stop_reason != "completed_budget":
                    break
            if stop_reason != "completed_budget":
                break
    summary = {"completed_steps": step, "planned_steps": total_steps, "stop_reason": stop_reason, "best_step": best_step,
               "diagnostic_step": diagnostic_step, "qualified_checkpoint_available": best_step is not None,
               "elapsed_seconds": time.perf_counter()-started, "synthetic_samples_seen": sum(counts.values()),
               "source_passes_approximate": sum(counts.values())/len(train_indices), "sampling": summarize_samples(sampling_records),
               "baseline": baseline, "last_validation": scores, **gpu_memory()}
    write_json(run/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def evaluate_all_groups(config, checkpoint):
    from .engine import setup
    from .model import build, configure, keras
    configure(mixed=False)
    run = Path(config["run"])
    images, _, indices, synth = setup(config)
    groups = validation_groups(config, images, indices, synth)
    results = {"scope": "all_source_image_development_includes_selection", "precision": "float32", "count_per_group": len(indices),
               "candidate_checkpoint": str(checkpoint), "candidate_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    for name, weights in [(config["baseline_name"], config["checkpoint"]), (config["candidate_name"], checkpoint)]:
        model, _ = build(Path(weights))
        results[name] = evaluate_groups(model, groups, config["batch_size"], run/"evaluation"/name)
        print(name+" "+json.dumps(results[name]), flush=True)
        del model
        keras.backend.clear_session()
    results["selection_gates"] = selection_gates(results[config["candidate_name"]], results[config["baseline_name"]])
    write_json(run/"full-development-evaluation.json", results)
    return results
