"""Small, reproducible PTRN fine-tuning run with fixed synthetic validation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .data import load_cache, write_json
from .geometry import geometry_metrics
from .synthesis import Synthesizer


def setup(config):
    if config["image_size"] != 224:
        raise ValueError("This PTRN checkpoint and synthesizer require image_size=224")
    cv2.setNumThreads(1)
    images, train_indices = load_cache(Path(config["data"]), "train")
    _, val_indices = load_cache(Path(config["data"]), "val")
    if "mixture" in config:
        from .mixed_synthesis import MixedSynthesizer
        synth = MixedSynthesizer(Path(config["backgrounds"]), config["rotation_degrees"], config["screen_probability"], config["mixture"])
    else:
        synth = Synthesizer(Path(config["backgrounds"]), config["rotation_degrees"], config["screen_probability"])
    return images, train_indices, val_indices, synth


def batches(images, indices, synth, batch_size, seed):
    """Every source is visited each epoch; final batch wraps within that epoch."""
    rng = np.random.default_rng(seed)
    step = 0
    while True:
        order = rng.permutation(indices)
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            if len(selected) < batch_size:
                selected = np.resize(np.concatenate([selected, order]), batch_size)
            yield synth.batch(images, selected, seed + step * 100003)
            step += 1


def gpu_memory():
    from .model import torch
    if torch.cuda.is_available():
        return {"allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    return {}


def evaluate(model, images, labels, batch_size):
    from .model import infer
    predictions = infer(model, images, batch_size)
    return {"matrix_mse": float(np.mean((predictions - labels) ** 2)), **geometry_metrics(labels, predictions)}


def evaluate_all(config, checkpoint):
    """Final source-image development evaluation, including the selection subset."""
    if "mixture" in config:
        from .experiment import evaluate_all_groups
        return evaluate_all_groups(config, checkpoint)
    from .model import build, configure, keras
    configure(mixed=False)
    images, _, indices, synth = setup(config)
    selected = np.random.default_rng(config["seed"] + 1).permutation(indices)
    x, y = synth.batch(images, selected, config["seed"] + 2)
    result = {"scope": "all_source_image_development_samples_includes_selection_subset",
              "count": len(selected), "selection_subset_count": config["validation_samples"], "precision": "float32"}
    for name, weights in [("b0", config["checkpoint"]), ("b1", checkpoint)]:
        model, _ = build(Path(weights))
        result[name] = evaluate(model, x, y, config["batch_size"])
        print(name + " " + json.dumps(result[name]), flush=True)
        del model
        keras.backend.clear_session()
    write_json(Path(config["run"]) / "full-development-evaluation.json", result)
    return result


def save_weights(model, path):
    temporary = path.with_name("temporary.weights.h5")
    model.save_weights(temporary)
    temporary.replace(path)


def benchmark(config, steps=100):
    from .model import build, configure, preprocess, set_stage
    environment = configure(mixed=True)
    images, indices, _, synth = setup(config)
    model, backbone = build(Path(config["checkpoint"]))
    trainable = set_stage(model, backbone, "last_block", config["finetune_lr"])
    stream = batches(images, indices, synth, config["batch_size"], config["seed"])
    durations = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, stream)
        for step in range(steps + 10):
            start = time.perf_counter()
            x, y = pending.result()
            pending = pool.submit(next, stream)
            model.reset_metrics()
            loss = float(model.train_on_batch(preprocess(x), y))
            if step >= 10:
                durations.append(time.perf_counter() - start)
            if step % 10 == 0:
                print(f"benchmark {step}/{steps + 10}, loss={loss:.5f}", flush=True)
    result = {**environment, "parameters": model.count_params(), "trainable_parameters": trainable,
              "batch_size": config["batch_size"], "measured_steps": steps, "warmup_steps_excluded": 10,
              "seconds_per_step_mean": float(np.mean(durations)), "seconds_per_step_p90": float(np.quantile(durations, .9)),
              "estimated_3000_step_minutes_excluding_validation": float(np.mean(durations) * 3000 / 60), **gpu_memory()}
    result["samples_per_second"] = config["batch_size"] / result["seconds_per_step_mean"]
    output = Path(config["run"]) / f"benchmark-batch{config['batch_size']}.json" if "mixture" in config else Path(".local/benchmark.json")
    write_json(output, result)
    print(json.dumps(result), flush=True)
    return result


def train(config):
    if "mixture" in config:
        from .experiment import train_groups
        return train_groups(config)
    from .model import build, configure, preprocess, set_stage
    environment = configure(mixed=True)
    run = Path(config["run"])
    if (run / "history.jsonl").exists():
        raise FileExistsError("Choose a fresh run directory; existing training is never overwritten")
    run.mkdir(parents=True, exist_ok=True)
    images, train_indices, val_indices, synth = setup(config)
    validation_rng = np.random.default_rng(config["seed"] + 1)
    selected = validation_rng.permutation(val_indices)[:config["validation_samples"]]
    xval, yval = synth.batch(images, selected, config["seed"] + 2)
    np.savez_compressed(run / "validation.local.npz", images=xval, labels=yval, cache_indices=selected)
    model, backbone = build(Path(config["checkpoint"]))
    metadata = {"config": config, "environment": environment, "parameters": model.count_params(),
                "initial_checkpoint_sha256": hashlib.sha256(Path(config["checkpoint"]).read_bytes()).hexdigest(),
                "train_sources": len(train_indices), "validation_sources": len(selected),
                "validation_scope": "fixed_synthetic_source_image_development_set", "stage_trainable_parameters": {}}
    baseline = evaluate(model, xval, yval, config["batch_size"])
    write_json(run / "baseline.json", baseline)
    print("B0 synthetic validation: " + json.dumps(baseline), flush=True)
    best = float("inf")
    best_step = None
    step = 0
    started = time.perf_counter()
    stream = batches(images, train_indices, synth, config["batch_size"], config["seed"])
    total_steps = config["warmup_steps"] + config["finetune_steps"]
    with (run / "history.jsonl").open("w") as log, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, stream)
        for stage, count, lr in [("head", config["warmup_steps"], config["warmup_lr"]),
                                  ("last_block", config["finetune_steps"], config["finetune_lr"])]:
            metadata["stage_trainable_parameters"][stage] = set_stage(model, backbone, stage, lr)
            write_json(run / "metadata.local.json", metadata)
            for _ in range(count):
                x, y = pending.result()
                pending = pool.submit(next, stream)
                model.reset_metrics()
                loss = float(model.train_on_batch(preprocess(x), y))
                if not np.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; run stopped")
                step += 1
                item = {"step": step, "stage": stage, "train_loss": loss, "elapsed_seconds": time.perf_counter() - started}
                if step % config["validate_every"] == 0 or step == total_steps:
                    scores = evaluate(model, xval, yval, config["batch_size"])
                    item["validation"] = scores
                    if scores["corner_nme_mean"] < best:
                        best, best_step = scores["corner_nme_mean"], step
                        save_weights(model, run / "best.weights.h5")
                        write_json(run / "best.json", {"step": step, **scores})
                    save_weights(model, run / "last.weights.h5")
                    print("validation " + json.dumps(item), flush=True)
                elif step % 50 == 0:
                    print(f"{stage} {step}/{total_steps} loss={loss:.5f} elapsed={item['elapsed_seconds']:.1f}s", flush=True)
                log.write(json.dumps(item) + "\n")
                log.flush()
    summary = {"completed_steps": step, "best_step": best_step, "elapsed_seconds": time.perf_counter() - started,
               "synthetic_samples_seen": step * config["batch_size"], "source_passes_approximate": step * config["batch_size"] / len(train_indices),
               "baseline": baseline, "best": json.loads((run / "best.json").read_text()), **gpu_memory()}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary
