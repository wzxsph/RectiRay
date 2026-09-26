"""Gated C-screening runner; explicit failed-B exploration remains labeled as such.

Source pixels, caches, predictions and checkpoints remain below the ignored local
run directory. This module never caches full spatial features.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import time

import cv2
import numpy as np

from .data import write_json
from .ratio_data import Sources, sha256
from .ratio_synthesis import letterbox, project
from .spatial_data import audit_schedule, build_schedule, validate_partition
from .spatial_synthesis import sample


VARIANTS = ("baseline_head", "spatial_head", "spatial_conv5", "spatial_coverage", "spatial_grid")
SEEDS = (20260923, 20260924, 20260925)
CHECKPOINTS = (1250, 5000)
TRAIN_VIEWS, DEV_VIEWS, SIZE = 80000, 5258, 224
EFFECTIVE_BATCH, WARMUP = 64, 500
LOSS_WEIGHTS = {"coordinate": 1., "ratio": .25, "coverage": 1., "grid": 10.,
                "missed": 3., "background": 1., "invalid": 1.,
                "photo_boundary": 5., "angle": 0.}
CACHE_FIELDS = {"pixels": (np.uint8, (SIZE, SIZE, 3)),
                "quad": (np.float32, (4, 2)),
                "inverse": (np.float32, (3, 3)),
                "log_ratio": (np.float32, ())}


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _b_decisions(location: Path) -> list[dict]:
    """Validate every present four-arm decision against its saved evaluations.

    ``location`` may be the run directory or a decision file inside it. A file
    argument never narrows the scan to one editable pass list: all recognized
    head/conv5/cached-revision decisions in that run are checked.
    """
    location = Path(location)
    run = location if location.is_dir() else location.parent
    stages = (("B", "head"), ("B-conv5", "last_block"),
              ("B-conv5-cached", "last_block"))
    from .spatial_model import ARMS
    decisions = []
    for folder, expected_stage in stages:
        for path in sorted(run.glob(f"{folder}-decision-*.json")):
            try:
                step = int(path.stem.rsplit("-", 1)[-1])
            except ValueError as error:
                raise RuntimeError("B decision filename lacks a shared checkpoint step") from error
            if step not in (500, 2000, 5000):
                raise RuntimeError("B decision uses an unapproved shared checkpoint")
            decision = json.loads(path.read_text())
            arms, passed = decision.get("arms"), decision.get("pass_arms")
            if (decision.get("stage") != expected_stage or decision.get("step") != step
                    or not isinstance(arms, dict) or set(arms) != ARMS
                    or not isinstance(passed, list) or len(set(passed)) != len(passed)
                    or not set(passed) <= ARMS):
                raise RuntimeError("B decision has inconsistent stage, step or arms")
            recomputed = []
            evidence_sha256 = {}
            for arm in sorted(ARMS):
                eval_path = run / folder / arm / f"eval-{step:04d}.json"
                if not eval_path.is_file():
                    raise RuntimeError("B decision lacks four-arm shared-step evaluation evidence")
                evidence_sha256[arm] = sha256(eval_path)
                evidence = json.loads(eval_path.read_text())
                record = arms[arm]
                if (evidence.get("arm") != arm or evidence.get("stage") != expected_stage
                        or evidence.get("step") != step or evidence.get("count") != 128):
                    raise RuntimeError("B evaluation evidence has mismatched arm, stage or step")
                for key in ("corner_nme_mean", "corner_nme_p90", "valid_quads"):
                    if record.get(key) != evidence.get(key):
                        raise RuntimeError("B decision differs from saved evaluation evidence")
                mean, p90, legal = (evidence.get(key) for key in
                                    ("corner_nme_mean", "corner_nme_p90", "valid_quads"))
                if (not isinstance(mean, (int, float)) or not isinstance(p90, (int, float))
                        or not math.isfinite(mean) or not math.isfinite(p90)
                        or not isinstance(legal, int) or not 0 <= legal <= 128):
                    raise RuntimeError("B evaluation evidence has invalid fit metrics")
                fit = mean <= .005 and p90 <= .01 and legal == 128
                if record.get("fit_pass") is not fit or evidence.get("fit_pass") is not fit:
                    raise RuntimeError("B fit_pass contradicts prespecified NME/P90/128-legal gate")
                if fit:
                    recomputed.append(arm)
            if set(passed) != set(recomputed):
                raise RuntimeError("B pass_arms contradicts four-arm evaluation evidence")
            global_arm = min((name for name in ARMS if name.startswith("gap_")),
                             key=lambda name: (arms[name]["corner_nme_mean"], name))
            spatial_arm = min((name for name in ARMS if name.startswith("spatial_")),
                              key=lambda name: (arms[name]["corner_nme_mean"], name))
            decisions.append({"path": str(path.resolve()), "sha256": sha256(path),
                              "evidence_sha256": evidence_sha256,
                              "pass_arms": list(passed), "global_arm": global_arm,
                              "spatial_arm": spatial_arm, "b_stage": expected_stage,
                              "b_step": step, "folder": folder,
                              "scope": "B_fixed_training_set_fit_only_not_generalization",
                              "stage_priority": stages.index((folder, expected_stage))})
    if not decisions:
        raise RuntimeError("C training requires B shared-checkpoint decision evidence")
    return decisions


def require_b_pass(location: Path) -> dict:
    """Select the latest evidenced shared B milestone with a genuine pass."""
    passing = [item for item in _b_decisions(location) if item["pass_arms"]]
    if not passing:
        raise RuntimeError("B did not pass: pass_arms is empty at every shared checkpoint")
    selected = max(passing, key=lambda item: (item["b_step"], item["stage_priority"]))
    return {key: value for key, value in selected.items() if key not in ("stage_priority", "folder")}


def select_b_decision(location: Path, *, allow_failed_b: bool = False) -> dict:
    if not allow_failed_b:
        return require_b_pass(location)
    decisions = _b_decisions(location)
    expected = {(folder, step) for folder in ("B", "B-conv5-cached")
                for step in (500, 2000, 5000)}
    actual = {(item["folder"], item["b_step"]) for item in decisions}
    if len(decisions) != 6 or actual != expected:
        raise RuntimeError("Failed-B exploration requires exactly six shared B decisions")
    if any(item["pass_arms"] for item in decisions):
        raise RuntimeError("Failed-B exploration requires all six B decisions to have empty pass_arms")
    by_key = {(item["folder"], item["b_step"]): item for item in decisions}
    global_decision = by_key[("B-conv5-cached", 5000)]
    spatial_decision = by_key[("B", 5000)]
    return {"path": global_decision["path"], "sha256": global_decision["sha256"],
            "evidence_sha256": global_decision["evidence_sha256"], "pass_arms": [],
            "global_arm": global_decision["global_arm"],
            "spatial_arm": spatial_decision["spatial_arm"],
            "b_stage": "failed_B_two_rounds", "b_step": 5000,
            "scope": "exploratory_operator_requested_after_B_failure",
            "authorization": "2026-09-23 user requested full training",
            "arm_selection": {"global": "B-conv5-cached/5000 minimum gap_ corner_nme_mean",
                              "spatial": "B/5000 minimum spatial_ corner_nme_mean"},
            "all_b_decisions": {f'{item["folder"]}-{item["b_step"]:04d}':
                                {"decision_sha256": item["sha256"],
                                 "evaluation_sha256": item["evidence_sha256"],
                                 "pass_arms": item["pass_arms"]} for item in decisions}}


def variant_spec(variant: str, decision: dict) -> dict:
    if variant not in VARIANTS:
        raise ValueError("Unknown C-screening variant")
    return {"variant": variant,
            "arm": decision["global_arm"] if variant == "baseline_head" else decision["spatial_arm"],
            "stage": "head" if variant in ("baseline_head", "spatial_head") else "last_block",
            "coverage": variant in ("spatial_coverage", "spatial_grid"),
            "grid": variant == "spatial_grid"}


def _source_binding() -> dict:
    root = Path(__file__).parent
    names = ("spatial_full_runner.py", "spatial_data.py", "spatial_synthesis.py",
             "spatial_model.py", "spatial_objectives.py", "spatial_evaluation.py",
             "spatial_geometry.py", "unified_geometry.py", "unified_evaluation.py")
    names += ("data.py", "model.py", "ratio_data.py", "ratio_model.py",
              "ratio_synthesis.py", "unified_model.py", "unified_data.py",
              "unified_synthesis.py", "spatial_runner.py")
    files = {name: sha256(root / name) for name in names}
    return {"files": files, "sha256": _digest_json(files)}


def _partition(args):
    split = json.loads((args.prior_run / "split.local.json").read_text())
    validate_partition(split)
    prior = json.loads((args.prior_run / "original-integrity-before.json").read_text())
    if sha256(args.manifest) != prior.get("manifest_sha256"):
        raise RuntimeError("Manifest differs from audited inherited partition")
    return split


def _audit_originals(args, *, expected_count=4620):
    """Rehash every source; persist only a digest of manifest-approved hashes."""
    sources = Sources(args.manifest, cache_size=1)
    if len(sources.records) != expected_count:
        raise RuntimeError("Inherited original count changed")
    approved = []
    for ordinal, record in enumerate(sources.records):
        expected = record["source_sha256"]
        try:
            actual = sha256(record["source_path"])
        except OSError as error:
            raise RuntimeError(f"Inherited original ordinal {ordinal} could not be read: "
                               f"{type(error).__name__}") from None
        if actual != expected:
            raise RuntimeError(f"Inherited original ordinal {ordinal} changed")
        approved.append(expected)
    return {"manifest_sha256": sha256(args.manifest),
            "source_count": len(approved), "source_hashes_sha256": _digest_json(approved)}


def _verify_original_audit(args, *, expected_count=4620):
    path = args.run / "C" / "cache" / "original-audit.json"
    if not path.is_file():
        raise RuntimeError("C cache lacks completed post-generation original audit")
    recorded = json.loads(path.read_text())
    current = _audit_originals(args, expected_count=expected_count)
    if recorded != current:
        raise RuntimeError("Original digest changed after C cache preparation")
    return current


def _cache_root(args, split):
    return args.run / "C" / "cache" / split


def _cache_schema(args, split, rows):
    return {"version": "spatial-c-input224-v2", "split": split, "views": len(rows),
            "input_size": SIZE, "manifest_sha256": sha256(args.manifest),
            "partition_sha256": sha256(args.prior_run / "split.local.json"),
            "schedule_sha256": _digest_json(rows), "source_binding": _source_binding(),
            "fields": {name: {"dtype": np.dtype(dtype).name, "shape": list(shape)}
                       for name, (dtype, shape) in CACHE_FIELDS.items()},
            "GT_ratio": "stored_for_supervision_and_explicit_oracle_diagnostic_only"}


def _check_or_write(path: Path, value: dict):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen configuration differs: {path.name}")
    else:
        write_json(path, value)


def _render(sources, row):
    source_index = int(row["source_index"])
    source = sources.read(source_index)
    view = sample(source, source_index, int(row["view_index"]), kind=row["kind"],
                  mode=row.get("mode", "front"), occupancy_bin=row.get("occupancy_bin", 1),
                  level=row.get("level", "clear"), pure_roll=row.get("pure_roll", False),
                  domain=row["domain"])
    image = view.clean if row.get("condition") in ("clean", "clear") else view.noisy
    pixels, lb = letterbox(image, SIZE)
    quad = project(view.quad, lb).astype(np.float32)
    inverse = np.linalg.inv(lb).astype(np.float32)
    info = {**row, **view.info, "source_index": source_index,
            "photo_size": [image.shape[1], image.shape[0]],
            "source_size": view.info["source_size"],
            "source_to_photo": view.info["source_to_photo"],
            "quad_photo": np.asarray(view.quad).tolist(),
            "quad_input": quad.tolist(), "log_ratio": float(view.log_ratio),
            "inverse_letterbox": inverse.tolist()}
    return pixels, quad, inverse, np.float32(view.log_ratio), info


class _RenderDataset:
    def __init__(self, manifest, rows, positions):
        self.manifest = str(manifest)
        self.rows = rows
        self.positions = positions
        self.sources = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, ordinal):
        if self.sources is None:
            self.sources = Sources(self.manifest, cache_size=2)
        index = self.positions[ordinal]
        pixels, quad, inverse, log_ratio, info = _render(self.sources, self.rows[index])
        return index, pixels, quad, inverse, log_ratio, json.dumps(info)


def _worker_init(_):
    cv2.setNumThreads(1)


def _prepare_split(args, split, indices):
    from .model import torch
    rows = build_schedule(indices, split=split)
    audit = audit_schedule(rows, indices=indices, split=split)
    expected = TRAIN_VIEWS if split == "train" else DEV_VIEWS
    if len(rows) != expected or (split == "train" and audit["sources"] != 4114):
        raise RuntimeError("C schedule omitted inherited source views")
    folder = _cache_root(args, split)
    folder.mkdir(parents=True, exist_ok=True)
    schema = _cache_schema(args, split, rows)
    _check_or_write(folder / "schema.json", schema)
    schedule_path = folder / "schedule.json"
    if not schedule_path.exists():
        write_json(schedule_path, rows)
    elif _digest_json(json.loads(schedule_path.read_text())) != schema["schedule_sha256"]:
        raise RuntimeError("Cached schedule changed")
    completion = folder / "complete.json"
    if completion.exists():
        result = json.loads(completion.read_text())
        if result["schema_sha256"] != _digest_json(schema):
            raise RuntimeError("Completed cache schema changed")
        return result
    n = len(rows)
    metadata_db = sqlite3.connect(folder / "metadata.sqlite")
    metadata_db.execute("CREATE TABLE IF NOT EXISTS records (idx INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    fields = {}
    for name, (dtype, tail) in CACHE_FIELDS.items():
        path = folder / f"{name}.npy"
        fields[name] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                                  dtype=dtype, shape=(n, *tail))
        if fields[name].shape != (n, *tail) or fields[name].dtype != np.dtype(dtype):
            raise RuntimeError("Existing C cache array has incompatible shape or dtype")
    done_path = folder / "done.npy"
    done_existed = done_path.exists()
    done = np.lib.format.open_memmap(done_path, mode="r+" if done_existed else "w+",
                                     dtype=np.bool_, shape=(n,))
    if done.shape != (n,) or done.dtype != np.dtype(np.bool_):
        raise RuntimeError("Existing C cache completion bitmap is incompatible")
    if not done_existed:
        done[:] = False
        done.flush()
    present = {int(row[0]) for row in metadata_db.execute("SELECT idx FROM records")}
    missing = sorted((i for i in range(n) if not done[i] or i not in present),
                     key=lambda i: (rows[i]["source_index"], i))
    dataset = _RenderDataset(args.manifest, rows, missing)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=args.workers,
                                         shuffle=False, worker_init_fn=_worker_init,
                                         pin_memory=False,
                                         **({"multiprocessing_context": "spawn", "persistent_workers": True,
                                             "prefetch_factor": 2} if args.workers else {}))
    for count, batch in enumerate(loader, start=1):
        positions, pixels, quad, inverse, log_ratio, metadata = batch
        ids = positions.numpy()
        fields["pixels"][ids] = pixels.numpy()
        fields["quad"][ids] = quad.numpy()
        fields["inverse"][ids] = inverse.numpy()
        fields["log_ratio"][ids] = log_ratio.numpy()
        for field in fields.values():
            field.flush()
        metadata_db.executemany("INSERT OR REPLACE INTO records(idx,payload) VALUES(?,?)",
                                [(int(i), value) for i, value in zip(ids, metadata, strict=True)])
        metadata_db.commit()
        done[ids] = True
        done.flush()
        if count % 100 == 0:
            print(json.dumps({"phase": "cache", "split": split,
                              "completed": int(done.sum()), "total": n}), flush=True)
    if not done.all() or metadata_db.execute("SELECT COUNT(*) FROM records").fetchone()[0] != n:
        raise RuntimeError("Incomplete C image cache")
    metadata = [json.loads(payload) for _, payload in
                metadata_db.execute("SELECT idx,payload FROM records ORDER BY idx")]
    metadata_db.close()
    if split == "train":
        sources = {int(row["source_index"]) for row in metadata}
        if sources != set(map(int, indices)):
            raise RuntimeError("Full training source coverage failed")
    write_json(folder / "metadata.json", metadata)
    result = {"schema_sha256": _digest_json(schema), "views": n,
              "source_count": len(set(int(row["source_index"]) for row in rows)),
              "split": split, "schedule_audit": audit,
              "sha256": {name: sha256(folder / f"{name}.npy") for name in CACHE_FIELDS},
              "metadata_sha256": sha256(folder / "metadata.json")}
    write_json(completion, result)
    return result


def prepare(args):
    select_b_decision(args.b_decision, allow_failed_b=getattr(args, "allow_failed_b", False))
    partition = _partition(args)
    args.run.mkdir(parents=True, exist_ok=True)
    if (not (_cache_root(args, "train") / "complete.json").exists()
            and shutil.disk_usage(args.run).free < 15 * 2**30):
        raise RuntimeError("At least 15 GiB free disk is required for the fixed C input cache")
    before = _audit_originals(args)
    caches = {split: _prepare_split(args, split, partition[split]) for split in ("train", "dev")}
    after = _audit_originals(args)
    if before != after:
        raise RuntimeError("Original digest changed while generating the C input cache")
    _check_or_write(args.run / "C" / "cache" / "original-audit.json", after)
    for split in ("train", "dev"):
        _, metadata, _ = _load_cache(args, split)
        report = _realized_cache_audit(metadata, split, partition[split])
        _check_or_write(_cache_root(args, split) / "realized-audit.json", report)
        if not report["passed"]:
            raise RuntimeError(f"{split} realized film occupancy or source coverage failed")
    return caches


def _realized_cache_audit(metadata, split, indices):
    """Recompute aggregate photo geometry from the entire frozen metadata cache."""
    photo_count = above_half = 0
    minimum = 1.0
    bins = Counter()
    keep_sources, photo_sources = set(), set()
    expected_sources = set(map(int, indices))
    for row in metadata:
        source = int(row["source_index"])
        if source not in expected_sources:
            raise RuntimeError("C cache metadata contains a source outside its partition")
        if row["kind"] != "photo":
            keep_sources.add(source)
            continue
        photo_sources.add(source)
        photo_count += 1
        width, height = row["photo_size"]
        quad = np.asarray(row["quad_photo"], dtype=np.float64)
        if (not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0
                or quad.shape != (4, 2) or not np.isfinite(quad).all()):
            raise RuntimeError("C cache photo geometry is invalid")
        area = .5 * abs(np.dot(quad[:, 0], np.roll(quad[:, 1], -1))
                          - np.dot(quad[:, 1], np.roll(quad[:, 0], -1)))
        occupancy = area / (width * height)
        if not math.isfinite(occupancy) or occupancy > 1 + 1e-9:
            raise RuntimeError("C cache photo occupancy is invalid")
        minimum = min(minimum, float(occupancy))
        above_half += int(occupancy > .5)
        bins[f"{math.floor(occupancy / .05) * .05:.2f}"] += 1
    fraction = above_half / photo_count if photo_count else 0.0
    passed = bool(photo_count > 0 and minimum >= .30 - 1e-9 and fraction >= .8
                  and keep_sources == expected_sources and photo_sources == expected_sources)
    return {"version": "spatial-c-realized-v1", "split": split, "views": len(metadata),
            "photo_count": photo_count, "photo_above_half": above_half,
            "photo_above_half_fraction": fraction,
            "film_occupancy_min": minimum if photo_count else None,
            "actual_film_occupancy_005": dict(sorted(bins.items())),
            "source_count": len(expected_sources), "sources_with_keep": len(keep_sources),
            "sources_with_photo": len(photo_sources), "passed": passed,
            "scope": "realized_cache_metadata_aggregate_local_only"}


def _load_cache(args, split):
    folder = _cache_root(args, split)
    completion = json.loads((folder / "complete.json").read_text())
    rows = build_schedule(_partition(args)[split], split=split)
    schema = _cache_schema(args, split, rows)
    saved_schedule = json.loads((folder / "schedule.json").read_text())
    if (json.loads((folder / "schema.json").read_text()) != schema
            or completion.get("schema_sha256") != _digest_json(schema)
            or _digest_json(saved_schedule) != schema["schedule_sha256"]):
        raise RuntimeError("C cache schema or schedule changed")
    fields = {}
    for name, (dtype, tail) in CACHE_FIELDS.items():
        path = folder / f"{name}.npy"
        if sha256(path) != completion["sha256"][name]:
            raise RuntimeError(f"{split} cached {name} changed")
        fields[name] = np.load(path, mmap_mode="r")
        if fields[name].dtype != np.dtype(dtype) or fields[name].shape != (completion["views"], *tail):
            raise RuntimeError(f"{split} cached {name} shape or dtype changed")
    if sha256(folder / "metadata.json") != completion["metadata_sha256"]:
        raise RuntimeError("C cache metadata changed")
    metadata = json.loads((folder / "metadata.json").read_text())
    if len(metadata) != completion["views"] or completion["views"] != len(rows):
        raise RuntimeError("C cache count changed")
    if any(any(info.get(key) != row[key] for key in ("source_index", "view_index", "kind", "domain"))
           for info, row in zip(metadata, rows, strict=True)):
        raise RuntimeError("C cache metadata differs from frozen schedule")
    return fields, metadata, completion


def _coverage_photo_geometry(fields, metadata, indices):
    """Per-view acquisition bounds from the frozen cache, never GT film edges."""
    inverse = np.asarray(fields["inverse"][indices], dtype=np.float32)
    photo_sizes = np.asarray([metadata[int(index)]["photo_size"] for index in indices],
                             dtype=np.float32)
    if (inverse.shape != (len(indices), 3, 3)
            or photo_sizes.shape != (len(indices), 2)
            or not np.isfinite(inverse).all() or not np.isfinite(photo_sizes).all()
            or (photo_sizes < 2).any()):
        raise RuntimeError("Cached C inverse letterbox or photo sizes are invalid")
    return inverse, photo_sizes


def _orders(seed: int):
    """Paired deterministic orders; first 16k preserve full source coverage."""
    orders = []
    for epoch in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2901]))
        if epoch == 0:
            order = np.r_[rng.permutation(16000), 16000 + rng.permutation(64000)]
        else:
            order = rng.permutation(TRAIN_VIEWS)
        orders.append(order)
    return np.concatenate(orders).reshape(-1, EFFECTIVE_BATCH)


def _pooled_feature(features):
    return features[-1].mean(dim=(2, 3)) if isinstance(features, (tuple, list)) else features


def _safe_log_edge(corners):
    """Finite ratio for initially collapsed heatmaps; invalid q still fails eval."""
    from .model import torch
    q = corners.detach().float()
    lengths = torch.linalg.vector_norm(torch.roll(q, -1, dims=1) - q, dim=-1)
    return (torch.maximum(lengths[:, 0], lengths[:, 2]).clamp_min(1e-3).log()
            - torch.maximum(lengths[:, 1], lengths[:, 3]).clamp_min(1e-3).log())


def _evaluate(model, ratio_module, fields, metadata, microbatch, device, *, ratio_active):
    from .model import torch
    from .ratio_model import tensor_input
    from .spatial_evaluation import evaluate_quad_predictions
    from .unified_model import _collect_unreachable_cycles
    predictions, ratios = [], []
    model.eval()
    ratio_module.eval()
    with torch.no_grad():
        for start in range(0, len(metadata), microbatch):
            pixels = np.array(fields["pixels"][start:start + microbatch])
            features = model.extract_features(tensor_input(pixels, device))
            corners = model.forward_features(features).float()
            delta = ratio_module(corners, _pooled_feature(features)) if ratio_active else corners.new_zeros(len(corners))
            predictions.append(corners.cpu().numpy())
            ratios.append((_safe_log_edge(corners) + delta).cpu().numpy())
            _collect_unreachable_cycles()
    quad = np.concatenate(predictions)
    log_ratio = np.concatenate(ratios)
    summary, rows = evaluate_quad_predictions(quad, log_ratio, metadata, SIZE,
                                               ratio_method="prediction")
    oracle, _ = evaluate_quad_predictions(quad, None, metadata, SIZE,
                                          ratio_method="oracle_gt_diagnostic")
    return {"prediction": summary, "diagnostic_only_GT_ratio": oracle,
            "ratio_uses_GT_in_prediction": False, "rows": len(rows)}, quad, log_ratio


def _checkpoint_config(args, spec, decision, train_cache, dev_cache):
    from .spatial_model import contract
    return {"version": "spatial-c-screen-v2", "variant": spec, "seed": args.seed,
            "size": SIZE, "effective_batch": EFFECTIVE_BATCH, "microbatch": args.microbatch,
            "warmup_steps": WARMUP, "loss_weights": LOSS_WEIGHTS, "checkpoints": list(CHECKPOINTS),
            "initialization": "approved_G0_per_variant_seed", "contract": contract(SIZE),
            "b_decision_sha256": decision["sha256"], "b_pass_arms": decision["pass_arms"],
            "b_evidence_sha256": decision["evidence_sha256"],
            "b_stage": decision["b_stage"], "b_step": decision["b_step"],
            "b_scope": decision["scope"],
            "b_authorization": decision.get("authorization"),
            "b_all_decisions": decision.get("all_b_decisions"),
            "b_arm_selection": decision.get("arm_selection"),
            "allow_failed_b": bool(getattr(args, "allow_failed_b", False)),
            "manifest_sha256": sha256(args.manifest), "b11_sha256": sha256(args.weights),
            "g0_sha256": sha256(args.g0_checkpoint), "source_binding": _source_binding(),
            "partition_sha256": sha256(args.prior_run / "split.local.json"),
            "original_audit_sha256": sha256(args.run / "C" / "cache" / "original-audit.json"),
            "train_cache": train_cache["schema_sha256"], "dev_cache": dev_cache["schema_sha256"],
            "train_realized_audit_sha256": sha256(_cache_root(args, "train") / "realized-audit.json"),
            "dev_realized_audit_sha256": sha256(_cache_root(args, "dev") / "realized-audit.json"),
            "dev_scope": "source_exposed_development_not_patient_independent",
            "GT_ratio": "training_label_and_separate_oracle_diagnostic_only"}


def _verified_checkpoint_evaluation(folder: Path, step: int, sidecar: dict) -> dict:
    """Require the sidecar-bound evaluation before accepting a resume target."""
    path = folder / f"eval-{step:04d}.json"
    expected = sidecar.get("evaluation_sha256")
    if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
        raise RuntimeError("C resume evaluation is missing or differs from sidecar SHA")
    report = json.loads(path.read_text())
    if report.get("step") != step or report.get("checkpoint_sha256") != sidecar.get("sha256"):
        raise RuntimeError("C resume evaluation does not identify its checkpoint")
    return report


def _training_log_scalars(samples: dict, gradient):
    """Copy all detached step diagnostics from the device in one transfer."""
    from .model import torch
    names = list(samples)
    values = torch.stack([torch.stack(samples[name]).sum() for name in names]
                         + [gradient.detach()]).cpu().tolist()
    return dict(zip(names, values[:-1], strict=True)), values[-1]


def _parameters_finite(params) -> bool:
    """Keep the per-step finite guard, with one host synchronization."""
    from .model import torch
    return bool(torch.stack([torch.isfinite(p).all() for p in params]).all().item())


def train(args):
    """GPU training requires a B pass or an explicitly bound failed-B exploration."""
    decision = select_b_decision(args.b_decision,
                                 allow_failed_b=getattr(args, "allow_failed_b", False))
    spec = variant_spec(args.variant, decision)
    if args.seed not in SEEDS or args.target not in CHECKPOINTS:
        raise ValueError("Unapproved C seed or target")
    if args.microbatch not in (1, 2, 4, 8, 16) or EFFECTIVE_BATCH % args.microbatch:
        raise ValueError("Microbatch must divide effective batch 64")
    _partition(args)
    _verify_original_audit(args)
    from .model import torch, keras
    from .ratio_model import tensor_input
    from .spatial_model import build_spatial, coordinate_loss, parameter_groups
    from .spatial_objectives import (RatioResidual, coverage_loss, grid_geometry_loss,
                                     ratio_residual_loss)
    from .unified_model import _collect_unreachable_cycles
    if not torch.cuda.is_available():
        raise RuntimeError("C training requires the approved GPU after B passes")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    keras.utils.set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_fields, train_metadata, train_cache = _load_cache(args, "train")
    dev_fields, dev_metadata, dev_cache = _load_cache(args, "dev")
    for split, metadata in (("train", train_metadata), ("dev", dev_metadata)):
        report = _realized_cache_audit(metadata, split, _partition(args)[split])
        _check_or_write(_cache_root(args, split) / "realized-audit.json", report)
        if not report["passed"]:
            raise RuntimeError(f"{split} realized film occupancy or source coverage failed")
    if len(train_metadata) != TRAIN_VIEWS or len(dev_metadata) != DEV_VIEWS:
        raise RuntimeError("C train/dev cache size mismatch")
    if set(row["source_index"] for row in train_metadata) & set(row["source_index"] for row in dev_metadata):
        raise RuntimeError("Development source leaked into training cache")
    out = args.run / "C" / args.variant / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    config = _checkpoint_config(args, spec, decision, train_cache, dev_cache)
    _check_or_write(out / "config.json", config)
    model = build_spatial(args.weights, args.g0_checkpoint, spec["arm"], size=SIZE,
                          stage=spec["stage"], initial_quad=np.asarray(train_fields["quad"]).mean(axis=0)).to("cuda")
    torch.manual_seed(args.seed + 1000)
    ratio_module = RatioResidual(feature_dim=1920, hidden=64, size=SIZE).to("cuda")
    groups = parameter_groups(model, spec["stage"])
    groups.append({"params": list(ratio_module.parameters()), "lr": 1e-4, "name": "ratio_residual"})
    optimizer = torch.optim.Adam(groups)
    completed = 0
    completed_report = None
    checkpoints = sorted(out.glob("step-*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        sidecar = json.loads(latest.with_suffix(".json").read_text())
        if (sha256(latest) != sidecar.get("sha256") or sidecar.get("config") != config
                or sidecar.get("step") != int(latest.stem.removeprefix("step-"))):
            raise RuntimeError("C resume checkpoint SHA/config mismatch")
        completed_report = _verified_checkpoint_evaluation(out, int(sidecar["step"]), sidecar)
        payload = torch.load(latest, map_location="cuda", weights_only=True)
        if payload["config"] != config or payload["step"] != sidecar["step"]:
            raise RuntimeError("C resume payload mismatch")
        model.load_state_dict(payload["model"], strict=True)
        ratio_module.load_state_dict(payload["ratio"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in payload["cuda_rng"]])
        completed = int(payload["step"])
    if completed > args.target:
        raise RuntimeError("Cannot rewind C screening run")
    if completed == args.target:
        return completed_report
    bn_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if "moving_mean" in key or "moving_variance" in key}
    if not bn_before:
        raise RuntimeError("Frozen BN state check has no buffers")
    orders = _orders(args.seed)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(completed + 1, args.target + 1):
        model.train()
        ratio_module.train()
        optimizer.zero_grad(set_to_none=True)
        batch = orders[step - 1]
        active = step > WARMUP
        log_step = step % 50 == 0
        snapshots = defaultdict(list) if log_step else None
        for offset in range(0, EFFECTIVE_BATCH, args.microbatch):
            ids = batch[offset:offset + args.microbatch]
            pixels = np.array(train_fields["pixels"][ids])
            truth = torch.as_tensor(np.array(train_fields["quad"][ids]), device="cuda")
            truth_log_ratio = torch.as_tensor(np.array(train_fields["log_ratio"][ids]), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model.extract_features(tensor_input(pixels, "cuda"))
                quad = model.forward_features(features)
            coordinate = coordinate_loss(quad, truth, SIZE)
            total = coordinate
            terms = {"coordinate": coordinate}
            if active:
                delta = ratio_module(quad, _pooled_feature(features))
                log_ratio = _safe_log_edge(quad) + delta
                ratio = ratio_residual_loss(log_ratio, truth_log_ratio)
                total = total + LOSS_WEIGHTS["ratio"] * ratio
                terms["ratio"] = ratio
                if spec["coverage"]:
                    inverse, photo_sizes = _coverage_photo_geometry(train_fields, train_metadata, ids)
                    cover = coverage_loss(quad, truth, (SIZE, SIZE),
                                          inverse_letterbox=torch.as_tensor(inverse, device="cuda"),
                                          photo_sizes=torch.as_tensor(photo_sizes, device="cuda"),
                                          missed_weight=LOSS_WEIGHTS["missed"],
                                          background_weight=LOSS_WEIGHTS["background"],
                                          invalid_weight=LOSS_WEIGHTS["invalid"],
                                          photo_boundary_weight=LOSS_WEIGHTS["photo_boundary"])
                    total = total + LOSS_WEIGHTS["coverage"] * cover.total
                    terms["coverage"] = cover.total
                    terms["photo_boundary"] = cover.photo_boundary
                if spec["grid"]:
                    grid = grid_geometry_loss(quad, truth, log_ratio.exp(), truth_log_ratio.exp(),
                                              (SIZE, SIZE), angle_weight=LOSS_WEIGHTS["angle"])
                    total = total + LOSS_WEIGHTS["grid"] * grid.total
                    terms["grid"] = grid.total
            (total * (args.microbatch / EFFECTIVE_BATCH)).backward()
            if log_step:
                for name, value in terms.items():
                    snapshots[name].append(value.detach() * (args.microbatch / EFFECTIVE_BATCH))
            _collect_unreachable_cycles()
        gradient = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        if not _parameters_finite(params):
            raise FloatingPointError("C model parameter became non-finite")
        if log_step:
            losses, gradient_norm = _training_log_scalars(snapshots, gradient)
            print(json.dumps({"phase": "C_train", "variant": args.variant, "seed": args.seed,
                              "step": step, "losses": losses, "gradient_norm": gradient_norm,
                              "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                              "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                              "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                              "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                              "elapsed_seconds": time.monotonic() - start}), flush=True)
        if step in CHECKPOINTS:
            current_decision = select_b_decision(args.b_decision,
                                                 allow_failed_b=getattr(args, "allow_failed_b", False))
            if current_decision != decision:
                raise RuntimeError("B selection/evidence changed during C training")
            _partition(args)
            _verify_original_audit(args)
            for key, value in bn_before.items():
                if not torch.equal(value, model.state_dict()[key].detach().cpu()):
                    raise RuntimeError("Frozen BN changed during C training")
            if _source_binding() != config["source_binding"]:
                raise RuntimeError("C source changed after configuration lock")
            target = out / f"step-{step:04d}.pt"
            temporary = out / f"step-{step:04d}.tmp"
            torch.save({"model": model.state_dict(), "ratio": ratio_module.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step, "config": config,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
                       temporary)
            report, prediction, log_ratio = _evaluate(model, ratio_module, dev_fields, dev_metadata,
                                                      args.microbatch, "cuda", ratio_active=True)
            report.update({"variant": args.variant, "seed": args.seed, "step": step,
                           "scope": "exposed_source_development_not_independent",
                           "checkpoint_sha256": sha256(temporary)})
            write_json(out / f"eval-{step:04d}.json", report)
            np.save(out / f"quad-{step:04d}.npy", prediction)
            np.save(out / f"logratio-{step:04d}.npy", log_ratio)
            write_json(target.with_suffix(".json"), {"sha256": sha256(temporary),
                       "step": step, "config": config, "evaluation_sha256": sha256(out / f"eval-{step:04d}.json")})
            temporary.replace(target)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--g0-checkpoint", type=Path)
    parser.add_argument("--b-decision", type=Path)
    parser.add_argument("--allow-failed-b", action="store_true",
                        help="Explicit exploratory C after all six evidenced B decisions failed")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--target", type=int, choices=CHECKPOINTS, default=CHECKPOINTS[0])
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.b_decision is not None and args.b_decision.resolve().parent != args.run.resolve():
        parser.error("--b-decision must be in --run; the gate scans all recognized milestones")
    args.b_decision = args.run
    if args.command == "prepare":
        print(json.dumps(prepare(args)), flush=True)
    else:
        if args.weights is None or args.g0_checkpoint is None or args.variant is None:
            parser.error("train requires --weights, --g0-checkpoint and --variant")
        select_b_decision(args.b_decision, allow_failed_b=args.allow_failed_b)
        from .spatial_runner import gpu_lock
        with gpu_lock():
            print(json.dumps(train(args)), flush=True)


if __name__ == "__main__":
    main()
