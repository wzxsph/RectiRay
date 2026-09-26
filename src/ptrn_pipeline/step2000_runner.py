"""One explicitly forked step-2000 PTRN run with interleaved validation.

This entry point deliberately does not reuse the old runner's strict resume
path. A new checkpoint records its parent, recipe, source, assets and exact
cursor-to-view sequence. It never overwrites the parent or application model.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .branch_checkpoint import (
    branch_sample_ids,
    create_branch_checkpoint,
    mapping_sha256,
    restore_branch_checkpoint,
    save_branch_checkpoint,
    sha256_file,
)
from .resources import gpu_lease
from .step2000_data import BranchSamples, prepare_sequence
from .telemetry import atomic_json
from .training import DoubleBuffer, train_step


PARENT_SHA256 = "016cfc5b3fb97699eb38b6ce8850c7d4802797602d8fba04f233959744120535"
RECIPE_VERSION = "ptrn-step2000-unified-v1"
GIB = 2**30


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "backgrounds", "cache", "parent", "parent_protocol", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--branch-seed", type=int, default=20260926)
    parser.add_argument("--gc-interval", type=int, default=4)
    parser.add_argument("--eval-limit", type=int, help="Diagnostic smoke only; omit for all 5736 development views")
    parser.add_argument("--skip-training", action="store_true", help="Create and verify zero-step branch on CPU")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.gc_interval < 1 or args.seed < 0 or args.branch_seed < 0:
        parser.error("Steps, seeds and GC interval must be positive")
    if args.eval_limit is not None and args.eval_limit < 1:
        parser.error("Positive diagnostic evaluation limit required")
    if args.skip_training and args.resume:
        parser.error("Zero-step preflight cannot resume a training run")
    return args


def _hash_files(root: Path, paths: list[str]) -> dict[str, str]:
    hashes = {}
    for relative in sorted(paths):
        file = root / relative
        if not file.is_file() or not file.resolve().is_relative_to(root.resolve()):
            raise FileNotFoundError(f"Missing or unsafe source file: {relative}")
        hashes[relative] = sha256_file(file)
    return hashes


def _bindings(args, train, dev, sequence_sha: str, old_binding: dict):
    root = args.source_root.resolve()
    code_paths = [str(path.relative_to(root)) for directory in (root / "src/ptrn_hip", root / "src/ptrn_pipeline")
                  for path in directory.glob("*.py")]
    code_paths.extend(["pyproject.toml", "configs/pipeline-v1.json"])
    code = _hash_files(root, code_paths)
    recipe_paths = [f"src/ptrn_hip/{name}" for name in (
        "partial_data.py", "partial_synthesis.py", "capture_synthesis.py",
        "spatial_synthesis.py", "ratio_synthesis.py", "partial_geometry.py")]
    recipe = mapping_sha256(_hash_files(root, recipe_paths))
    asset_paths = ["data/manifest.json", "data/split.json", "data/backgrounds-manifest.json",
                   "weights/b11.weights.h5", "weights/g0.pt"]
    assets = _hash_files(args.bundle.resolve(), asset_paths)
    if old_binding.get("development") != dev.fingerprint:
        raise ValueError("Frozen development fingerprint changed from parent")
    if old_binding.get("effective_batch") != 64 or old_binding.get("stage") != "head":
        raise ValueError("Unexpected parent training contract")
    binding = {**train.fingerprint, "effective_batch": 64, "stage": "head", "loss": old_binding["loss"],
               "development": dev.fingerprint, "recipe_version": RECIPE_VERSION,
               "recipe_sha256": recipe, "sample_manifest_sha256": sequence_sha,
               "code": code, "assets": assets}
    source_hashes = {"recipe": recipe, "code": mapping_sha256(code), "assets": mapping_sha256(assets),
                     "sample_manifest": sequence_sha}
    return binding, source_hashes


def _schedules(steps: int, eval_limit: int | None):
    def count(full):
        return min(eval_limit, 5736) if eval_limit is not None else (None if full else 512)
    schedule = {}
    for at, deadline in ((250, 500), (500, 1000), (1000, 2500), (2500, 3000)):
        if at < steps:
            schedule[at] = {"count": count(at == 1000), "deadline": min(deadline, steps)}
    schedule[steps] = {"count": count(True), "deadline": steps}
    return schedule


def evaluation_quota(remaining: int, remaining_updates: int, cap: int = 32) -> int:
    """Keep a checkpoint's GPU predictions ahead of its stated deadline."""
    if remaining <= 0:
        return 0
    if remaining_updates <= 0:
        return cap
    return min(cap, max(1, math.ceil(remaining / remaining_updates)))


def _status(output: Path, **values):
    atomic_json(output / "runner-status.json", {"time": time.time(), **values})


def _start_dev_prepare(args, output: Path):
    prepare_output = output / "dev-prepare"
    if (prepare_output / "complete.json").exists():
        return None, None, prepare_output
    if prepare_output.exists() and any(prepare_output.iterdir()):
        # Preserve a partial attempt for diagnosis rather than mixing logs.
        suffix = 2
        while (output / f"dev-prepare-attempt-{suffix}").exists():
            suffix += 1
        prepare_output = output / f"dev-prepare-attempt-{suffix}"
    prepare_output.mkdir(parents=True, exist_ok=False)
    log = (prepare_output / "console.log").open("w")
    command = [sys.executable, "-m", "ptrn_pipeline.evaluation", "prepare", "--bundle", str(args.bundle),
               "--backgrounds", str(args.backgrounds), "--cache", str(args.cache),
               "--output", str(prepare_output), "--workers", "4"]
    return (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True), log, prepare_output)


def _new_eval_output(output: Path, label: str) -> Path:
    candidate = output / f"eval-{label}"
    if not candidate.exists():
        return candidate
    attempt = 2
    while (output / f"eval-{label}-attempt-{attempt}").exists():
        attempt += 1
    return output / f"eval-{label}-attempt-{attempt}"


def _launch_job(args, checkpoint: Path, output: Path, label: str, count: int | None,
                workers: int, *, use_graph: bool = False):
    from .async_eval import EvaluationJob
    return EvaluationJob(args.bundle, args.backgrounds, args.cache, checkpoint,
                         _new_eval_output(output, label), batch=1, workers=workers,
                         count=count, use_graph=use_graph)


def _required_count(args, requested):
    return min(args.eval_limit, 5736) if args.eval_limit is not None else (requested or 5736)


def run(args):
    from ptrn_hip.partial_data import PartialDataset
    from ptrn_hip.partial_model import build_partial
    from ptrn_hip.partial_runner import worker_init

    if sha256_file(args.parent) != PARENT_SHA256:
        raise ValueError("Unexpected parent step-2000 checkpoint SHA")
    old_binding = json.loads(args.parent_protocol.read_text())
    train = PartialDataset(args.bundle, args.backgrounds, split="train", seed=args.seed,
                           recipe_version=RECIPE_VERSION, verify=not args.skip_training)
    dev = PartialDataset(args.bundle, args.backgrounds, split="dev", seed=args.seed,
                         verify=not args.skip_training)
    if len(train.ids) != 4114 or len(dev.ids) != 478 or len(dev) != 5736:
        raise ValueError("Unexpected train/development split size")
    output = args.output
    if args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("New branch output must be empty")
        output.mkdir(parents=True, exist_ok=True)
    elif not output.is_dir() or not args.resume.resolve().is_relative_to(output.resolve()):
        raise ValueError("Resume checkpoint must belong to the existing branch output")
    sequence, sequence_sha = prepare_sequence(train, args.steps, output)
    binding, source_hashes = _bindings(args, train, dev, sequence_sha, old_binding)
    if args.resume is None:
        atomic_json(output / "branch-binding.json", binding)
    elif json.loads((output / "branch-binding.json").read_text()) != binding:
        raise ValueError("New code, data, recipe or sample sequence differs from branch binding")

    device = "cpu" if args.skip_training else "cuda"
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("Allocated BF16-capable CUDA GPU required")
    model = build_partial(args.bundle / "weights/b11.weights.h5", args.bundle / "weights/g0.pt",
                          size=224, stage="head").to(device)
    optimizer = torch.optim.Adam(model.optimizer_groups("head"))
    zero = output / "branch-step-0.pt"
    if args.resume is None:
        provenance = create_branch_checkpoint(args.parent, zero, model, optimizer,
            expected_parent_sha256=PARENT_SHA256, expected_parent_binding=old_binding,
            branch_binding=binding, source_hashes=source_hashes, branch_seed=args.branch_seed)
        atomic_json(output / "branch-provenance.json", provenance)
        start, cursor = 0, 0
    else:
        provenance = json.loads((output / "branch-provenance.json").read_text())
        start, cursor, _ = restore_branch_checkpoint(args.resume, model, optimizer,
            expected_binding=binding, expected_provenance=provenance, expected_source_hashes=source_hashes)
    _status(output, status="prepared", branch_step=start, cursor=cursor,
            parent_sha256=PARENT_SHA256, sample_manifest_sha256=sequence_sha)
    if args.skip_training:
        _status(output, status="preflight_complete", branch_step=0, cursor=0,
                parent_sha256=PARENT_SHA256, sample_manifest_sha256=sequence_sha)
        return
    if start >= args.steps:
        raise ValueError("Branch already reached target step")
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    bn_before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                 if "moving_mean" in key or "moving_variance" in key}
    if not bn_before:
        raise RuntimeError("Frozen BatchNorm state missing")
    data = BranchSamples(train, sequence)
    prepare_process, prepare_log, prepare_output = _start_dev_prepare(args, output)
    schedule = _schedules(args.steps, args.eval_limit)
    pending = []
    jobs = []
    errors = []
    current_workers = None
    loader = None
    buffer = None

    def job_state(job):
        state = job.status()
        if state == "failed" and not any(row.get("evaluation") == str(job.output) for row in errors):
            errors.append({"evaluation": str(job.output), "error": "Prediction or metric failure"})
        return state

    def active_jobs():
        return [job for job in jobs if job_state(job) in ("predicting", "metrics")]

    def prepare_state():
        if prepare_process is None:
            return "complete" if (prepare_output / "complete.json").exists() else "unavailable"
        code = prepare_process.poll()
        if code is None:
            return "running"
        if code != 0 or not (prepare_output / "complete.json").exists():
            return "failed"
        return "complete"

    def desired_workers():
        dev_active = prepare_state() == "running"
        metric_workers = sum(job.metric_workers for job in active_jobs())
        return max(4, 14 - (4 if dev_active else 0) - metric_workers)

    def maybe_start_next():
        active = active_jobs()
        if not pending or any(job_state(job) == "predicting" for job in active) or len(active) >= 2:
            return
        label, path, count, deadline = pending.pop(0)
        try:
            torch.cuda.synchronize()
            workers = 2 if prepare_state() == "running" else 4
            if 2 + 4 + (4 if prepare_state() == "running" else 0) + sum(
                prior.metric_workers for prior in active
            ) + workers > 16:
                pending.insert(0, (label, path, count, deadline))
                return
            job = _launch_job(args, path, output, label, count, workers)
            job.deadline = deadline
            job.metric_workers = workers
            jobs.append(job)
        except Exception as exc:
            errors.append({"evaluation": label, "error": repr(exc)})

    def tick_jobs(step):
        for job in jobs:
            if job_state(job) == "predicting":
                quota = evaluation_quota(job.count - job.cursor, job.deadline - step)
                try:
                    job.tick(quota)
                except Exception as exc:
                    errors.append({"evaluation": str(job.output), "error": repr(exc)})
        maybe_start_next()

    def enqueue(step, checkpoint):
        plan = schedule[step]
        pending.append((f"branch-{step:04d}", checkpoint, plan["count"], plan["deadline"]))
        maybe_start_next()

    # Recover skipped evaluation requests after an interruption. A fresh
    # attempt retains the previous partial output for audit.
    if args.resume is not None:
        for step in sorted(schedule):
            checkpoint = output / f"branch-step-{step}.pt"
            if step <= start and checkpoint.exists() and not any(
                (path / "evaluation.json").exists() for path in output.glob(f"eval-branch-{step:04d}*")):
                plan = schedule[step]
                pending.append((f"branch-{step:04d}", checkpoint, plan["count"], plan["deadline"]))

    cursor = start * 64
    begin = time.time()
    try:
        for step in range(start + 1, args.steps + 1):
            workers = desired_workers()
            if buffer is None or workers != current_workers:
                if buffer is not None:
                    buffer.close()
                    del buffer, loader
                    gc.collect()
                loader = DataLoader(data, sampler=branch_sample_ids(cursor, args.steps), batch_size=64,
                    num_workers=workers, pin_memory=True, prefetch_factor=2,
                    multiprocessing_context="spawn", worker_init_fn=worker_init)
                buffer = DoubleBuffer(loader)
                current_workers = workers
            begun = time.perf_counter()
            wait_before = buffer.wait_seconds
            copies_before = buffer.h2d_copy_seconds
            handoffs_before = buffer.h2d_incomplete_handoffs
            batch = next(buffer)
            terms = train_step(model, optimizer, batch, effective_batch=64, step=step,
                               gc_interval=args.gc_interval, loss_options=binding["loss"])
            cursor += 64
            row = {"branch_step": step, "cursor": cursor, "total_wall_seconds": time.perf_counter()-begun,
                   "loader_wait_seconds": buffer.wait_seconds-wait_before,
                   "h2d_copy_seconds": buffer.h2d_copy_seconds-copies_before,
                   "h2d_incomplete_handoffs": buffer.h2d_incomplete_handoffs-handoffs_before,
                   "workers": workers, "peak_reserved_gib": torch.cuda.max_memory_reserved()/GIB, **terms}
            with (output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step % 10 == 0 or step == args.steps:
                print(json.dumps(row), flush=True)
            if row["peak_reserved_gib"] > 22:
                save_branch_checkpoint(output / f"branch-step-{step}-memory-stop.pt", model, optimizer,
                    branch_step=step, branch_cursor=cursor, branch_binding=binding, provenance=provenance)
                raise MemoryError("GPU reserved memory exceeded 22 GiB budget")
            if step % 250 == 0 or step == args.steps:
                if not all(torch.equal(value, model.state_dict()[key].detach().cpu())
                           for key, value in bn_before.items()):
                    raise RuntimeError("Frozen BatchNorm state changed")
                checkpoint = output / f"branch-step-{step}.pt"
                save_branch_checkpoint(checkpoint, model, optimizer, branch_step=step,
                                       branch_cursor=cursor, branch_binding=binding, provenance=provenance)
                if step in schedule:
                    enqueue(step, checkpoint)
                if shutil.disk_usage(args.cache).free < 15 * GIB:
                    raise OSError("Cache disk reserve fell below 15 GiB")
            tick_jobs(step)
            if step % 10 == 0 or step == args.steps:
                _status(output, status="training", branch_step=step, cursor=cursor,
                        elapsed_seconds=time.time()-begin, workers=workers,
                        dev_prepare=prepare_state(),
                        evaluations={str(job.output): {"status": job_state(job), "predicted": job.cursor,
                                                       "count": job.count} for job in jobs},
                        evaluation_errors=errors)
    finally:
        if buffer is not None:
            buffer.close()
            del buffer, loader
        if sys.exc_info()[0] is not None:
            if prepare_process is not None and prepare_process.poll() is None:
                try:
                    os.killpg(prepare_process.pid, signal.SIGTERM)
                    prepare_process.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
            for job in jobs:
                if job.child.poll() is None:
                    atomic_json(job.output / "inference-failed.json",
                                {"error": "Training process stopped before evaluation completed"})
                    try:
                        os.killpg(job.child.pid, signal.SIGTERM)
                        job.child.wait(timeout=10)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
                if job.owns_bank:
                    job.bank.close()
        if prepare_log is not None:
            prepare_log.close()
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    # The parent is needed as a same-exam baseline, but its preparation must
    # not hold the new branch's training GPU idle at startup.
    pending.append(("parent-2000", args.parent, _required_count(args, None), args.steps))
    while pending or any(job_state(job) in ("predicting", "metrics") for job in jobs):
        maybe_start_next()
        for job in jobs:
            if job_state(job) == "predicting":
                try:
                    job.tick(32)
                except Exception as exc:
                    errors.append({"evaluation": str(job.output), "error": repr(exc)})
        _status(output, status="final_evaluation", branch_step=args.steps, cursor=cursor,
                evaluations={str(job.output): {"status": job_state(job), "predicted": job.cursor,
                                               "count": job.count} for job in jobs}, evaluation_errors=errors)
        time.sleep(.05)
    if prepare_process is not None:
        prepare_process.wait()
    if prepare_state() == "failed":
        errors.append({"dev_prepare": "Reference development cache preparation failed"})
    diagnostic_complete = not errors and all(job_state(job) == "complete" for job in jobs)
    # Reference CUDA Graph capture is not safe while the training DataLoader's
    # pin-memory/H2D threads are live. Perform the exact numerical audit only
    # after all training input workers have been joined.
    reference_jobs = []
    references = [("reference-parent-2000", args.parent)]
    if args.steps >= 1000:
        references.append(("reference-branch-1000", output / "branch-step-1000.pt"))
    references.append((f"reference-branch-{args.steps:04d}", output / f"branch-step-{args.steps}.pt"))
    for label, checkpoint in references:
        if not checkpoint.exists():
            errors.append({"evaluation": label, "error": "Required checkpoint missing"})
            continue
        try:
            job = _launch_job(args, checkpoint, output, label, _required_count(args, None),
                              workers=4, use_graph=True)
            reference_jobs.append(job)
            while job_state(job) in ("predicting", "metrics"):
                if job_state(job) == "predicting":
                    job.tick(32)
                _status(output, status="reference_evaluation", branch_step=args.steps,
                        cursor=cursor, evaluation=label, predicted=job.cursor,
                        count=job.count, evaluation_errors=errors)
                time.sleep(.05)
        except Exception as exc:
            errors.append({"evaluation": label, "error": repr(exc)})
    complete = not errors and all(job_state(job) == "complete" for job in jobs + reference_jobs)
    required_full = {f"eval-branch-{step:04d}" for step in (1000, args.steps) if step <= args.steps}
    required_full.add("eval-parent-2000")
    full = (args.eval_limit is None and complete and all(
        any(job.output.name == name and job.count == 5736 and job_state(job) == "complete"
            for job in reference_jobs) for name in {name.replace("eval-", "eval-reference-") for name in required_full}))
    atomic_json(output / "completion.json", {"training_complete": True, "branch_steps": args.steps,
        "development_evaluations_complete": complete,
        "diagnostic_evaluations_complete": diagnostic_complete,
        "reference_graph_evaluations_complete": all(job_state(job) == "complete" for job in reference_jobs),
        "full_development_evaluated": full,
        "effectiveness_verified": False, "evaluation_errors": errors,
        "elapsed_seconds": time.time()-begin})
    _status(output, status="complete" if complete else "evaluation_failed", branch_step=args.steps,
            cursor=cursor, evaluation_errors=errors)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("KERAS_BACKEND", "torch")
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_training:
        run(args)
        return
    with gpu_lease(args.cache.parent):
        lock_path = args.output.with_suffix(".runner.lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(args)
            except BaseException as exc:
                if args.output.exists():
                    _status(args.output, status="failed", error=repr(exc))
                raise


if __name__ == "__main__":
    main()
