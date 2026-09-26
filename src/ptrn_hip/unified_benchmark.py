"""Disposable CUDA preflight on fixed real-source training schedule positions.

This records cache equivalence and two 100-update timing probes. It never writes
model weights, advances a formal checkpoint or consumes the 560k training budget.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np

from .data import write_json
from .ratio_data import sha256

EFFECTIVE_BATCH = 64
UPDATES = 100
SLOTS = EFFECTIVE_BATCH * UPDATES
COMPARISON_ATOL = 1e-6
COMPARISON_RTOL = 1e-5
GRADIENT_RELATIVE_L2 = 1e-5
GRADIENT_COSINE_MIN = 1 - 1e-10
GRADIENT_PEAK_SCALE = 2e-6
UPDATE_MAX_ABS = 1e-7
UPDATE_RELATIVE_L2 = 1e-4


def _numeric_comparison(actual, expected):
    """Float64 diagnostics, with a finite convention for an all-zero reference."""
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)
    actual, expected = array(actual), array(expected)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError("Comparison requires equal, nonempty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("Non-finite equivalence comparison")
    difference = actual - expected
    a_norm, e_norm = float(np.linalg.norm(actual)), float(np.linalg.norm(expected))
    cosine = (float(np.dot(actual.ravel(), expected.ravel()) / (a_norm * e_norm))
              if a_norm and e_norm else float(a_norm == e_norm))
    return {"shape": list(actual.shape), "elements": int(actual.size),
            "max_abs_difference": float(np.max(np.abs(difference))),
            "reference_max_abs": float(np.max(np.abs(expected))),
            "actual_max_abs": float(np.max(np.abs(actual))),
            "reference_l2_norm": e_norm, "actual_l2_norm": a_norm,
            "relative_l2": float(np.linalg.norm(difference) / max(e_norm, 1e-30)),
            "cosine_similarity": min(1.0, max(-1.0, cosine)),
            "sign_flip_count": int(np.count_nonzero(actual * expected < 0)),
            "bitwise_equal": bool(np.array_equal(actual, expected))}


def _parameter_comparison(actual, expected):
    if not actual or len(actual) != len(expected):
        raise ValueError("Parameter comparison length mismatch")
    reports = [_numeric_comparison(a, b) for a, b in zip(actual, expected, strict=True)]
    def flatten(values):
        return np.concatenate([v.detach().cpu().numpy().ravel() if hasattr(v, "detach")
                               else np.asarray(v).ravel() for v in values])
    return {"parameters": reports, "global": _numeric_comparison(flatten(actual), flatten(expected))}


def _require_comparison(report, label):
    report["passed"] = all(all(row.values()) for row in report["checks"])
    if not report["passed"]:
        error = AssertionError(label + " equivalence failed")
        error.equivalence_details = report
        raise error
    return report


def _check_gradient_equivalence(actual, expected, *, strict=False):
    """Keep strict same-shape checks; gate changed GEMM shape by scale and direction."""
    report = _parameter_comparison(actual, expected)
    report["mode"] = "same_microbatch_strict" if strict else "cross_batch_float32"
    if strict:
        from .model import torch
        checks = []
        for a, b in zip(actual, expected, strict=True):
            checks.append({"elementwise_close": bool(torch.allclose(
                torch.as_tensor(a), torch.as_tensor(b), atol=COMPARISON_ATOL, rtol=COMPARISON_RTOL))})
        report["limits"] = {"atol": COMPARISON_ATOL, "rtol": COMPARISON_RTOL}
    else:
        report["limits"] = {"relative_l2": GRADIENT_RELATIVE_L2, "cosine_min": GRADIENT_COSINE_MIN,
                            "max_abs_rule": "1e-6 + 2e-6 * reference_parameter_max_abs"}
        checks = [{"relative_l2": row["relative_l2"] <= GRADIENT_RELATIVE_L2,
                   "cosine": row["cosine_similarity"] >= GRADIENT_COSINE_MIN,
                   "scaled_max_abs": row["max_abs_difference"] <= COMPARISON_ATOL + GRADIENT_PEAK_SCALE * row["reference_max_abs"]}
                  for row in report["parameters"]]
    report["checks"] = checks
    return _require_comparison(report, "Head gradient")


def _adam_probe(initial, gradients, device, learning_rate=1e-4):
    """Execute the formal clip+Adam on independent copies, never the model's head."""
    from .model import torch
    _parameter_comparison(gradients, initial)  # shape/finite checks, not equality
    copies = [torch.nn.Parameter(value.detach().to(device).clone()) for value in initial]
    optimizer = torch.optim.Adam([{"params": copies, "lr": learning_rate, "name": "unified_head"}])
    for parameter, gradient in zip(copies, gradients, strict=True):
        parameter.grad = gradient.detach().to(device).clone()
    norm = float(torch.nn.utils.clip_grad_norm_(copies, 1.0, error_if_nonfinite=True))
    optimizer.step()
    updated = [parameter.detach().cpu().clone() for parameter in copies]
    updates = [value - original.detach().cpu() for value, original in zip(updated, initial, strict=True)]
    return updated, updates, {"learning_rate": learning_rate, "betas": [.9, .999], "epsilon": 1e-8,
                              "clip_norm": 1.0, "unclipped_gradient_norm": norm}


def _check_update_equivalence(actual, expected):
    report = _parameter_comparison(actual, expected)
    report["limits"] = {"max_abs_difference": UPDATE_MAX_ABS, "relative_l2": UPDATE_RELATIVE_L2,
                        "sign_flip_count": 0}
    report["checks"] = [{"max_abs": row["max_abs_difference"] <= UPDATE_MAX_ABS,
                         "relative_l2": row["relative_l2"] <= UPDATE_RELATIVE_L2,
                         "direction": row["sign_flip_count"] == 0} for row in report["parameters"]]
    return _require_comparison(report, "First Adam update")


def _head_pass(head, parameters, features, quad, ratio, inverse, micro):
    from .unified_model import forward_features, losses
    for parameter in parameters:
        parameter.grad = None
    count = len(features)
    if count != EFFECTIVE_BATCH or count % micro:
        raise ValueError("Head comparison requires one complete effective batch")
    predictions, means = [], np.zeros(3)
    for start in range(0, count, micro):
        prediction = forward_features(head, features[start:start + micro])
        terms = losses(prediction, quad[start:start + micro], ratio[start:start + micro], 224,
                       inverse_letterbox=inverse[start:start + micro])
        (terms[0] * (micro / count)).backward()
        predictions.append(prediction.detach().cpu().numpy())
        means += np.array([float(value.detach()) for value in terms]) * (micro / count)
    return np.concatenate(predictions), means, [parameter.grad.detach().cpu().clone() for parameter in parameters]


def _check_outputs(actual, expected):
    report = _numeric_comparison(actual, expected)
    try:
        np.testing.assert_allclose(actual, expected, atol=COMPARISON_ATOL, rtol=COMPARISON_RTOL)
    except AssertionError as error:
        error.equivalence_details = report
        raise
    return report


def _bn_state(backbone):
    from .model import keras
    return {value.path: value.value.detach().cpu().clone()
            for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)
            for value in layer.weights}


def _check_bn(backbone, expected):
    from .model import torch
    actual = _bn_state(backbone)
    if actual.keys() != expected.keys() or any(not torch.equal(actual[key], value) for key, value in expected.items()):
        raise RuntimeError("Frozen BatchNorm changed during preflight")


def _gpu_memory():
    from .model import torch
    return {
        "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "gpu_current_allocated_mib": float(torch.cuda.memory_allocated() / 2**20),
    }


def _clear_cuda():
    """Call after a phase returns, once its remaining local graphs are released."""
    from .model import keras, torch
    gc.collect()
    keras.backend.clear_session()
    torch.cuda.empty_cache()


def _cache_consistency(args, schedule):
    """Strict cache check plus numerical/optimizer checks for batch16 versus 64."""
    from .model import keras, torch
    from .unified_model import (build_unified, parameter_groups, tensor_input,
                                feature_predict, losses)
    from .unified_runner import ViewDataset, loader, stamp, cached_split

    model, backbone, head = build_unified(args.weights, args.g0_checkpoint, size=224)
    model.to("cuda")
    groups = parameter_groups(model, backbone, head, "head")
    parameters = [parameter for group in groups for parameter in group["params"]]
    initial_head = [parameter.detach().cpu().clone() for parameter in parameters]
    bn_before = _bn_state(backbone)
    torch.cuda.reset_peak_memory_stats()
    micro, accumulation = 16, EFFECTIVE_BATCH // 16
    data = loader(ViewDataset(args.manifest, schedule, 224, positions=range(EFFECTIVE_BATCH)), micro, args.workers)
    cache, predictions = [], []
    online_losses = np.zeros(3)
    feature_seconds = online_seconds = 0.0
    seen = 0
    diagnostics = {"passed": False, "optimizer_updates": 0, "formal_budget_consumed": 0,
                   "scope": "fixed initial G0 first64 train slots; not trajectory or clinical equivalence",
                   "same_shape": {}, "cross_batch": {}, "first_adam": {}}
    try:
        disk_features, disk_metadata = cached_split(args, "train")
        stored_features = np.array(disk_features[:EFFECTIVE_BATCH])
        stored_metadata = disk_metadata[:EFFECTIVE_BATCH]
        del disk_features, disk_metadata
        for indices, pixels, quad, log_ratio, inverse, encoded, _ in data:
            np.testing.assert_array_equal(indices.numpy(), np.arange(seen, seen + len(indices)))
            if len(indices) != micro:
                raise RuntimeError("Preflight cache comparison batch is incomplete")
            for position, value in zip(indices.tolist(), encoded, strict=True):
                fresh, stored = json.loads(value), stored_metadata[position]
                for key in ("source_index", "view_index", "kind", "mode", "photo_size", "quad_input", "log_ratio", "inverse_letterbox"):
                    if fresh.get(key) != stored.get(key):
                        raise RuntimeError("Disk target metadata differs from fixed re-rendered schedule")
            quad_cuda, ratio_cuda, inverse_cuda = quad.to("cuda"), log_ratio.to("cuda"), inverse.to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = model(tensor_input(pixels.numpy(), "cuda"), training=False)
            terms = losses(prediction, quad_cuda, ratio_cuda, 224, inverse_letterbox=inverse_cuda)
            (terms[0] / accumulation).backward()
            torch.cuda.synchronize()
            online_seconds += time.perf_counter() - started
            online_losses += np.array([float(value.detach()) for value in terms]) / accumulation
            predictions.append(prediction.detach().cpu().numpy())
            started = time.perf_counter()
            features = feature_predict(backbone, pixels.numpy(), "cuda")
            torch.cuda.synchronize()
            feature_seconds += time.perf_counter() - started
            cache.append((features, quad, log_ratio, inverse))
            seen += len(indices)
            stamp(Path(args.run) / "preflight-benchmark-status.json", phase="cache_equivalence",
                  comparison_views=seen, comparison_total=EFFECTIVE_BATCH, formal_training_started=False)
            del prediction, terms, quad_cuda, ratio_cuda, inverse_cuda
            gc.collect()
        if seen != EFFECTIVE_BATCH:
            raise RuntimeError("Cache comparison did not cover 64 fixed views")
        online_gradients = [parameter.grad.detach().cpu().clone() for parameter in parameters]
        online_prediction = np.concatenate(predictions)
        feature_batch = np.concatenate([entry[0] for entry in cache])
        diagnostics["disk_features"] = _check_outputs(stored_features, feature_batch)
        diagnostics["disk_metadata_exact_matches"] = seen
        quad_batch = torch.cat([entry[1] for entry in cache]).to("cuda")
        ratio_batch = torch.cat([entry[2] for entry in cache]).to("cuda")
        inverse_batch = torch.cat([entry[3] for entry in cache]).to("cuda")

        # Same-shape control tests actual disk cache and retains the old strict
        # component tolerance. It cannot be bypassed by the cross-shape gates.
        same_prediction, same_losses, same_gradients = _head_pass(
            head, parameters, stored_features, quad_batch, ratio_batch, inverse_batch, micro)
        diagnostics["same_shape"]["prediction"] = _check_outputs(same_prediction, online_prediction)
        diagnostics["same_shape"]["loss"] = _check_outputs(same_losses, online_losses)
        diagnostics["same_shape"]["gradients"] = _check_gradient_equivalence(same_gradients, online_gradients, strict=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        cached_prediction, cached_losses, cached_gradients = _head_pass(
            head, parameters, stored_features, quad_batch, ratio_batch, inverse_batch, EFFECTIVE_BATCH)
        torch.cuda.synchronize()
        cached_seconds = time.perf_counter() - started
        diagnostics["cross_batch"]["prediction"] = _check_outputs(cached_prediction, online_prediction)
        diagnostics["cross_batch"]["loss"] = _check_outputs(cached_losses, online_losses)
        diagnostics["cross_batch"]["gradients"] = _check_gradient_equivalence(cached_gradients, online_gradients)

        probes = {}
        for name, gradients in (("online16x4", online_gradients), ("cached64", cached_gradients)):
            updated, updates, config = _adam_probe(initial_head, gradients, "cuda", groups[0]["lr"])
            diagnostics["optimizer_updates"] += 1
            with torch.no_grad():
                prediction = (torch.as_tensor(stored_features, device="cuda") @ updated[0].to("cuda")
                              + updated[1].to("cuda"))
                terms = losses(prediction, quad_batch, ratio_batch, 224, inverse_letterbox=inverse_batch)
            probes[name] = (updates, prediction.cpu().numpy(), np.array([float(value) for value in terms]))
            diagnostics["first_adam"][name] = {**config, "loss_quad_ratio": probes[name][2].tolist()}
        diagnostics["first_adam"]["updates"] = _check_update_equivalence(probes["cached64"][0], probes["online16x4"][0])
        diagnostics["first_adam"]["prediction"] = _check_outputs(probes["cached64"][1], probes["online16x4"][1])
        diagnostics["first_adam"]["loss"] = _check_outputs(probes["cached64"][2], probes["online16x4"][2])
        if not all(torch.equal(parameter.detach().cpu(), expected)
                   for parameter, expected in zip(parameters, initial_head, strict=True)):
            raise RuntimeError("Cache comparison changed head parameters")
        _check_bn(backbone, bn_before)
        diagnostics.update({
            "passed": True, "real_training_views": seen, "schedule_positions": [0, EFFECTIVE_BATCH - 1],
            "size": 224, "online_microbatch": micro, "online_accumulation": accumulation,
            "cached_control_microbatch": micro, "cached_head_batch": EFFECTIVE_BATCH, "cached_head_accumulation": 1,
            "precision": "float32", "feature_shape": [EFFECTIVE_BATCH, 1920],
            "features_bytes": int(feature_batch.nbytes),
            "comparison_atol": COMPARISON_ATOL, "comparison_rtol": COMPARISON_RTOL,
            "prediction_max_abs_difference": diagnostics["cross_batch"]["prediction"]["max_abs_difference"],
            "loss_max_abs_difference": diagnostics["cross_batch"]["loss"]["max_abs_difference"],
            "head_gradient_max_abs_difference": diagnostics["cross_batch"]["gradients"]["global"]["max_abs_difference"],
            "prediction_bitwise_equal": bool(np.array_equal(cached_prediction, online_prediction)),
            "online_loss_quad_ratio": online_losses.tolist(), "cached_loss_quad_ratio": cached_losses.tolist(),
            "online_forward_backward_seconds": online_seconds, "feature_cache_forward_seconds": feature_seconds,
            "cached_head_forward_backward_seconds": cached_seconds,
            "timing_scope": "one effective batch; excludes added controls and disposable Adam probes",
            "head_parameters_unchanged": True, "BN_unchanged": True,
            "optimizer_scope": "two independent disposable head copies; no original-head update", **_gpu_memory(),
        })
        write_json(Path(args.run) / "preflight-cache-equivalence.json", diagnostics)
        return diagnostics
    except Exception as error:
        diagnostics["failure"] = {"type": type(error).__name__, "message": str(error)}
        if hasattr(error, "equivalence_details"):
            diagnostics["failed_comparison"] = error.equivalence_details
        error.cache_equivalence = diagnostics
        write_json(Path(args.run) / "preflight-cache-equivalence.json", diagnostics)
        raise
    finally:
        del data, model, backbone, head, groups, parameters
        gc.collect()
        keras.backend.clear_session()
        torch.cuda.empty_cache()


def _joint_benchmark(args, schedule, size, micro):
    """Discardable G0-initialized model, exact future joint-training operations."""
    from .model import keras, torch
    from .unified_model import build_unified, parameter_groups, tensor_input, losses
    from .unified_runner import ViewDataset, loader, stamp, assert_source

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Approved BF16 CUDA benchmark is unavailable")
    if (size, micro) not in {(224, 16), (384, 4)}:
        raise ValueError("Preflight is fixed to J224 16x4 and J384 4x16")
    model, backbone, head = build_unified(args.weights, args.g0_checkpoint, size=size)
    model.to("cuda")
    groups = parameter_groups(model, backbone, head, "last_block")
    parameters = [parameter for group in groups for parameter in group["params"]]
    optimizer = torch.optim.Adam(groups)
    bn_before = _bn_state(backbone)
    accumulation = EFFECTIVE_BATCH // micro
    torch.cuda.reset_peak_memory_stats()
    data = loader(ViewDataset(args.manifest, schedule, size, positions=range(SLOTS)), micro, args.workers)
    group_name = f"J{size}"
    stamp(Path(args.run) / "preflight-benchmark-status.json", phase="benchmark_joint",
          group=group_name, optimizer_step=0, optimizer_updates=UPDATES,
          formal_training_started=False, diagnostic_only=True)
    started = time.perf_counter()
    training = iter(data)
    step_seconds = []
    records = []
    expected_slot = 0
    try:
        for step in range(1, UPDATES + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            means = np.zeros(3)
            for _ in range(accumulation):
                indices, pixels, quad, log_ratio, inverse, _, _ = next(training)
                np.testing.assert_array_equal(indices.numpy(), np.arange(expected_slot, expected_slot + micro))
                if len(indices) != micro:
                    raise RuntimeError("Joint benchmark microbatch is incomplete")
                expected_slot += micro
                quad, log_ratio, inverse = quad.to("cuda"), log_ratio.to("cuda"), inverse.to("cuda")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = model(tensor_input(pixels.numpy(), "cuda"), training=False)
                total, quad_loss, ratio_loss = losses(prediction, quad, log_ratio, size, inverse_letterbox=inverse)
                (total / accumulation).backward()
                means += [float(total.detach()), float(quad_loss.detach()), float(ratio_loss.detach())]
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError("Non-finite diagnostic optimizer weights")
            torch.cuda.synchronize()
            duration = time.perf_counter() - step_started
            step_seconds.append(duration)
            means /= accumulation
            elapsed = time.perf_counter() - started
            record = {"group": group_name, "optimizer_step": step, "samples_seen": step * EFFECTIVE_BATCH,
                      "seconds_per_update": duration, "loss": float(means[0]),
                      "quad_loss": float(means[1]), "ratio_loss": float(means[2]),
                      "gradient_norm": gradient, "elapsed_seconds": elapsed,
                      "views_per_second": step * EFFECTIVE_BATCH / elapsed,
                      "eta_benchmark_seconds": (UPDATES - step) * elapsed / step,
                      "estimated_1250_updates_seconds_excluding_evaluation": 1250 * elapsed / step,
                      **_gpu_memory()}
            records.append(record)
            stamp(Path(args.run) / "preflight-benchmark-status.json", phase="benchmark_joint",
                  optimizer_updates=UPDATES, formal_training_started=False, **record)
            if step == 1 or step % 10 == 0:
                assert_source(args.run)
                print(json.dumps({"phase": "preflight_benchmark", **record}), flush=True)
        if expected_slot != SLOTS:
            raise RuntimeError("Joint benchmark did not consume exactly 6400 schedule slots")
        _check_bn(backbone, bn_before)
        elapsed = time.perf_counter() - started
        return {
            "group": group_name, "size": size, "updates": UPDATES,
            "effective_batch": EFFECTIVE_BATCH, "microbatch": micro, "accumulation": accumulation,
            "samples_seen": SLOTS, "schedule_positions": [0, SLOTS - 1],
            "initialization": "approved G0 plus zero delta; diagnostic model discarded, not stage-1 selected candidate",
            "precision": "bfloat16 autocast, float32 geometry loss",
            "head_learning_rate": 1e-4, "conv5_learning_rate": 1e-5,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "elapsed_seconds_including_loader_startup": elapsed,
            "updates_per_second": UPDATES / elapsed,
            "views_per_second": SLOTS / elapsed,
            "seconds_per_update_mean": float(np.mean(step_seconds)),
            "seconds_per_update_median": float(np.median(step_seconds)),
            "seconds_per_update_p90": float(np.percentile(step_seconds, 90)),
            "post_first_10_seconds_per_update_mean": float(np.mean(step_seconds[10:])),
            "estimated_1250_updates_seconds_excluding_evaluation": 1250 * elapsed / UPDATES,
            "eta_scope": "training data loading, forward, loss, backward, clipping, Adam; excludes development evaluation, model loading, checkpoint export",
            "BN_unchanged": True, "formal_budget_consumed": 0,
            "checkpoints_written": False, "records": records, **_gpu_memory(),
        }
    finally:
        del training, data, optimizer, parameters, groups, model, backbone, head
        gc.collect()
        keras.backend.clear_session()
        torch.cuda.empty_cache()


def benchmark(args):
    """args requires weights, g0_checkpoint, manifest, run and workers; return report.

    Call after source freeze and source/schedule audit, before formal training.
    This function never opens the sealed test or changes formal checkpoints.
    """
    from .unified_runner import assert_source, read_schedule, configure, stamp

    run = Path(args.run)
    inventory = assert_source(run)
    schedule = read_schedule(run, "train")
    if len(schedule) != 80000:
        raise RuntimeError("Preflight requires the frozen 80000-slot training schedule")
    binding = {
        "source_tree_sha256": inventory["tree_sha256"], "source_sha": inventory["git_base"],
        "manifest_sha256": sha256(args.manifest),
        "schedule_sha256": sha256(run / "schedules" / "train.json"),
        "weights_sha256": sha256(args.weights), "g0_sha256": sha256(args.g0_checkpoint),
        "cache_comparison_views": EFFECTIVE_BATCH, "joint_updates_per_size": UPDATES,
        "joint_schedule_slots": SLOTS, "microbatches": {"J224": 16, "J384": 4},
        "workers": int(args.workers),
    }
    path = run / "preflight-benchmark.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("binding") != binding:
            raise RuntimeError("Existing preflight benchmark binding differs from the frozen run")
        if previous.get("state") == "complete":
            stamp(run / "preflight-benchmark-status.json", phase="complete", reused_matching_report=True,
                  formal_training_started=False)
            return previous
    report = {"state": "running", "binding": binding, "formal_training_budget_consumed": 0,
              "diagnostic_optimizer_updates_planned": 2 * UPDATES + 2,
              "diagnostic_optimizer_updates_scope": "joint timing probes plus independent cache Adam copies",
              "joint_optimizer_updates_planned": 2 * UPDATES,
              "cache_adam_probe_updates_planned": 2,
              "diagnostic_training_exposures_planned": 2 * SLOTS,
              "diagnostic_training_exposures_scope": "joint timing probes only; cache Adam copies reuse existing gradients",
              "cache_comparison_unique_views": EFFECTIVE_BATCH,
              "diagnostic_weights_saved": False, "image_files_written": False,
              "sealed_test_opened": False, "joint": {}}
    try:
        report["environment"] = configure()
        report["cache_equivalence"] = _cache_consistency(args, schedule)
        _clear_cuda()
        assert_source(run)
        write_json(path, report)
        print(json.dumps({"phase": "preflight_cache_equivalence", **report["cache_equivalence"]}), flush=True)
        for size, micro in ((224, 16), (384, 4)):
            report["joint"][f"J{size}"] = _joint_benchmark(args, schedule, size, micro)
            _clear_cuda()
            assert_source(run)
            write_json(path, report)
        report["state"] = "complete"
        report["joint_optimizer_updates_completed"] = 2 * UPDATES
        report["cache_adam_probe_updates_completed"] = report["cache_equivalence"]["optimizer_updates"]
        report["diagnostic_optimizer_updates_completed"] = 2 * UPDATES + report["cache_adam_probe_updates_completed"]
        report["diagnostic_training_exposures_completed"] = 2 * SLOTS
        report["estimated_joint_training_seconds_excluding_evaluation"] = sum(
            value["estimated_1250_updates_seconds_excluding_evaluation"] for value in report["joint"].values())
        report["diagnostic_models_discarded"] = True
        write_json(path, report)
        stamp(run / "preflight-benchmark-status.json", phase="complete", formal_training_started=False,
              diagnostic_updates=report["diagnostic_optimizer_updates_completed"], diagnostic_models_discarded=True,
              estimated_joint_training_seconds_excluding_evaluation=report["estimated_joint_training_seconds_excluding_evaluation"])
        return report
    except Exception as error:
        report["state"] = "failed"
        if hasattr(error, "cache_equivalence"):
            report["cache_equivalence"] = error.cache_equivalence
        completed_joint = sum(value["updates"] for value in report["joint"].values())
        completed_cache = report.get("cache_equivalence", {}).get("optimizer_updates", 0)
        status_path = run / "preflight-benchmark-status.json"
        if status_path.exists():
            progress = json.loads(status_path.read_text())
            if progress.get("phase") == "benchmark_joint" and progress.get("group") not in report["joint"]:
                completed_joint += int(progress.get("optimizer_step", 0))
        report["joint_optimizer_updates_completed"] = completed_joint
        report["cache_adam_probe_updates_completed"] = completed_cache
        report["diagnostic_optimizer_updates_completed"] = completed_joint + completed_cache
        report["diagnostic_training_exposures_completed"] = completed_joint * EFFECTIVE_BATCH
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(path, report)
        stamp(run / "preflight-benchmark-status.json", phase="failed", formal_training_started=False,
              error_type=type(error).__name__, message=str(error))
        write_json(run / "preflight-benchmark-failure.json", report["failure"])
        raise
