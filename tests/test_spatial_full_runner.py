"""CPU-only C runner gates, partition and local-cache integrity checks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ptrn_hip import spatial_full_runner as full
from ptrn_hip.spatial_data import build_schedule, validate_partition
from ptrn_hip.ratio_data import sha256


def _decision(run, passed, *, folder="B", step=2000, forged=False):
    errors = {"gap_h8": .018, "gap_quad": .012, "spatial_quad": .004,
              "spatial_heatmap": .003}
    arms = {name: {"step": 5000, "arm": name, "stage": "head", "count": 128,
                   "corner_nme_mean": error, "corner_nme_p90": .008 if name in passed else .02,
                   "valid_quads": 128 if name in passed else 100,
                   "fit_pass": name in passed}
            for name, error in errors.items()}
    stage = "head" if folder == "B" else "last_block"
    for name, record in arms.items():
        record.update(step=step, stage=stage)
        target = run / folder / name / f"eval-{step:04d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record))
    if forged:
        passed = list(passed) + ["gap_quad"]
    path = run / f"{folder}-decision-{step:04d}.json"
    path.write_text(json.dumps({"stage": stage, "step": step,
                                "arms": arms, "pass_arms": passed}))
    return path


def test_c_train_cannot_bypass_empty_or_inconsistent_b_decision(tmp_path):
    decision = _decision(tmp_path, [])
    with pytest.raises(RuntimeError, match="pass_arms is empty"):
        full.require_b_pass(tmp_path)
    # The train entrypoint rejects before touching CUDA, data or checkpoint paths.
    with pytest.raises(RuntimeError, match="pass_arms is empty"):
        full.train(SimpleNamespace(b_decision=tmp_path))
    decision.unlink()
    _decision(tmp_path, ["spatial_quad"], forged=True)
    with pytest.raises(RuntimeError, match="pass_arms contradicts"):
        full.require_b_pass(tmp_path)
    _decision(tmp_path, ["spatial_quad"])
    result = full.require_b_pass(tmp_path)
    assert result["pass_arms"] == ["spatial_quad"]
    assert result["global_arm"] == "gap_quad"
    assert result["spatial_arm"] == "spatial_heatmap"
    assert full.variant_spec("baseline_head", result)["arm"] == "gap_quad"
    assert full.variant_spec("spatial_grid", result) == {
        "variant": "spatial_grid", "arm": "spatial_heatmap", "stage": "last_block",
        "coverage": True, "grid": True}


def test_early_2000_pass_and_conv5_cached_revision_with_eval_evidence(tmp_path):
    _decision(tmp_path, [], step=500)
    _decision(tmp_path, ["spatial_quad"], step=2000)
    result = full.require_b_pass(tmp_path)
    assert result["b_stage"] == "head" and result["b_step"] == 2000
    _decision(tmp_path, ["spatial_heatmap"], folder="B-conv5-cached", step=500)
    assert full.require_b_pass(tmp_path)["b_step"] == 2000  # Latest checkpoint wins.
    _decision(tmp_path, ["spatial_heatmap"], folder="B-conv5", step=5000)
    result = full.require_b_pass(tmp_path)
    assert result["b_stage"] == "last_block" and result["b_step"] == 5000
    assert result["pass_arms"] == ["spatial_heatmap"]
    # A decision row cannot overwrite contradictory saved evidence.
    path = tmp_path / "B-conv5/spatial_heatmap/eval-5000.json"
    evidence = json.loads(path.read_text())
    evidence["valid_quads"] = 127
    path.write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match="differs from saved evaluation evidence"):
        full.require_b_pass(tmp_path)


def test_explicit_failed_b_requires_all_six_bound_decisions(tmp_path):
    for folder in ("B", "B-conv5-cached"):
        for step in (500, 2000, 5000):
            _decision(tmp_path, [], folder=folder, step=step)
    with pytest.raises(RuntimeError, match="pass_arms is empty"):
        full.select_b_decision(tmp_path)
    selected = full.select_b_decision(tmp_path, allow_failed_b=True)
    assert selected["pass_arms"] == []
    assert selected["global_arm"] == "gap_quad"
    assert selected["spatial_arm"] == "spatial_heatmap"
    assert selected["scope"] == "exploratory_operator_requested_after_B_failure"
    assert selected["authorization"] == "2026-09-23 user requested full training"
    assert len(selected["all_b_decisions"]) == 6
    assert all(len(item["evaluation_sha256"]) == 4
               for item in selected["all_b_decisions"].values())
    (tmp_path / "B-conv5-cached/spatial_heatmap/eval-5000.json").unlink()
    with pytest.raises(RuntimeError, match="four-arm shared-step"):
        full.select_b_decision(tmp_path, allow_failed_b=True)


def test_explicit_failed_b_rejects_forged_or_missing_round(tmp_path):
    for folder in ("B", "B-conv5-cached"):
        for step in (500, 2000, 5000):
            _decision(tmp_path, [], folder=folder, step=step)
    (tmp_path / "B-decision-0500.json").unlink()
    with pytest.raises(RuntimeError, match="exactly six"):
        full.select_b_decision(tmp_path, allow_failed_b=True)
    _decision(tmp_path, [], folder="B", step=500)
    path = tmp_path / "B-decision-5000.json"
    decision = json.loads(path.read_text())
    decision["arms"]["gap_h8"]["fit_pass"] = True
    path.write_text(json.dumps(decision))
    with pytest.raises(RuntimeError, match="fit_pass contradicts"):
        full.select_b_decision(tmp_path, allow_failed_b=True)


def test_full_schedule_covers_train_only_and_first_16000(tmp_path):
    split = {"train": list(range(4114)), "dev": list(range(4114, 4592)),
             "quarantine": list(range(4592, 4620))}
    assert validate_partition(split)["source_overlap"] == 0
    rows = build_schedule(split["train"], split="train")
    assert len(rows) == full.TRAIN_VIEWS
    assert {r["source_index"] for r in rows} == set(split["train"])
    assert not {r["source_index"] for r in rows} & set(split["dev"])
    first = rows[:16000]
    assert {r["source_index"] for r in first if r["kind"] == "photo"} == set(split["train"])
    assert {r["source_index"] for r in first if r["kind"] != "photo"} == set(split["train"])
    orders = full._orders(full.SEEDS[0])
    assert orders.shape == (5000, 64)
    np.testing.assert_array_equal(np.sort(orders[:250].reshape(-1)), np.arange(16000))


@pytest.mark.parametrize("microbatch", (1, 2, 4, 8, 16))
def test_training_log_scalars_preserve_effective_batch_mean(microbatch):
    from ptrn_hip.model import torch

    snapshots = {"coordinate": [], "ratio": []}
    expected = {name: 0. for name in snapshots}
    for offset in range(0, full.EFFECTIVE_BATCH, microbatch):
        scale = microbatch / full.EFFECTIVE_BATCH
        terms = {"coordinate": torch.tensor(offset / 10 + 1., requires_grad=True),
                 "ratio": torch.tensor(offset / 100 + .5, requires_grad=True)}
        for name, value in terms.items():
            snapshots[name].append(value.detach() * scale)
            expected[name] += float(value.detach()) * scale
    losses, norm = full._training_log_scalars(snapshots, torch.tensor(2.5))
    assert losses == pytest.approx(expected, abs=1e-5)
    assert norm == 2.5
    assert all(not value.requires_grad for values in snapshots.values() for value in values)


def test_parameter_finite_guard_rejects_nan_and_infinity():
    from ptrn_hip.model import torch

    first = torch.nn.Parameter(torch.tensor([1., 2.]))
    second = torch.nn.Parameter(torch.tensor([3.]))
    assert full._parameters_finite((first, second))
    with torch.no_grad():
        first[0] = float("nan")
    assert not full._parameters_finite((first, second))
    with torch.no_grad():
        first[0] = 1.
        second[0] = float("inf")
    assert not full._parameters_finite((first, second))


def test_development_evaluation_collects_after_every_batch(monkeypatch):
    from ptrn_hip.model import torch
    from ptrn_hip import spatial_evaluation, unified_model

    forward_sizes, collections = [], []

    class Model:
        def eval(self):
            pass

        def extract_features(self, pixels):
            forward_sizes.append(len(pixels))
            return pixels.mean(dim=(1, 2)).repeat(1, 640)

        def forward_features(self, features):
            corners = torch.tensor([[10., 10.], [200., 10.],
                                    [200., 200.], [10., 200.]])
            return corners.expand(len(features), -1, -1)

    class Ratio:
        def eval(self):
            pass

        def __call__(self, corners, pooled):
            return corners.new_zeros(len(corners))

    def fake_evaluate(quad, log_ratio, metadata, size, *, ratio_method):
        assert quad.shape == (3, 4, 2) and size == full.SIZE
        assert log_ratio is None or (log_ratio.shape == (3,) and np.isfinite(log_ratio).all())
        return {"ratio_method": ratio_method}, list(metadata)

    monkeypatch.setattr(unified_model, "_collect_unreachable_cycles",
                        lambda: collections.append(len(forward_sizes)))
    monkeypatch.setattr(spatial_evaluation, "evaluate_quad_predictions", fake_evaluate)
    fields = {"pixels": np.zeros((3, full.SIZE, full.SIZE, 3), np.uint8)}
    report, quad, log_ratio = full._evaluate(Model(), Ratio(), fields, [{}, {}, {}], 2,
                                             "cpu", ratio_active=True)
    assert forward_sizes == [2, 1]
    assert collections == [1, 2]
    assert report["rows"] == 3 and quad.shape == (3, 4, 2) and log_ratio.shape == (3,)


def test_config_lock_and_sha_bound_resume(tmp_path):
    assert full.LOSS_WEIGHTS["missed"] == 3.
    assert full.LOSS_WEIGHTS["background"] == 1.
    assert full.LOSS_WEIGHTS["photo_boundary"] == 5.
    target = tmp_path / "config.json"
    value = {"seed": full.SEEDS[0], "loss_weights": full.LOSS_WEIGHTS,
             "b_decision_sha256": "abc", "source_binding": "def"}
    full._check_or_write(target, value)
    full._check_or_write(target, dict(value))
    with pytest.raises(RuntimeError, match="Frozen configuration differs"):
        full._check_or_write(target, {**value, "b_decision_sha256": "changed"})
    with pytest.raises(RuntimeError, match="Frozen configuration differs"):
        full._check_or_write(target, {**value, "allow_failed_b": True})
    assert full._digest_json(value) != full._digest_json({**value, "seed": full.SEEDS[1]})


def test_coverage_geometry_uses_cached_inverse_and_photo_size_only():
    inverse = np.stack((np.array([[1., 0., -16.], [0., 1., 0.], [0., 0., 1.]]),
                        np.eye(3))).astype(np.float32)
    fields = {"inverse": inverse}
    metadata = [{"photo_size": [32, 64], "quad_photo": [[99, 99]]},
                {"photo_size": [64, 64], "quad_photo": [[0, 0]]}]
    matrix, sizes = full._coverage_photo_geometry(fields, metadata, np.array([1, 0]))
    np.testing.assert_array_equal(matrix, inverse[[1, 0]])
    np.testing.assert_array_equal(sizes, [[64, 64], [32, 64]])
    metadata[0]["photo_size"] = [1, 64]
    with pytest.raises(RuntimeError, match="photo sizes"):
        full._coverage_photo_geometry(fields, metadata, np.array([0]))


def test_originals_rehashed_after_cache_and_before_training(tmp_path, monkeypatch):
    source_a = tmp_path / "synthetic-a.bin"
    source_b = tmp_path / "synthetic-b.bin"
    source_a.write_bytes(b"a")
    source_b.write_bytes(b"b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": [
        {"source_path": str(source_a), "source_sha256": sha256(source_a)},
        {"source_path": str(source_b), "source_sha256": sha256(source_b)}]}))
    args = SimpleNamespace(run=tmp_path / "run", manifest=manifest, b_decision=tmp_path / "run")
    args.run.mkdir()
    initial = full._audit_originals(args, expected_count=2)
    assert initial["source_count"] == 2
    audit_path = args.run / "C/cache/original-audit.json"
    full._check_or_write(audit_path, initial)
    assert full._verify_original_audit(args, expected_count=2) == initial
    source_b.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="original ordinal 1 changed"):
        full._verify_original_audit(args, expected_count=2)

    # The outer prepare orchestration hashes both before and after rendering.
    monkeypatch.setattr(full, "_partition", lambda unused: {"train": [], "dev": []})
    monkeypatch.setattr(full, "select_b_decision", lambda *unused, **kwargs: {})
    monkeypatch.setattr(full, "_prepare_split", lambda unused, split, indices: {"split": split})
    # This fixture checks the source rehash path, independently of the host's
    # free-space budget. Production prepare still enforces its 15 GiB guard.
    monkeypatch.setattr(full.shutil, "disk_usage", lambda unused: SimpleNamespace(free=16 * 2**30))
    calls = iter((initial, {**initial, "source_hashes_sha256": "different"}))
    monkeypatch.setattr(full, "_audit_originals", lambda unused: next(calls))
    with pytest.raises(RuntimeError, match="changed while generating"):
        full.prepare(args)


def test_resume_sidecar_must_bind_existing_evaluation_even_at_target(tmp_path):
    checkpoint = tmp_path / "step-1250.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = sha256(checkpoint)
    evaluation = tmp_path / "eval-1250.json"
    evaluation.write_text(json.dumps({"step": 1250, "checkpoint_sha256": digest,
                                      "prediction": {"strict_success": 0}}))
    sidecar = {"sha256": digest, "step": 1250, "evaluation_sha256": sha256(evaluation)}
    assert full._verified_checkpoint_evaluation(tmp_path, 1250, sidecar)["step"] == 1250
    evaluation.write_text(json.dumps({"step": 1250, "checkpoint_sha256": digest,
                                      "prediction": {"strict_success": 1}}))
    with pytest.raises(RuntimeError, match="sidecar SHA"):
        full._verified_checkpoint_evaluation(tmp_path, 1250, sidecar)
    evaluation.unlink()
    with pytest.raises(RuntimeError, match="missing or differs"):
        full._verified_checkpoint_evaluation(tmp_path, 1250, sidecar)


def test_cached_image_sha_checked_and_metadata_kept_local(tmp_path, monkeypatch):
    args = SimpleNamespace(run=tmp_path)
    folder = tmp_path / "C/cache/dev"
    folder.mkdir(parents=True)
    rows = [{"source_index": 0, "view_index": 0, "kind": "photo", "domain": "dev"}]
    schema = {"schedule_sha256": full._digest_json(rows)}
    monkeypatch.setattr(full, "_partition", lambda unused: {"dev": [0]})
    monkeypatch.setattr(full, "build_schedule", lambda indices, split: rows)
    monkeypatch.setattr(full, "_cache_schema", lambda *unused: schema)
    (folder / "schema.json").write_text(json.dumps(schema))
    (folder / "schedule.json").write_text(json.dumps(rows))
    digests = {}
    for name, (dtype, tail) in full.CACHE_FIELDS.items():
        path = folder / f"{name}.npy"
        array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(1, *tail))
        array[:] = 0
        array.flush()
        digests[name] = sha256(path)
    metadata = [dict(rows[0])]
    (folder / "metadata.json").write_text(json.dumps(metadata))
    (folder / "complete.json").write_text(json.dumps({"schema_sha256": full._digest_json(schema),
        "sha256": digests,
        "metadata_sha256": sha256(folder / "metadata.json"), "views": 1}))
    fields, actual, _ = full._load_cache(args, "dev")
    assert actual == metadata and fields["pixels"].shape == (1, 224, 224, 3)
    (folder / "schedule.json").write_text(json.dumps([{**rows[0], "source_index": 1}]))
    with pytest.raises(RuntimeError, match="schema or schedule changed"):
        full._load_cache(args, "dev")
    (folder / "schedule.json").write_text(json.dumps(rows))
    with (folder / "pixels.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="cached pixels changed"):
        full._load_cache(args, "dev")


def test_realized_cache_audit_recomputes_geometry_and_source_coverage():
    def row(source, kind, scale=.7):
        return {"source_index": source, "kind": kind, "photo_size": [100, 100],
                "quad_photo": [[0, 0], [100 * scale, 0],
                               [100 * scale, 100 * scale], [0, 100 * scale]]}
    rows = [row(0, "digital"), row(1, "digital"),
            row(0, "photo", .8), row(1, "photo", .8)]
    audit = full._realized_cache_audit(rows, "train", [0, 1])
    json.dumps(audit)
    assert type(audit["photo_above_half"]) is int
    assert type(audit["film_occupancy_min"]) is float
    assert type(audit["passed"]) is bool
    assert audit["passed"] and audit["sources_with_photo"] == 2
    rows[-1] = row(1, "photo", .5)
    assert not full._realized_cache_audit(rows, "train", [0, 1])["passed"]
    rows[-1] = row(1, "photo", .4)
    assert not full._realized_cache_audit(rows, "train", [0, 1])["passed"]


def test_render_small_synthetic_source_has_full_canvas_geometry():
    class SyntheticSources:
        def read(self, index):
            assert index == 0
            return np.full((128, 176, 3), 120, np.uint8)

    for kind, view_index in (("digital", 1), ("photo", 100)):
        row = {"source_index": 0, "view_index": view_index, "kind": kind,
               "mode": "front", "occupancy_bin": 1, "level": "clear",
               "pure_roll": False, "domain": "train", "condition": "clear"}
        pixels, quad, inverse, log_ratio, info = full._render(SyntheticSources(), row)
        assert pixels.shape == (224, 224, 3) and quad.shape == (4, 2)
        assert inverse.shape == (3, 3) and np.isfinite(inverse).all()
        assert log_ratio == pytest.approx(np.log(176 / 128))
        assert info["source_size"] == [176, 128] and info["kind"] == kind
        if kind == "photo":
            assert info["photo_size"] != info["source_size"]


def test_gt_ratio_is_labeled_diagnostic_never_main_prediction(monkeypatch):
    from ptrn_hip.model import torch
    from ptrn_hip import spatial_evaluation
    calls = []

    def fake_bridge(quad, log_ratios, metadata, size, *, ratio_method):
        calls.append((ratio_method, log_ratios is None))
        return {"ratio_method": ratio_method, "ratio_uses_gt": ratio_method == "oracle_gt_diagnostic"}, [{}]

    monkeypatch.setattr(spatial_evaluation, "evaluate_quad_predictions", fake_bridge)

    class TinyModel:
        def eval(self): pass
        def extract_features(self, pixels): return torch.ones((len(pixels), 1920))
        def forward_features(self, features):
            return torch.tensor([[[10., 10.], [210., 10.], [210., 210.], [10., 210.]]]).repeat(len(features), 1, 1)

    class TinyRatio:
        def eval(self): pass
        def __call__(self, corners, features): return torch.zeros(len(corners))

    fields = {"pixels": np.zeros((1, 224, 224, 3), np.uint8)}
    result, quad, ratio = full._evaluate(TinyModel(), TinyRatio(), fields,
                                         [{"kind": "photo"}], 1, "cpu", ratio_active=True)
    assert calls == [("prediction", False), ("oracle_gt_diagnostic", True)]
    assert result["prediction"]["ratio_uses_gt"] is False
    assert result["diagnostic_only_GT_ratio"]["ratio_uses_gt"] is True
    assert result["ratio_uses_GT_in_prediction"] is False
    assert quad.shape == (1, 4, 2) and ratio.shape == (1,)
