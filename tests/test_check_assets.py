"""Synthetic-only tests for the read-only asset and checkpoint preflight."""

from __future__ import annotations

import hashlib
import copy
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_assets as check  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def bundle(tmp_path):
    root, backgrounds = tmp_path / "bundle", tmp_path / "backgrounds"
    splits = {"train": [0, 1, 2], "dev": [3, 4], "quarantine": [5]}
    records = []
    for index in range(6):
        split = next(name for name, ids in splits.items() if index in ids)
        relative = f"data/films/film-{index:04d}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((24, 32, 3), 20 + index * 20, np.uint8)).save(path)
        records.append({"id": f"film-{index:04d}", "source_path": relative,
                        "source_sha256": sha(path), "split": split,
                        "original_size": [32, 24]})
    rows = []
    for index, split in enumerate(("train", "dev")):
        relative = f"photo-{index:04d}.png"
        path = backgrounds / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((28, 36, 3), 160 + index * 30, np.uint8)).save(path)
        rows.append({"id": f"background-{index:04d}", "path": relative,
                     "sha256": sha(path), "split": split, "source_kind": "real_photo"})
    write_json(root / "data/manifest.json", {"records": records})
    write_json(root / "data/split.json", splits)
    write_json(root / "data/backgrounds-manifest.json", {"rows": rows})
    weights = root / "weights"
    weights.mkdir()
    (weights / "b11.weights.h5").write_bytes(b"synthetic-b11")
    (weights / "g0.pt").write_bytes(b"synthetic-g0")
    return root, backgrounds


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 1)

    def contract(self):
        return {"contract": "synthetic-visible-plane-v3"}

    def forward(self, pixels):
        return self.head(pixels)


def tiny_model(_bundle):
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer


def historical_fixture(bundle, monkeypatch, *, steps=4):
    """Build a small but internally bound parent/run without private weights."""
    from ptrn_hip.partial_data import PartialDataset
    from ptrn_hip.partial_geometry import CONTRACT
    from ptrn_pipeline import step2000_runner

    root, backgrounds = bundle
    monkeypatch.setattr(check, "TRAIN_COUNT", 3)
    monkeypatch.setattr(check, "DEV_COUNT", 2)
    monkeypatch.setattr(check, "QUARANTINE_COUNT", 1)
    monkeypatch.setattr(check, "check_initializers", lambda _bundle: None)
    monkeypatch.setattr(check, "build_cpu_model", tiny_model)
    old_train = PartialDataset(root, backgrounds, split="train", verify=False)
    dev = PartialDataset(root, backgrounds, split="dev", verify=False)
    protocol = {**old_train.fingerprint, "development": dev.fingerprint,
                "effective_batch": 64, "stage": "head",
                "loss": {"coverage_weight": 1., "mask_weight": 1., "ratio_weight": .25},
                "code": {path.name: sha(path) for path in (ROOT / "src/ptrn_hip").glob("*.py")
                         if path.name not in check.LEGACY_UNBOUND_MODULES},
                "weights": {name: sha(root / "weights" / name)
                            for name in ("b11.weights.h5", "g0.pt")}}
    parent_protocol = root.parent / "frozen-protocol.json"
    write_json(parent_protocol, protocol)
    model, optimizer = tiny_model(root)
    # A real parent has populated Adam state. One synthetic update suffices to
    # exercise strict state loading without pretending to train 2,000 steps.
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    parent = root.parent / "step-2000.pt"
    torch.save({"contract": CONTRACT, "model_contract": model.contract(),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "binding": protocol, "step": 2000, "cursor": 128000}, parent)
    parent_sha = sha(parent)
    monkeypatch.setattr(step2000_runner, "PARENT_SHA256", parent_sha)
    return SimpleNamespace(bundle=root, backgrounds=backgrounds, parent=parent,
                           parent_protocol=parent_protocol, parent_sha=parent_sha,
                           protocol=protocol, steps=steps)


def arguments(fixture, checkpoint: Path, *, run=None, steps=None):
    return SimpleNamespace(bundle=fixture.bundle, backgrounds=fixture.backgrounds,
                           checkpoint=checkpoint,
                           checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None,
                           parent_protocol=fixture.parent_protocol, run=run,
                           source_root=ROOT, steps=steps or fixture.steps,
                           seed=20260925, branch_seed=20260926)


def test_data_mode_checks_all_declared_assets_including_quarantine(bundle):
    root, backgrounds = bundle
    result = check.main(["--mode", "data", "--bundle", str(root),
                         "--backgrounds", str(backgrounds)])
    assert result["all_film_records_checked"] == 6
    assert result["all_background_rows_checked"] == 2
    assert result["split_counts"] == {"train": 3, "dev": 2, "quarantine": 1}
    assert "source_path" not in json.dumps(result)
    (root / "data/films/film-0005.png").write_bytes(b"damaged quarantine copy")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check.main(["--mode", "data", "--bundle", str(root),
                    "--backgrounds", str(backgrounds)])


def test_initializer_check_requires_matching_g0_sidecar(bundle, monkeypatch):
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    from ptrn_hip import ratio_model, unified_model

    root, _ = bundle
    b11 = root / "weights/b11.weights.h5"
    g0 = root / "weights/g0.pt"
    configuration = {"group": "G0", "input_size": 224, "ratio_head": False,
                     "initial_weights_sha256": sha(b11)}
    torch.save({"state_dict": {}, "config": configuration, "step": 1250}, g0)
    write_json(g0.with_suffix(".json"),
               {"sha256": sha(g0), "config": configuration, "step": 1250})
    monkeypatch.setattr(ratio_model, "B11_SHA", sha(b11))
    monkeypatch.setattr(unified_model, "G0_SHA", sha(g0))
    check.check_initializers(root)
    valid_sidecar = g0.with_suffix(".json").read_bytes()
    changed = json.loads(g0.with_suffix(".json").read_text())
    changed["config"]["input_size"] = 384
    write_json(g0.with_suffix(".json"), changed)
    with pytest.raises(ValueError, match="configuration differs"):
        check.check_initializers(root)
    sidecar = g0.with_suffix(".json")
    sidecar.unlink()
    with pytest.raises(ValueError, match="missing"):
        check.check_initializers(root)
    outside = root.parent / "external-sidecar.json"
    outside.write_bytes(valid_sidecar)
    sidecar.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        check.check_initializers(root)


def test_start_checks_parent_protocol_and_records_a_new_source_binding(bundle, monkeypatch):
    history = historical_fixture(bundle, monkeypatch)
    args = arguments(history, history.parent)
    originals = {path: path.read_bytes() for path in
                 (args.checkpoint, args.parent_protocol, *args.bundle.glob("data/*.json"))}
    result = check.main(["--mode", "start", "--bundle", str(args.bundle),
                         "--backgrounds", str(args.backgrounds),
                         "--checkpoint", str(args.checkpoint),
                         "--checkpoint-sha256", args.checkpoint_sha256,
                         "--parent-protocol", str(args.parent_protocol),
                         "--source-root", str(args.source_root), "--steps", str(args.steps)])
    assert result["parent_sha256"] == history.parent_sha
    assert result["development_views"] == 24
    assert result["source_hashes"]["code"] != history.protocol["code"]["partial_data.py"]
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        check.checked_checkpoint(args.checkpoint, "0" * 64)
    altered = json.loads(history.parent_protocol.read_text())
    altered["development"]["seed"] += 1
    write_json(history.parent_protocol, altered)
    with pytest.raises(ValueError, match="frozen protocol"):
        check.mode_start(args)


def test_evaluate_requires_exact_training_source_and_bound_assets(bundle, monkeypatch):
    history = historical_fixture(bundle, monkeypatch)
    args = arguments(history, history.parent)
    result = check.mode_evaluate(args)
    assert result["development_views"] == 24
    assert len(history.protocol["code"]) == 48
    assert set(path.name for path in (ROOT / "src/ptrn_hip").glob("*.py")) - set(history.protocol["code"]) == {
        "capture_review.py", "capture_synthesis.py"}
    # Later preview modules may exist beside the archived renderer, but none
    # of the original files may be omitted from the checkpoint's binding.
    missing_original = dict(history.protocol["code"])
    del missing_original["partial_synthesis.py"]
    with pytest.raises(ValueError, match="incomplete"):
        check.check_code_binding(missing_original, ROOT)
    from ptrn_pipeline.branch_upgrade import current_code_mapping

    code = current_code_mapping(ROOT)
    check.check_code_binding(code, ROOT)
    code["src/ptrn_hip/partial_geometry.py"] = "0" * 64
    with pytest.raises(ValueError, match="isolated training-source checkout"):
        check.check_code_binding(code, ROOT)
    with pytest.raises(ValueError, match="incomplete"):
        check.check_code_binding({"partial_data.py": sha(ROOT / "src/ptrn_hip/partial_data.py")}, ROOT)
    with pytest.raises(ValueError, match="all five historical assets"):
        check.check_checkpoint_assets({"binding": {"assets": {
            "weights/g0.pt": sha(history.bundle / "weights/g0.pt")}}}, history.bundle)
    history.protocol["weights"]["g0.pt"] = "0" * 64
    with pytest.raises(ValueError, match="Bound asset SHA-256 mismatch"):
        check.check_checkpoint_assets({"binding": history.protocol}, history.bundle)


def test_resume_checks_saved_sequence_and_training_state(bundle, monkeypatch):
    history = historical_fixture(bundle, monkeypatch)
    from ptrn_hip.partial_data import PartialDataset
    from ptrn_hip.partial_geometry import CONTRACT
    from ptrn_pipeline.branch_checkpoint import BRANCH_FORMAT
    from ptrn_pipeline.step2000_data import prepare_sequence

    train = PartialDataset(history.bundle, history.backgrounds, split="train",
                           recipe_version="ptrn-step2000-unified-v1", verify=False)
    dev = PartialDataset(history.bundle, history.backgrounds, split="dev", verify=False)
    run = history.bundle.parent / "run"
    _, sequence_sha = prepare_sequence(train, history.steps, run)
    args = arguments(history, run / "branch-step-1.pt", run=run)
    binding, source_hashes = check.prospective_binding(args, train, dev,
                                                       history.protocol, sequence_sha)
    provenance = {"format": BRANCH_FORMAT, "parent_sha256": history.parent_sha,
                  "parent_step": 2000, "parent_cursor": 128000,
                  "parent_binding": history.protocol, "branch_seed": 20260926,
                  "source_hashes": source_hashes}
    write_json(run / "branch-binding.json", binding)
    write_json(run / "branch-provenance.json", provenance)
    model, optimizer = tiny_model(history.bundle)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    torch.save({"contract": CONTRACT, "model_contract": model.contract(),
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "binding": binding, "branch_provenance": provenance,
                "step": 1, "cursor": 64, "branch_step": 1, "branch_cursor": 64,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": [torch.zeros(16, dtype=torch.uint8)]}, args.checkpoint)
    args.checkpoint_sha256 = sha(args.checkpoint)
    result = check.mode_resume(args)
    assert result["branch_step"] == 1 and result["consumed_sample_cursor"] == 64
    assert check.mode_evaluate(args)["development_views"] == 24
    payload = torch.load(args.checkpoint, weights_only=True)
    damaged = copy.deepcopy(payload)
    next(iter(damaged["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(3, 3)
    torch.save(damaged, args.checkpoint)
    args.checkpoint_sha256 = sha(args.checkpoint)
    with pytest.raises(ValueError, match="moment shape"):
        check.mode_resume(args)
    damaged = copy.deepcopy(payload)
    damaged["cuda_rng"] = [torch.zeros(16, dtype=torch.float32)]
    torch.save(damaged, args.checkpoint)
    args.checkpoint_sha256 = sha(args.checkpoint)
    with pytest.raises(ValueError, match="CUDA RNG state is malformed"):
        check.mode_resume(args)
    torch.save(payload, args.checkpoint)
    args.checkpoint_sha256 = sha(args.checkpoint)
    sequence_path = run / "sample-sequence.npy"
    with sequence_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Sample sequence byte SHA-256"):
        check.mode_resume(args)


@pytest.mark.parametrize("mutation, message", [
    ("shape", "moment shape"), ("missing_moment", "moments are missing"),
    ("missing_parameter", "moments are missing"), ("step", "update counter"),
    ("duplicate_parameter", "identifiers"), ("extra_parameter", "unbound parameters"),
    ("nonfinite_lr", "non-finite"),
])
def test_optimizer_validation_rejects_states_adam_load_would_accept(mutation, message):
    from ptrn_hip.partial_geometry import CONTRACT

    model, optimizer = tiny_model(None)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    payload = {"contract": CONTRACT, "model_contract": model.contract(),
               "model": model.state_dict(), "optimizer": copy.deepcopy(optimizer.state_dict())}
    state = payload["optimizer"]["state"]
    first = next(iter(state))
    if mutation == "shape":
        state[first]["exp_avg"] = torch.zeros(3, 3)
    elif mutation == "missing_moment":
        del state[first]["exp_avg_sq"]
    elif mutation == "missing_parameter":
        del state[first]
    elif mutation == "step":
        state[first]["step"] = torch.tensor(-1.)
    elif mutation == "duplicate_parameter":
        payload["optimizer"]["param_groups"][0]["params"][1] = first
    elif mutation == "extra_parameter":
        state[999] = copy.deepcopy(state[first])
    elif mutation == "nonfinite_lr":
        payload["optimizer"]["param_groups"][0]["lr"] = float("nan")
    replacement, adam = tiny_model(None)
    with pytest.raises(ValueError, match=message):
        check.checked_model(payload, replacement, adam)


def test_reports_are_local_ignored_non_overwriting_and_reject_links(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text(".local/\n")
    monkeypatch.setattr(check, "ROOT", repo)
    report = repo / ".local/checks/report.json"
    assert check.report_path(report, repo) == report
    check.write_report(report, {"status": "checked"})
    original = report.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        check.write_report(report, {"status": "changed"})
    assert report.read_bytes() == original
    for path in (tmp_path / "outside.json", repo / "tracked.json",
                 repo / ".local/../escaped.json"):
        with pytest.raises(ValueError, match="ignored .local"):
            check.report_path(path, repo)
    external = tmp_path / "external"
    external.mkdir()
    (repo / ".local/link").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        check.write_report(repo / ".local/link/result.json", {})
    (repo / ".local/dangling.json").symlink_to(external / "absent.json")
    with pytest.raises(ValueError, match="symlink"):
        check.write_report(repo / ".local/dangling.json", {})
    assert list(external.iterdir()) == []
    (repo / ".gitignore").write_text("")
    with pytest.raises(ValueError, match="ignored by Git"):
        check.report_path(repo / ".local/other.json", repo)


def test_failed_data_check_does_not_publish_report(bundle, tmp_path, monkeypatch):
    root, backgrounds = bundle
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text(".local/\n")
    monkeypatch.setattr(check, "ROOT", repo)
    report = repo / ".local/check.json"
    (root / "data/films/film-0000.png").write_bytes(b"broken")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check.main(["--mode", "data", "--bundle", str(root),
                    "--backgrounds", str(backgrounds), "--report", str(report)])
    assert not report.exists()
