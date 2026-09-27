"""Read-only checks for private RectiRay data and checkpoint bindings.

The data mode needs no model or GPU. The other modes load checkpoints on the CPU;
they never change a run, source image, weight, or frozen training module. A
successful check is a prerequisite, not a claim of model effectiveness.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
from types import SimpleNamespace

from data_assets import safe_asset_path, validate_bundle


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TRAIN_COUNT = 4114
DEV_COUNT = 478
QUARANTINE_COUNT = 28
BOUND_ASSETS = frozenset({
    "data/manifest.json", "data/split.json", "data/backgrounds-manifest.json",
    "weights/b11.weights.h5", "weights/g0.pt",
})
# These two modules were added after the 48-file pipeline-v1 protocol. They
# are absent from that protocol and the restored legacy data renderer does not
# import them. All original modules remain mandatory and individually hashed.
LEGACY_UNBOUND_MODULES = frozenset({"capture_review.py", "capture_synthesis.py"})


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object for the requested protocol or run sidecar")
    return value


def require_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def checked_checkpoint(path: Path, expected_sha256: str):
    require_sha(expected_sha256, "checkpoint SHA-256")
    if digest(path) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    import torch

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint payload is not a mapping")
    return value


def checked_asset(root: Path, relative: str, expected: str) -> None:
    require_sha(expected, "bound asset digest")
    file = safe_asset_path(root, relative)
    if digest(file) != expected:
        raise ValueError("Bound asset SHA-256 mismatch")


def check_initializers(bundle: Path) -> None:
    """Validate the two approved initializers, including G0's required sidecar."""
    from ptrn_hip.ratio_model import B11_SHA
    from ptrn_hip.unified_model import G0_SHA, _checked_payload

    b11 = safe_asset_path(bundle, "weights/b11.weights.h5")
    g0 = safe_asset_path(bundle, "weights/g0.pt")
    safe_asset_path(bundle, "weights/g0.json")
    if digest(b11) != B11_SHA:
        raise ValueError("B1.1 initializer SHA-256 mismatch")
    payload = _checked_payload(g0, G0_SHA)
    config = payload["config"]
    if (payload.get("step") != 1250 or config.get("group") != "G0"
            or config.get("input_size") != 224 or config.get("ratio_head") is not False
            or config.get("initial_weights_sha256") != B11_SHA):
        raise ValueError("G0 initializer identity or configuration mismatch")


def build_cpu_model(bundle: Path):
    import torch
    from ptrn_hip.partial_model import build_partial

    model = build_partial(bundle / "weights/b11.weights.h5", bundle / "weights/g0.pt",
                          size=224, stage="head").to("cpu")
    optimizer = torch.optim.Adam(model.optimizer_groups("head"))
    return model, optimizer


def checked_model(checkpoint: dict, model, optimizer=None) -> None:
    from ptrn_hip.partial_geometry import CONTRACT
    from ptrn_pipeline.branch_checkpoint import _check_model_optimizer

    if checkpoint.get("contract") != CONTRACT or checkpoint.get("model_contract") != model.contract():
        raise ValueError("Checkpoint geometry or model contract mismatch")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Checkpoint model state is missing")
    check_finite_state(checkpoint["model"], "model")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        if not isinstance(checkpoint.get("optimizer"), dict):
            raise ValueError("Checkpoint optimizer state is missing")
        check_finite_state(checkpoint["optimizer"], "Adam")
        check_adam_state(checkpoint["optimizer"], optimizer)
        optimizer.load_state_dict(checkpoint["optimizer"])
        _check_model_optimizer(model, optimizer, checkpoint["model"], checkpoint["optimizer"])


def check_finite_state(value, name: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint {name} state has non-finite parameters")
    elif isinstance(value, dict):
        for item in value.values():
            check_finite_state(item, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite_state(item, name)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Checkpoint {name} state has non-finite parameters")


def check_adam_state(saved: dict, optimizer) -> None:
    """Check moments against their actual parameters without taking an update.

    Adam.load_state_dict accepts tensors of the wrong shape and can also accept
    missing moments; those failures otherwise surface only on the next update.
    All optimized heads in this frozen experiment have populated Adam state.
    """
    import torch

    groups, states = saved.get("param_groups"), saved.get("state")
    if (not isinstance(groups, list) or len(groups) != len(optimizer.param_groups)
            or not isinstance(states, dict) or not states):
        raise ValueError("Checkpoint Adam groups or trained state are missing")
    seen = set()
    for group, live in zip(groups, optimizer.param_groups):
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("Checkpoint Adam parameter group is malformed")
        identifiers = group["params"]
        if len(identifiers) != len(live["params"]):
            raise ValueError("Checkpoint Adam parameter count differs from model")
        for identifier, parameter in zip(identifiers, live["params"]):
            if type(identifier) is not int or identifier in seen:
                raise ValueError("Checkpoint Adam parameter identifiers are invalid or repeated")
            seen.add(identifier)
            state = states.get(identifier)
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group.get("amsgrad", False):
                required.add("max_exp_avg_sq")
            if not isinstance(state, dict) or set(state) != required:
                raise ValueError("Checkpoint Adam moments are missing or unexpected")
            for key in required - {"step"}:
                moment = state[key]
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or moment.layout != torch.strided):
                    raise ValueError("Checkpoint Adam moment shape or dtype differs from model parameter")
            step = state["step"]
            if (not isinstance(step, torch.Tensor) or step.ndim != 0
                    or step.is_complex() or step.dtype == torch.bool
                    or not math.isfinite(step.item()) or step.item() < 1
                    or int(step.item()) != step.item()):
                raise ValueError("Checkpoint Adam update counter is invalid")
    if set(states) != seen:
        raise ValueError("Checkpoint Adam state contains unbound parameters")


def checked_datasets(bundle: Path, backgrounds: Path, seed: int):
    from ptrn_hip.partial_data import PartialDataset
    from ptrn_pipeline.step2000_runner import RECIPE_VERSION

    train = PartialDataset(bundle, backgrounds, split="train", seed=seed,
                           recipe_version=RECIPE_VERSION, verify=False)
    dev = PartialDataset(bundle, backgrounds, split="dev", seed=seed, verify=False)
    if len(train.ids) != TRAIN_COUNT or len(dev.ids) != DEV_COUNT or len(dev) != DEV_COUNT * 12:
        raise ValueError("This bundle is not the historical 4114/478 development contract; new data cannot use step2000_runner")
    return train, dev


def checked_source_root(root: Path) -> Path:
    """A source hash is meaningful only for the code Python will actually run."""
    import ptrn_hip
    import ptrn_pipeline

    root = Path(root).resolve()
    for package in (ptrn_hip, ptrn_pipeline):
        imported = Path(package.__file__).resolve()
        if not imported.is_relative_to(root / "src"):
            raise ValueError("Imported code is outside --source-root; activate this checkout or set PYTHONPATH")
    return root


def expected_sequence(train, steps: int):
    """Reproduce step2000_data's NPY bytes without writing a run directory."""
    import numpy as np
    from ptrn_pipeline.step2000_data import _expected_metadata

    metadata = _expected_metadata(train, steps)
    sequence = np.empty(metadata["total_samples"], dtype="<i4")
    period = len(train)
    for round_index, first in enumerate(range(0, len(sequence), period)):
        order = np.random.default_rng(np.random.SeedSequence(
            [train.seed, round_index, 188])).permutation(period)
        end = min(first + period, len(sequence))
        sequence[first:end] = order[:end - first]
    output = io.BytesIO()
    np.save(output, sequence, allow_pickle=False)
    return metadata, sequence, hashlib.sha256(output.getvalue()).hexdigest()


def checked_sequence(run: Path, train, steps: int) -> str:
    import numpy as np

    specification, expected, _ = expected_sequence(train, steps)
    metadata = checked_json(run / "sample-sequence.json")
    if any(metadata.get(key) != value for key, value in specification.items()):
        raise ValueError("Sample sequence specification differs from this run")
    path = run / "sample-sequence.npy"
    sequence_sha = digest(path)
    if metadata.get("array_sha256") != sequence_sha:
        raise ValueError("Sample sequence byte SHA-256 differs from run metadata")
    actual = np.load(path, mmap_mode="r", allow_pickle=False)
    if actual.dtype != np.dtype("<i4") or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError("Sample sequence contents or shape differ")
    return sequence_sha


def prospective_binding(args, train, dev, old_binding: dict, sequence_sha: str):
    from ptrn_pipeline.step2000_runner import _bindings

    source_root = checked_source_root(args.source_root)
    options = SimpleNamespace(source_root=source_root, bundle=args.bundle)
    return _bindings(options, train, dev, sequence_sha, old_binding)


def check_parent_binding(parent: dict, protocol: dict, dev, bundle: Path) -> None:
    from ptrn_hip.partial_geometry import CONTRACT

    if parent.get("contract") != CONTRACT or parent.get("binding") != protocol:
        raise ValueError("Parent checkpoint and frozen protocol differ")
    if (parent.get("step") != 2000 or parent.get("cursor") != 2000 * 64
            or protocol.get("effective_batch") != 64 or protocol.get("stage") != "head"):
        raise ValueError("Parent step, cursor, or training stage differs")
    if protocol.get("development") != dev.fingerprint:
        raise ValueError("Frozen development fingerprint differs from the parent; new bundle cannot use step2000_runner")
    weights = protocol.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Parent initializer binding is missing")
    assets = {
        "data/manifest.json": protocol.get("manifest"),
        "data/split.json": protocol.get("split"),
        "data/backgrounds-manifest.json": protocol.get("background_manifest"),
        "weights/b11.weights.h5": weights.get("b11.weights.h5"),
        "weights/g0.pt": weights.get("g0.pt"),
    }
    for relative, expected in assets.items():
        checked_asset(bundle, relative, expected)


def mode_start(args) -> dict:
    from ptrn_pipeline.step2000_runner import PARENT_SHA256

    if args.checkpoint_sha256 != PARENT_SHA256:
        raise ValueError("Only the audited step-2000 parent can start this branch")
    parent = checked_checkpoint(args.checkpoint, args.checkpoint_sha256)
    check_initializers(args.bundle)
    train, dev = checked_datasets(args.bundle, args.backgrounds, args.seed)
    protocol = checked_json(args.parent_protocol)
    check_parent_binding(parent, protocol, dev, args.bundle)
    _, _, sequence_sha = expected_sequence(train, args.steps)
    _, source_hashes = prospective_binding(args, train, dev, protocol, sequence_sha)
    model, optimizer = build_cpu_model(args.bundle)
    checked_model(parent, model, optimizer)
    if not parent["optimizer"].get("state"):
        raise ValueError("Step-2000 parent lacks trained Adam state")
    return {"mode": "start", "parent_sha256": args.checkpoint_sha256,
            "development_views": len(dev), "sample_sequence_sha256": sequence_sha,
            "source_hashes": source_hashes,
            "status": "eligible for a new, separately bound branch; not a replay of earlier branch weights"}


def check_code_binding(code: dict, source_root: Path) -> None:
    from ptrn_pipeline.branch_upgrade import current_code_mapping

    root = checked_source_root(source_root)
    if not isinstance(code, dict) or not code:
        raise ValueError("Checkpoint has no source-code binding")
    if any("/" in name for name in code):
        if current_code_mapping(root) != code:
            raise ValueError("Checkpoint source binding differs; use its isolated training-source checkout")
    else:
        expected_names = {path.name for path in (root / "src/ptrn_hip").glob("*.py")} - LEGACY_UNBOUND_MODULES
        if set(code) != expected_names:
            raise ValueError("Legacy checkpoint source binding is incomplete or differs from this checkout")
        for name, expected in code.items():
            if not isinstance(name, str) or name != Path(name).name or not name.endswith(".py"):
                raise ValueError("Unsafe legacy source-code binding")
            checked_asset(root / "src/ptrn_hip", name, expected)


def check_checkpoint_assets(checkpoint: dict, bundle: Path) -> None:
    binding = checkpoint.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Checkpoint has no data binding")
    if "assets" in binding:
        assets = binding["assets"]
    else:
        weights = binding.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("Checkpoint has no initializer binding")
        assets = {
            "data/manifest.json": binding.get("manifest"),
            "data/split.json": binding.get("split"),
            "data/backgrounds-manifest.json": binding.get("background_manifest"),
            "weights/b11.weights.h5": weights.get("b11.weights.h5"),
            "weights/g0.pt": weights.get("g0.pt"),
        }
    if not isinstance(assets, dict) or set(assets) != BOUND_ASSETS:
        raise ValueError("Checkpoint must bind all five historical assets")
    for relative, expected in assets.items():
        checked_asset(bundle, relative, expected)


def mode_resume(args) -> dict:
    from ptrn_pipeline.branch_checkpoint import restore_branch_checkpoint
    from ptrn_pipeline.step2000_runner import PARENT_SHA256
    import torch

    run = args.run.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if not run.is_dir() or checkpoint_path.parent != run or not checkpoint_path.name.startswith("branch-step-"):
        raise ValueError("Resume checkpoint must be inside its original run directory")
    checkpoint = checked_checkpoint(args.checkpoint, args.checkpoint_sha256)
    check_initializers(args.bundle)
    train, dev = checked_datasets(args.bundle, args.backgrounds, args.seed)
    binding = checked_json(run / "branch-binding.json")
    provenance = checked_json(run / "branch-provenance.json")
    old_binding = provenance.get("parent_binding")
    if not isinstance(old_binding, dict) or provenance.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("Branch parent provenance differs from the audited parent")
    if args.parent_protocol is not None and checked_json(args.parent_protocol) != old_binding:
        raise ValueError("Branch parent provenance differs from frozen parent protocol")
    # The parent file is not needed to resume, but its exact frozen development
    # and five assets must still match the current private bundle.
    if old_binding.get("development") != dev.fingerprint:
        raise ValueError("Branch development fingerprint differs from frozen parent")
    weights = old_binding.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Parent initializer binding is missing")
    for relative, expected in {
        "data/manifest.json": old_binding.get("manifest"),
        "data/split.json": old_binding.get("split"),
        "data/backgrounds-manifest.json": old_binding.get("background_manifest"),
        "weights/b11.weights.h5": weights.get("b11.weights.h5"),
        "weights/g0.pt": weights.get("g0.pt"),
    }.items():
        checked_asset(args.bundle, relative, expected)
    if provenance.get("branch_seed") != args.branch_seed:
        raise ValueError("Branch seed differs from original run")
    sequence_sha = checked_sequence(run, train, args.steps)
    expected_binding, source_hashes = prospective_binding(args, train, dev, old_binding, sequence_sha)
    if binding != expected_binding or checkpoint.get("binding") != binding:
        raise ValueError("Run checkpoint, code, data, recipe, or sample-sequence binding differs")
    if provenance.get("source_hashes") != source_hashes or checkpoint.get("branch_provenance") != provenance:
        raise ValueError("Branch source provenance differs from run sidecar")
    cuda_rng = checkpoint.get("cuda_rng")
    if not isinstance(cuda_rng, list) or not all(
            isinstance(state, torch.Tensor) and state.dtype == torch.uint8
            and state.ndim == 1 and state.numel() > 0 and state.is_contiguous()
            for state in cuda_rng):
        raise ValueError("Checkpoint CUDA RNG state is malformed")
    branch_step = checkpoint.get("branch_step")
    if type(branch_step) is not int or branch_step < 0:
        raise ValueError("Checkpoint branch step is malformed")
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Checkpoint Adam state is missing")
    if branch_step > 0 and not cuda_rng:
        raise ValueError("Nonzero branch checkpoint lacks CUDA RNG state")
    if branch_step > 0 and not optimizer_state.get("state"):
        raise ValueError("Nonzero branch checkpoint lacks Adam state")
    check_finite_state(checkpoint.get("model"), "model")
    check_finite_state(checkpoint.get("optimizer"), "Adam")
    model, optimizer = build_cpu_model(args.bundle)
    checked_model(checkpoint, model, optimizer)
    del checkpoint
    before = torch.get_rng_state().clone()
    try:
        step, cursor, _ = restore_branch_checkpoint(args.checkpoint, model, optimizer,
            expected_binding=binding, expected_provenance=provenance,
            expected_source_hashes=source_hashes)
    finally:
        torch.set_rng_state(before)
    if step >= args.steps:
        raise ValueError("Branch already reached the requested target step")
    return {"mode": "resume", "checkpoint_sha256": args.checkpoint_sha256,
            "branch_step": step, "consumed_sample_cursor": cursor,
            "sample_sequence_sha256": sequence_sha,
            "status": "source-, asset-, optimizer-, and cursor-bound resume candidate"}


def mode_evaluate(args) -> dict:
    from ptrn_hip.partial_data import PartialDataset
    from ptrn_hip.partial_geometry import CONTRACT

    checkpoint = checked_checkpoint(args.checkpoint, args.checkpoint_sha256)
    check_initializers(args.bundle)
    dev = PartialDataset(args.bundle, args.backgrounds, split="dev", seed=args.seed, verify=False)
    binding = checkpoint.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Checkpoint has no data binding")
    if binding.get("development") != dev.fingerprint or len(dev) != DEV_COUNT * 12:
        raise ValueError("Checkpoint and full development fingerprint differ")
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("Checkpoint geometry contract differs")
    if "branch_provenance" in checkpoint:
        from ptrn_pipeline.branch_checkpoint import (
            _check_branch_binding, _check_provenance, _check_source_hashes,
        )

        provenance = checkpoint["branch_provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("Branch provenance must be a mapping")
        hashes = _check_source_hashes(provenance.get("source_hashes"))
        _check_branch_binding(binding, hashes)
        _check_provenance(provenance, binding, hashes)
        step = checkpoint.get("branch_step")
        if (type(step) is not int or step < 0 or checkpoint.get("step") != step
                or checkpoint.get("branch_cursor") != step * 64
                or checkpoint.get("cursor") != step * 64):
            raise ValueError("Branch checkpoint step/cursor mismatch")
    check_code_binding(binding.get("code"), args.source_root)
    check_checkpoint_assets(checkpoint, args.bundle)
    model, _ = build_cpu_model(args.bundle)
    checked_model(checkpoint, model)
    return {"mode": "evaluate", "checkpoint_sha256": args.checkpoint_sha256,
            "development_views": len(dev),
            "status": "strict source- and asset-bound full-development evaluation candidate"}


def report_path(path: Path, source_root: Path) -> Path:
    # The helper's repository owns its reports even if --source-root selects a
    # separate historical checkout. Do not resolve links before checking them.
    path = path.expanduser().absolute()
    root = ROOT.absolute()
    if ".." in path.parts or not path.is_relative_to(root / ".local"):
        raise ValueError("Private check reports must be under this repository's ignored .local/")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Private check report path contains a symlink")
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing asset-check report")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(path)],
                             cwd=root, capture_output=True, check=False)
    if ignored.returncode != 0:
        raise ValueError("Private check reports must be ignored by Git")
    return path


SAFE_DATA_FIELDS = frozenset({
    "manifest_sha256", "split_sha256", "background_manifest_sha256",
    "split_counts", "background_counts", "background_real_photo_counts",
    "duplicates", "all_film_records_checked", "all_background_rows_checked",
})


def write_report(path: Path, result: dict) -> None:
    path = report_path(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = report_path(path, ROOT)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("data", "start", "resume", "evaluate"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--branch-seed", type=int, default=20260926)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.mode != "data" and (args.checkpoint is None or args.checkpoint_sha256 is None):
        parser.error("start/resume/evaluate require --checkpoint and --checkpoint-sha256")
    if args.mode == "start" and args.parent_protocol is None:
        parser.error("start requires --parent-protocol")
    if args.mode == "resume" and args.run is None:
        parser.error("resume requires --run")
    if args.mode == "data" and any(value is not None for value in
                                  (args.checkpoint, args.checkpoint_sha256, args.parent_protocol, args.run)):
        parser.error("data mode accepts only bundle and background inputs")
    if args.steps < 1 or args.seed < 0 or not 0 <= args.branch_seed < 2**63:
        parser.error("Steps must be positive, seed nonnegative, and branch seed in [0, 2**63)")
    if args.report is not None:
        args.report = report_path(args.report, args.source_root)
    return args


def main(argv=None):
    args = parse_args(argv)
    # Set visibility before importing any model/torch/Keras module.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["KERAS_BACKEND"] = "torch"
    # Whitelist aggregate fields so a future validator cannot accidentally put
    # filenames or source-identifying paths into stdout or the optional report.
    summary = {key: value for key, value in validate_bundle(args.bundle, args.backgrounds).items()
               if key in SAFE_DATA_FIELDS}
    if args.mode == "data":
        result = {"mode": "data", **summary, "status": "all declared assets checked"}
    else:
        result = {"mode": args.mode, **summary,
                  **{"start": mode_start, "resume": mode_resume, "evaluate": mode_evaluate}[args.mode](args)}
    if args.report is not None:
        write_report(args.report, result)
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError, KeyError, TypeError) as error:
        reason = "required local file is missing or unreadable" if isinstance(error, OSError) else str(error)
        print(f"asset check failed: {reason}", file=sys.stderr)
        raise SystemExit(1) from error
