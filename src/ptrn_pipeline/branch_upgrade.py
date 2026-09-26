"""Audit a code-only upgrade of an interrupted step-2000 branch.

The step-250 checkpoint is a training-state artifact. A resource/evaluation
fix changes its code binding, so the ordinary strict restore correctly refuses
it. This module issues a *new* checkpoint with the new source binding while
preserving every model, Adam, RNG and sample-position value from the old one.
Only the three independently reviewed orchestration/evaluation files may
change. The recipe, data assets and sample sequence remain byte-bound.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import torch

from ptrn_hip.partial_geometry import CONTRACT

from .branch_checkpoint import (
    _check_branch_binding,
    _check_provenance,
    _check_sha,
    _check_source_hashes,
    _equal_state,
    _json_copy,
    _write_new_checkpoint,
    mapping_sha256,
    sha256_file,
)


UPGRADE_FORMAT = "ptrn-branch-code-upgrade-v1"
PERMITTED_CODE_CHANGES = frozenset({
    "src/ptrn_pipeline/step2000_runner.py",
    "src/ptrn_pipeline/async_eval.py",
    "src/ptrn_pipeline/branch_upgrade.py",
})
_PRESERVED_CHECKPOINT_KEYS = frozenset({
    "contract", "model_contract", "model", "optimizer", "step", "cursor",
    "branch_step", "branch_cursor", "torch_rng", "cuda_rng",
})


def current_code_mapping(source_root: str | Path) -> dict[str, str]:
    """Hash exactly the source paths included by step2000_runner._bindings."""
    root = Path(source_root).resolve()
    relative_paths = [
        str(path.relative_to(root))
        for directory in (root / "src/ptrn_hip", root / "src/ptrn_pipeline")
        for path in directory.glob("*.py")
    ]
    relative_paths.extend(("pyproject.toml", "configs/pipeline-v1.json"))
    mapping = {}
    for relative in sorted(relative_paths):
        path = root / relative
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise FileNotFoundError(f"Missing or unsafe source file: {relative}")
        mapping[relative] = sha256_file(path)
    if not mapping:
        raise ValueError("No bound source files found")
    return mapping


def _validated_step(checkpoint: dict, binding: dict, expected_step: int) -> tuple[int, int]:
    if checkpoint.get("contract") != CONTRACT or not isinstance(checkpoint.get("model_contract"), dict):
        raise ValueError("Checkpoint geometry or model contract is invalid")
    if not _PRESERVED_CHECKPOINT_KEYS.issubset(checkpoint):
        raise ValueError("Checkpoint is missing training or RNG state")
    step, cursor = checkpoint["branch_step"], checkpoint["branch_cursor"]
    if (isinstance(step, bool) or not isinstance(step, int) or step != expected_step
            or checkpoint["step"] != step or isinstance(cursor, bool) or not isinstance(cursor, int)
            or cursor != step * binding["effective_batch"] or checkpoint["cursor"] != cursor):
        raise ValueError("Checkpoint step or sample cursor mismatch")
    if not isinstance(checkpoint["model"], dict) or not isinstance(checkpoint["optimizer"], dict):
        raise ValueError("Checkpoint model or optimizer state is missing")
    if not isinstance(checkpoint["torch_rng"], torch.Tensor):
        raise ValueError("Checkpoint CPU RNG state is missing")
    if not isinstance(checkpoint["cuda_rng"], list) or not all(
        isinstance(state, torch.Tensor) for state in checkpoint["cuda_rng"]
    ):
        raise ValueError("Checkpoint CUDA RNG state is invalid")
    return step, cursor


def upgrade_branch_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_sha256: str,
    expected_old_binding: dict,
    expected_old_provenance: dict,
    source_root: str | Path,
    asset_root: str | Path,
    sample_sequence_path: str | Path,
    approved_code_changes: Sequence[str],
    expected_step: int = 250,
) -> dict[str, Any]:
    """Create one immutable checkpoint after an explicitly approved code diff.

    The expected SHA and old metadata must be obtained independently from the
    previous run. Returns the new binding, provenance and checkpoint SHA.
    This is a provenance migration, not a promise of bitwise replay under the
    new runtime or a licence to alter training data/arithmetic.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Upgrade destination cannot be the source checkpoint")
    if destination.exists():
        raise FileExistsError(destination)
    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step < 1:
        raise ValueError("Expected branch step must be positive")
    _check_sha(expected_source_sha256, "expected_source_sha256")
    if sha256_file(source) != expected_source_sha256:
        raise ValueError("Source checkpoint SHA mismatch")

    old_binding = _json_copy(expected_old_binding)
    old_provenance = _json_copy(expected_old_provenance)
    old_hashes = _check_source_hashes(old_provenance.get("source_hashes"))
    old_binding = _check_branch_binding(old_binding, old_hashes)
    old_provenance = _check_provenance(old_provenance, old_binding, old_hashes)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("binding") != old_binding or checkpoint.get("branch_provenance") != old_provenance:
        raise ValueError("Source checkpoint binding or provenance differs from the audited sidecars")
    step, cursor = _validated_step(checkpoint, old_binding, expected_step)

    root = Path(asset_root).resolve()
    for relative, expected_sha in old_binding["assets"].items():
        asset = root / relative
        if not asset.is_file() or not asset.resolve().is_relative_to(root):
            raise FileNotFoundError(f"Missing or unsafe bound asset: {relative}")
        if sha256_file(asset) != expected_sha:
            raise ValueError(f"Bound asset changed: {relative}")
    if sha256_file(sample_sequence_path) != old_binding["sample_manifest_sha256"]:
        raise ValueError("Bound sample sequence changed")

    new_code = current_code_mapping(source_root)
    old_code = old_binding["code"]
    changed = sorted(path for path in old_code.keys() | new_code.keys()
                     if old_code.get(path) != new_code.get(path))
    declared = list(approved_code_changes)
    if len(declared) != len(set(declared)) or set(declared) != set(changed):
        raise ValueError(f"Code changes differ from explicit approval: actual={changed}, approved={declared}")
    if not changed or not set(changed).issubset(PERMITTED_CODE_CHANGES):
        raise ValueError(f"Changed files include training or recipe code outside the permitted set: {changed}")

    # Start from the old binding, changing only the code map. This also keeps
    # the recipe, assets, development set and sequence hashes exactly equal.
    new_binding = copy.deepcopy(old_binding)
    new_binding["code"] = new_code
    new_hashes = {**old_hashes, "code": mapping_sha256(new_code)}
    new_binding = _check_branch_binding(new_binding, new_hashes)
    migration = {
        "format": UPGRADE_FORMAT,
        "source_checkpoint_sha256": expected_source_sha256,
        "branch_step": step,
        "branch_cursor": cursor,
        "from_code_sha256": old_hashes["code"],
        "to_code_sha256": new_hashes["code"],
        "changed_code": {
            path: {"before": old_code.get(path), "after": new_code.get(path)}
            for path in changed
        },
    }
    new_provenance = copy.deepcopy(old_provenance)
    history = new_provenance.get("code_migrations", [])
    if not isinstance(history, list):
        raise ValueError("Invalid previous code migration history")
    new_provenance["code_migrations"] = [*history, migration]
    new_provenance["source_hashes"] = new_hashes
    new_provenance = _check_provenance(new_provenance, new_binding, new_hashes)

    upgraded = dict(checkpoint)
    upgraded["binding"] = new_binding
    upgraded["branch_provenance"] = new_provenance
    if current_code_mapping(source_root) != new_code:
        raise RuntimeError("Source files changed during code upgrade")
    _write_new_checkpoint(destination, upgraded)
    verified = torch.load(destination, map_location="cpu", weights_only=True)
    if (verified.get("binding") != new_binding or verified.get("branch_provenance") != new_provenance
            or not all(_equal_state(checkpoint[key], verified[key]) for key in _PRESERVED_CHECKPOINT_KEYS)):
        raise RuntimeError("Upgraded checkpoint did not preserve all training state")
    return {
        "binding": new_binding,
        "provenance": new_provenance,
        "source_hashes": new_hashes,
        "migration": migration,
        "checkpoint_sha256": sha256_file(destination),
        "branch_step": step,
        "branch_cursor": cursor,
    }


def upgrade_run_directory(
    old_run_dir: str | Path,
    new_run_dir: str | Path,
    *,
    expected_checkpoint_sha256: str,
    source_root: str | Path,
    asset_root: str | Path,
    approved_code_changes: Sequence[str],
    step: int = 250,
) -> dict[str, Any]:
    """Publish a complete, separately named resume directory."""
    old_run_dir, new_run_dir = Path(old_run_dir), Path(new_run_dir)
    if new_run_dir.exists():
        raise FileExistsError(new_run_dir)
    source = old_run_dir / f"branch-step-{step}.pt"
    old_binding = json.loads((old_run_dir / "branch-binding.json").read_text())
    old_provenance = json.loads((old_run_dir / "branch-provenance.json").read_text())
    sequence = old_run_dir / "sample-sequence.npy"
    sequence_metadata = old_run_dir / "sample-sequence.json"
    metadata = json.loads(sequence_metadata.read_text())
    if (sha256_file(sequence) != old_binding.get("sample_manifest_sha256")
            or metadata.get("array_sha256") != old_binding.get("sample_manifest_sha256")):
        raise ValueError("Saved sample sequence is incomplete or differs from the checkpoint")

    new_run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{new_run_dir.name}.upgrade-", dir=new_run_dir.parent))
    try:
        destination = staging / source.name
        result = upgrade_branch_checkpoint(source, destination,
            expected_source_sha256=expected_checkpoint_sha256,
            expected_old_binding=old_binding, expected_old_provenance=old_provenance,
            source_root=source_root, asset_root=asset_root,
            sample_sequence_path=sequence, approved_code_changes=approved_code_changes,
            expected_step=step)
        for name in ("sample-sequence.npy", "sample-sequence.json"):
            shutil.copy2(old_run_dir / name, staging / name)
        if sha256_file(staging / "sample-sequence.npy") != result["binding"]["sample_manifest_sha256"]:
            raise RuntimeError("Copied sample sequence changed")
        for name, document in (("branch-binding.json", result["binding"]),
                               ("branch-provenance.json", result["provenance"]),
                               ("code-upgrade.json", {
                                   "migration": result["migration"],
                                   "checkpoint_sha256": result["checkpoint_sha256"],
                                   "source_checkpoint_name": source.name,
                               })):
            (staging / name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
        if new_run_dir.exists():
            raise FileExistsError(new_run_dir)
        os.rename(staging, new_run_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", type=Path, required=True)
    parser.add_argument("--new-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--allow-code-change", action="append", required=True)
    parser.add_argument("--step", type=int, default=250)
    args = parser.parse_args(argv)
    result = upgrade_run_directory(args.old_run_dir, args.new_run_dir,
        expected_checkpoint_sha256=args.checkpoint_sha256,
        source_root=args.source_root, asset_root=args.asset_root,
        approved_code_changes=args.allow_code_change,
        step=args.step)
    print(json.dumps({"new_run_dir": str(args.new_run_dir),
                      "checkpoint_sha256": result["checkpoint_sha256"],
                      "step": result["branch_step"], "cursor": result["branch_cursor"],
                      "changed_code": sorted(result["migration"]["changed_code"])}, sort_keys=True))


if __name__ == "__main__":
    main()
