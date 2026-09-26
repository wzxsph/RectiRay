"""Explicit, lossless step-2000 fork into a separately bound training run.

The old checkpoint's *recorded* binding and file digest are checked, but its
code hashes are deliberately not recomputed from the changed working tree.
Branch checkpoints use a fresh data cursor and RNG. They are not a continuation
of the parent's sample stream or a claim of bitwise GPU training replay.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch

from ptrn_hip.partial_geometry import CONTRACT


BRANCH_FORMAT = "ptrn-step2000-branch-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_NAMES = frozenset({"recipe", "code", "assets", "sample_manifest"})


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _json_copy(value: Any) -> Any:
    """Require auditable JSON metadata and reject NaN or non-string keys."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    copy = json.loads(encoded)
    if copy != value:
        raise ValueError("Binding/provenance must contain JSON-compatible string-keyed data")
    return copy


def _check_sha(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _check_source_hashes(source_hashes: dict) -> dict:
    hashes = _json_copy(source_hashes)
    if not isinstance(hashes, dict) or hashes.keys() != _SOURCE_NAMES:
        raise ValueError(f"Source hashes must contain exactly {sorted(_SOURCE_NAMES)}")
    for name, value in hashes.items():
        _check_sha(value, name)
    return hashes


def _check_hash_map(value: dict, name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty file-to-SHA256 mapping")
    for filename, digest in value.items():
        if not isinstance(filename, str) or not filename or filename.startswith("/") or ".." in Path(filename).parts:
            raise ValueError(f"Unsafe {name} path")
        _check_sha(digest, f"{name}[{filename}]")


def _check_branch_binding(binding: dict, source_hashes: dict) -> dict:
    binding = _json_copy(binding)
    if not isinstance(binding, dict) or binding.get("effective_batch") != 64 or binding.get("stage") != "head":
        raise ValueError("Branch requires the unchanged head stage and effective batch 64")
    if not isinstance(binding.get("recipe_version"), str) or not binding["recipe_version"]:
        raise ValueError("Branch recipe_version is required")
    _check_hash_map(binding.get("code"), "code")
    _check_hash_map(binding.get("assets"), "assets")
    _check_sha(binding.get("recipe_sha256"), "recipe_sha256")
    _check_sha(binding.get("sample_manifest_sha256"), "sample_manifest_sha256")
    if binding["recipe_sha256"] != source_hashes["recipe"] or binding["sample_manifest_sha256"] != source_hashes["sample_manifest"]:
        raise ValueError("Recipe or sample manifest SHA disagrees with source provenance")
    if hashlib.sha256(json.dumps(binding["code"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != source_hashes["code"]:
        raise ValueError("Code mapping SHA disagrees with source provenance")
    if hashlib.sha256(json.dumps(binding["assets"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() != source_hashes["assets"]:
        raise ValueError("Asset mapping SHA disagrees with source provenance")
    return binding


def _check_provenance(provenance: dict, binding: dict, source_hashes: dict) -> dict:
    provenance = _json_copy(provenance)
    if not isinstance(provenance, dict) or provenance.get("format") != BRANCH_FORMAT:
        raise ValueError("Wrong branch format")
    _check_sha(provenance.get("parent_sha256"), "parent_sha256")
    old = provenance.get("parent_binding")
    if (not isinstance(old, dict) or old.get("effective_batch") != 64 or old.get("stage") != "head"
            or old.get("development") != binding.get("development")
            or provenance.get("parent_step") != 2000 or provenance.get("parent_cursor") != 128000):
        raise ValueError("Invalid parent provenance or frozen development binding")
    seed = provenance.get("branch_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Invalid branch seed")
    if provenance.get("source_hashes") != source_hashes:
        raise ValueError("Branch provenance/source SHA mismatch")
    return provenance


def mapping_sha256(mapping: dict) -> str:
    """Digest a canonical filename-to-digest mapping for branch provenance."""
    _check_hash_map(mapping, "mapping")
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _equal_state(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_equal_state(a, b) for a, b in zip(left, right))
    return left == right


def _check_model_optimizer(model: Any, optimizer: Any, model_state: dict, optimizer_state: dict) -> None:
    if not _equal_state(model.state_dict(), model_state) or not _equal_state(optimizer.state_dict(), optimizer_state):
        raise ValueError("Model or Adam state changed while loading checkpoint")


def _write_new_checkpoint(path: str | Path, payload: dict) -> None:
    """fsync then publish without replacing an existing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def branch_sample_ids(start_cursor: int, target_steps: int, effective_batch: int = 64) -> range:
    """Unambiguous absolute sample IDs for deterministic DataLoader samplers."""
    if isinstance(start_cursor, bool) or isinstance(target_steps, bool) or isinstance(effective_batch, bool):
        raise ValueError("Sample positions must be integers")
    if not all(isinstance(x, int) for x in (start_cursor, target_steps, effective_batch)):
        raise ValueError("Sample positions must be integers")
    if effective_batch != 64 or start_cursor < 0 or start_cursor % effective_batch or target_steps < 0 or start_cursor > target_steps * effective_batch:
        raise ValueError("Invalid branch sample cursor or target")
    return range(start_cursor, target_steps * effective_batch)


def create_branch_checkpoint(
    parent_path: str | Path,
    destination: str | Path,
    model: Any,
    optimizer: Any,
    *,
    expected_parent_sha256: str,
    expected_parent_binding: dict,
    branch_binding: dict,
    source_hashes: dict,
    branch_seed: int,
    expected_parent_step: int = 2000,
) -> dict:
    """Load exact parent model/Adam and write a zero-step branch checkpoint.

    `expected_parent_binding` must come from the separately saved old frozen
    protocol. The changed source tree is bound only in `branch_binding`.
    Returns immutable provenance for all later branch checkpoints.
    """
    parent_path, destination = Path(parent_path), Path(destination)
    if parent_path.resolve() == destination.resolve():
        raise ValueError("Branch destination cannot be the parent checkpoint")
    if expected_parent_step != 2000:
        raise ValueError("Only the audited step-2000 parent can start this branch")
    _check_sha(expected_parent_sha256, "expected_parent_sha256")
    if isinstance(branch_seed, bool) or not isinstance(branch_seed, int) or not 0 <= branch_seed < 2**63:
        raise ValueError("Invalid branch seed")
    hashes = _check_source_hashes(source_hashes)
    new_binding = _check_branch_binding(branch_binding, hashes)
    old_binding = _json_copy(expected_parent_binding)
    actual_parent_sha = sha256_file(parent_path)
    if actual_parent_sha != expected_parent_sha256:
        raise ValueError("Parent checkpoint SHA mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if (parent.get("contract") != CONTRACT or parent.get("model_contract") != model.contract()
            or parent.get("binding") != old_binding):
        raise ValueError("Parent contract or stored binding mismatch")
    if (parent.get("step") != expected_parent_step or parent.get("cursor") != expected_parent_step * old_binding["effective_batch"]
            or old_binding["effective_batch"] != 64 or old_binding.get("stage") != "head"):
        raise ValueError("Parent step/cursor/optimizer stage mismatch")
    if new_binding.get("development") != old_binding.get("development"):
        raise ValueError("Frozen development set must remain unchanged across branch")
    model.load_state_dict(parent["model"], strict=True)
    optimizer.load_state_dict(parent["optimizer"])
    _check_model_optimizer(model, optimizer, parent["model"], parent["optimizer"])
    provenance = {
        "format": BRANCH_FORMAT,
        "parent_sha256": actual_parent_sha,
        "parent_step": expected_parent_step,
        "parent_cursor": parent["cursor"],
        "parent_binding": old_binding,
        "branch_seed": branch_seed,
        "source_hashes": hashes,
    }
    # New scene order is independently seeded, never taken from the old cursor.
    torch.manual_seed(branch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(branch_seed)
    save_branch_checkpoint(destination, model, optimizer, branch_step=0,
                           branch_cursor=0, branch_binding=new_binding, provenance=provenance)
    return provenance


def save_branch_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    branch_step: int,
    branch_cursor: int,
    branch_binding: dict,
    provenance: dict,
) -> None:
    """Write one immutable branch checkpoint with independently bound cursor."""
    hashes = _check_source_hashes(provenance.get("source_hashes"))
    binding = _check_branch_binding(branch_binding, hashes)
    provenance = _check_provenance(provenance, binding, hashes)
    if isinstance(branch_step, bool) or not isinstance(branch_step, int) or branch_step < 0:
        raise ValueError("Invalid branch step")
    if isinstance(branch_cursor, bool) or not isinstance(branch_cursor, int) or branch_cursor != branch_step * binding["effective_batch"]:
        raise ValueError("Branch cursor must equal effective batch times branch step")
    payload = {
        "contract": CONTRACT,
        "model_contract": model.contract(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "binding": binding,
        "step": branch_step,
        "cursor": branch_cursor,
        "branch_step": branch_step,
        "branch_cursor": branch_cursor,
        "branch_provenance": provenance,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    _write_new_checkpoint(path, payload)


def restore_branch_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any,
    *,
    expected_binding: dict,
    expected_provenance: dict,
    expected_source_hashes: dict,
) -> tuple[int, int, dict]:
    """Restore model, Adam, RNG and the exact next absolute sample ID."""
    hashes = _check_source_hashes(expected_source_hashes)
    binding = _check_branch_binding(expected_binding, hashes)
    provenance = _check_provenance(expected_provenance, binding, hashes)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("contract") != CONTRACT or checkpoint.get("model_contract") != model.contract()
            or checkpoint.get("binding") != binding or checkpoint.get("branch_provenance") != provenance):
        raise ValueError("Branch contract, data, code or provenance mismatch")
    step, cursor = checkpoint.get("branch_step"), checkpoint.get("branch_cursor")
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or isinstance(cursor, bool) or not isinstance(cursor, int) or cursor != step * binding["effective_batch"]
            or checkpoint.get("step") != step or checkpoint.get("cursor") != cursor):
        raise ValueError("Branch step/sample cursor mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _check_model_optimizer(model, optimizer, checkpoint["model"], checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available():
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        elif step == 0:
            # The zero-step branch can be created on a CPU-only instance and
            # first resumed on the GPU server. Its CUDA RNG is then seeded
            # explicitly, before the first forward/backward update.
            torch.cuda.manual_seed_all(provenance["branch_seed"])
        else:
            raise ValueError("Nonzero GPU checkpoint is missing CUDA RNG state")
    return step, cursor, provenance
