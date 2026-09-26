"""A durable, source-agnostic sample sequence for the step-2000 branch.

The array contains dataset view indexes only. It carries no source path, case
identifier or patient grouping metadata. Every training worker consumes the
same cursor-to-view mapping even after an interrupted run is resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .telemetry import atomic_json


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _expected_metadata(dataset, steps: int) -> dict:
    if steps < 1 or len(dataset) < 1:
        raise ValueError("Positive training length and target steps required")
    return {
        "format": "ptrn-step2000-sequence-v1",
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_views_per_round": len(dataset),
        "seed": dataset.seed,
        "effective_batch": 64,
        "steps": steps,
        "total_samples": 64 * steps,
        "permutation": "numpy.default_rng(SeedSequence([seed,round_index,188])).permutation",
    }


def prepare_sequence(dataset, steps: int, output: Path) -> tuple[np.ndarray, str]:
    """Create once or verify an existing cursor map; never replace it silently."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    array_path = output / "sample-sequence.npy"
    metadata_path = output / "sample-sequence.json"
    expected = _expected_metadata(dataset, steps)
    if array_path.exists() != metadata_path.exists():
        raise ValueError("Incomplete sample sequence manifest")
    if not array_path.exists():
        total = expected["total_samples"]
        sequence = np.empty(total, dtype="<i4")
        period = len(dataset)
        for round_index, first in enumerate(range(0, total, period)):
            permutation = np.random.default_rng(
                np.random.SeedSequence([dataset.seed, round_index, 188])
            ).permutation(period)
            end = min(first + period, total)
            sequence[first:end] = permutation[: end - first]
        temporary = array_path.with_name(array_path.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, sequence, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, array_path)
        finally:
            temporary.unlink(missing_ok=True)
        atomic_json(metadata_path, {**expected, "array_sha256": file_sha256(array_path)})
    metadata = json.loads(metadata_path.read_text())
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError("Existing sample sequence specification differs")
    if file_sha256(array_path) != metadata.get("array_sha256"):
        raise ValueError("Sample sequence SHA mismatch")
    sequence = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if sequence.dtype != np.dtype("<i4") or sequence.shape != (expected["total_samples"],):
        raise ValueError("Sample sequence shape or dtype mismatch")
    if int(sequence.min()) < 0 or int(sequence.max()) >= len(dataset):
        raise ValueError("Sample sequence contains an invalid dataset index")
    return sequence, metadata["array_sha256"]


class BranchSamples(Dataset):
    """Map an absolute branch cursor to a deterministic new-recipe view."""

    def __init__(self, dataset, sequence: np.ndarray):
        self.dataset = dataset
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, cursor: int):
        if not 0 <= cursor < len(self.sequence):
            raise IndexError(cursor)
        round_index = cursor // len(self.dataset)
        self.dataset.round_index = round_index
        return self.dataset[int(self.sequence[cursor])]
