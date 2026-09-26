"""Synthetic checks for the independent, data-safe review tools."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.compare_real_photos import (PARENT_TRAINING_ONLY_UPDATES, RENDERER_PATH,
                                       RENDERER_SHA256, validate_code_binding)
from tools.summarize_development import checked_run, describe, summarize


def test_reviewed_renderer_requires_explicit_opt_in():
    repo = Path(__file__).resolve().parents[1]
    source = repo / RENDERER_PATH
    assert hashlib.sha256(source.read_bytes()).hexdigest() == RENDERER_SHA256
    checkpoint = {"binding": {"code": {RENDERER_PATH: "0" * 64}}}
    with pytest.raises(ValueError, match="source binding differs"):
        validate_code_binding(checkpoint, repo, allow_renderer_update=False)
    changes = validate_code_binding(checkpoint, repo, allow_renderer_update=True)
    assert len(changes) == 1
    assert changes[0]["current_sha256"] == RENDERER_SHA256
    with pytest.raises(ValueError, match="no source binding"):
        validate_code_binding({"binding": {"code": {}}}, repo, allow_renderer_update=True)


def test_both_checkpoint_code_key_formats_are_verified():
    repo = Path(__file__).resolve().parents[1]
    old, current = PARENT_TRAINING_ONLY_UPDATES["partial_data.py"]
    assert hashlib.sha256((repo / "src/ptrn_hip/partial_data.py").read_bytes()).hexdigest() == current
    parent = {"binding": {"code": {"partial_data.py": old}}}
    with pytest.raises(ValueError, match="source binding differs"):
        validate_code_binding(parent, repo, allow_renderer_update=True)
    assert len(validate_code_binding(parent, repo, allow_renderer_update=True,
                                     allow_parent_source_updates=True)) == 1
    pyproject = hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest()
    branch = {"binding": {"code": {"pyproject.toml": pyproject, RENDERER_PATH: "0" * 64}}}
    assert len(validate_code_binding(branch, repo, allow_renderer_update=True)) == 1


def test_invalid_geometry_still_contributes_coordinate_and_ratio_error():
    rows = [
        {"valid_geometry": True, "additional_crop_fraction": .1, "extra_background_fraction": .2,
         "corner_error_diagonal": .01, "absolute_ratio_error": .02},
        {"valid_geometry": False, "corner_error_diagonal": .09, "absolute_ratio_error": .18},
    ]
    result = describe(rows)
    assert result["valid_geometry"] == 1
    assert result["additional_crop_fraction"]["n"] == 1
    assert result["additional_crop_fraction"]["mean"] == .1
    assert result["corner_error_diagonal"]["n"] == 2
    assert result["corner_error_diagonal"]["mean"] == pytest.approx(.05)
    assert result["absolute_ratio_error"]["mean"] == pytest.approx(.1)


def test_aggregate_rejects_scene_mismatch():
    def run(kind):
        return {"name": kind, "binding": {"development": {"split_name": "dev"}, "step": 1,
                 "checkpoint_sha256": "0" * 64, "batch": 1, "prediction_backend": "cuda_graph"},
                "rows": [{"kind": kind, "integrity": "complete", "background_family": "plain",
                          "input_missing_fraction": 0, "valid_geometry": True,
                          "additional_crop_fraction": 0, "extra_background_fraction": 0,
                          "corner_error_diagonal": 0, "absolute_ratio_error": 0}]}
    with pytest.raises(ValueError, match="Ordered development scenes differ"):
        summarize([run("original"), run("photo")])


def test_complete_evaluation_requires_every_prediction(tmp_path):
    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding = {"count": 2, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
               "step": 2, "development": {"split_name": "dev"}, "runtime_code": {},
               "batch": 1, "prediction_backend": "cuda_graph"}
    rows = [{"index": i, "valid_geometry": True} for i in range(2)]
    for name, data in {
        "binding.json": binding,
        "inference-complete.json": {"count": 2, "checkpoint_sha256": digest},
        "metrics-complete.json": {"count": 2},
        "evaluation.json": {"count": 2, "full_development": True, "execution": binding,
                            "valid_geometry": 2, "invalid_geometry": 0, "rows": rows},
        "pred-000000.json": [{"index": 0}],
    }.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete prediction shards"):
        checked_run("tiny", tmp_path, 2)
    (tmp_path / "pred-000001.json").write_text(json.dumps([{"index": 1}]), encoding="utf-8")
    assert checked_run("tiny", tmp_path, 2)["binding"]["checkpoint_sha256"] == digest
