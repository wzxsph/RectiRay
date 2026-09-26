"""Code migration must not advance or alter an interrupted training state."""

import json

import numpy as np
import pytest
import torch

from ptrn_pipeline.branch_checkpoint import (
    BRANCH_FORMAT,
    mapping_sha256,
    restore_branch_checkpoint,
    save_branch_checkpoint,
    sha256_file,
)
from ptrn_pipeline.branch_upgrade import (
    PERMITTED_CODE_CHANGES,
    current_code_mapping,
    upgrade_branch_checkpoint,
    upgrade_run_directory,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def contract(self):
        return {"tiny": "unchanged"}

    def forward(self, values):
        return self.layer(values)


def _adam_update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(4, 2)).square().mean().backward()
    optimizer.step()


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return left == right


@pytest.fixture
def example(tmp_path):
    root = tmp_path / "source"
    files = {
        "src/ptrn_hip/partial_model.py": "model unchanged\n",
        "src/ptrn_pipeline/step2000_runner.py": "old runner\n",
        "src/ptrn_pipeline/async_eval.py": "old evaluator\n",
        "pyproject.toml": "old project\n",
        "configs/pipeline-v1.json": "{}\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    old_code = current_code_mapping(root)
    asset = root / "assets/manifest.json"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text("unchanged source assets\n")
    assets = {"manifest.json": sha256_file(asset)}
    old_hashes = {"recipe": "b" * 64, "code": mapping_sha256(old_code),
                  "assets": mapping_sha256(assets), "sample_manifest": "c" * 64}
    binding = {"effective_batch": 64, "stage": "head", "development": {"fixed": "old exam"},
               "recipe_version": "ptrn-step2000-unified-v1", "recipe_sha256": old_hashes["recipe"],
               "sample_manifest_sha256": old_hashes["sample_manifest"], "code": old_code, "assets": assets}
    provenance = {"format": BRANCH_FORMAT, "parent_sha256": "d" * 64,
                  "parent_step": 2000, "parent_cursor": 128000,
                  "parent_binding": {"effective_batch": 64, "stage": "head",
                                     "development": binding["development"]},
                  "branch_seed": 20260926, "source_hashes": old_hashes}
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _adam_update(model, optimizer)
    # The test sequence contains no cases or image data; only its SHA matters.
    sequence = old_run / "sample-sequence.npy"
    np.save(sequence, np.array([0, 1, 2], dtype="<i4"), allow_pickle=False)
    sequence_sha = sha256_file(sequence)
    binding["sample_manifest_sha256"] = sequence_sha
    provenance["source_hashes"]["sample_manifest"] = sequence_sha
    checkpoint = old_run / "branch-step-250.pt"
    save_branch_checkpoint(checkpoint, model, optimizer, branch_step=250,
                           branch_cursor=16000, branch_binding=binding, provenance=provenance)
    (old_run / "branch-binding.json").write_text(json.dumps(binding))
    (old_run / "branch-provenance.json").write_text(json.dumps(provenance))
    (old_run / "sample-sequence.json").write_text(json.dumps({"array_sha256": sequence_sha}))
    old_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)

    for relative, content in {
        "src/ptrn_pipeline/step2000_runner.py": "new runner resource schedule\n",
        "src/ptrn_pipeline/async_eval.py": "new evaluator resource policy\n",
        "src/ptrn_pipeline/branch_upgrade.py": "audited migration tool\n",
    }.items():
        (root / relative).write_text(content)
    return root, old_run, checkpoint, binding, provenance, old_checkpoint


def _upgrade(example, destination):
    root, _, checkpoint, binding, provenance, _ = example
    return upgrade_branch_checkpoint(checkpoint, destination,
        expected_source_sha256=sha256_file(checkpoint), expected_old_binding=binding,
        expected_old_provenance=provenance, source_root=root,
        asset_root=root / "assets", sample_sequence_path=example[1] / "sample-sequence.npy",
        approved_code_changes=sorted(PERMITTED_CODE_CHANGES))


def test_preserves_model_adam_rng_cursor_and_unchanged_bindings(example, tmp_path):
    _, _, checkpoint, old_binding, old_provenance, before = example
    destination = tmp_path / "upgraded.pt"
    result = _upgrade(example, destination)
    after = torch.load(destination, map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "step", "cursor",
                "branch_step", "branch_cursor", "contract", "model_contract"):
        assert _equal(before[key], after[key]), key
    assert (after["step"], after["cursor"]) == (250, 16000)
    assert {key: value for key, value in result["binding"].items() if key != "code"} == {
        key: value for key, value in old_binding.items() if key != "code"}
    assert {key: value for key, value in result["source_hashes"].items() if key != "code"} == {
        key: value for key, value in old_provenance["source_hashes"].items() if key != "code"}
    assert result["migration"]["source_checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["checkpoint_sha256"] == sha256_file(destination)
    assert sorted(result["migration"]["changed_code"]) == sorted(PERMITTED_CODE_CHANGES)

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    step, cursor, restored = restore_branch_checkpoint(destination, model, optimizer,
        expected_binding=result["binding"], expected_provenance=result["provenance"],
        expected_source_hashes=result["source_hashes"])
    assert (step, cursor) == (250, 16000)
    assert restored == result["provenance"]
    assert _equal(model.state_dict(), before["model"])
    assert _equal(optimizer.state_dict(), before["optimizer"])
    with pytest.raises(FileExistsError):
        _upgrade(example, destination)


def test_rejects_wrong_sha_metadata_or_unreviewed_code_change(example, tmp_path):
    root, _, checkpoint, binding, provenance, _ = example
    target = tmp_path / "should-not-exist.pt"
    kwargs = dict(expected_source_sha256=sha256_file(checkpoint),
                  expected_old_binding=binding, expected_old_provenance=provenance,
                  source_root=root, asset_root=root / "assets",
                  sample_sequence_path=example[1] / "sample-sequence.npy",
                  approved_code_changes=sorted(PERMITTED_CODE_CHANGES))
    with pytest.raises(ValueError, match="Source checkpoint SHA mismatch"):
        upgrade_branch_checkpoint(checkpoint, target, **{**kwargs, "expected_source_sha256": "0" * 64})
    with pytest.raises(ValueError, match="development binding"):
        upgrade_branch_checkpoint(checkpoint, target,
            **{**kwargs, "expected_old_binding": {**binding, "development": {"fixed": "other"}}})
    with pytest.raises(ValueError, match="explicit approval"):
        upgrade_branch_checkpoint(checkpoint, target,
            **{**kwargs, "approved_code_changes": ["src/ptrn_pipeline/async_eval.py"]})
    (root / "src/ptrn_hip/partial_model.py").write_text("changed architecture\n")
    with pytest.raises(ValueError, match="outside the permitted set"):
        upgrade_branch_checkpoint(checkpoint, target,
            **{**kwargs, "approved_code_changes": sorted([*PERMITTED_CODE_CHANGES,
                                                            "src/ptrn_hip/partial_model.py"])})
    (root / "src/ptrn_hip/partial_model.py").write_text("model unchanged\n")
    (root / "assets/manifest.json").write_text("asset silently replaced\n")
    with pytest.raises(ValueError, match="Bound asset changed"):
        upgrade_branch_checkpoint(checkpoint, target, **kwargs)
    (root / "assets/manifest.json").write_text("unchanged source assets\n")
    (example[1] / "sample-sequence.npy").write_bytes(b"different sequence")
    with pytest.raises(ValueError, match="Bound sample sequence changed"):
        upgrade_branch_checkpoint(checkpoint, target, **kwargs)
    assert not target.exists()


def test_publishes_separate_resume_run_with_exact_sample_sequence(example, tmp_path):
    root, old_run, old_checkpoint, _, _, _ = example
    new_run = tmp_path / "new-run"
    result = upgrade_run_directory(old_run, new_run,
        expected_checkpoint_sha256=sha256_file(old_checkpoint), source_root=root,
        asset_root=root / "assets",
        approved_code_changes=sorted(PERMITTED_CODE_CHANGES))
    assert (new_run / "branch-step-250.pt").exists()
    assert (new_run / "branch-binding.json").exists()
    assert (new_run / "branch-provenance.json").exists()
    assert sha256_file(new_run / "sample-sequence.npy") == sha256_file(old_run / "sample-sequence.npy")
    assert json.loads((new_run / "branch-binding.json").read_text()) == result["binding"]
    assert json.loads((new_run / "branch-provenance.json").read_text()) == result["provenance"]
    assert sha256_file(old_checkpoint) == result["migration"]["source_checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        upgrade_run_directory(old_run, new_run,
            expected_checkpoint_sha256=sha256_file(old_checkpoint), source_root=root,
            asset_root=root / "assets",
            approved_code_changes=sorted(PERMITTED_CODE_CHANGES))
