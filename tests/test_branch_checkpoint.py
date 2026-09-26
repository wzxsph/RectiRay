"""Step-2000 fork checks use tiny Adam tensors; no patient data or GPU needed."""

import copy

import pytest
import torch

from ptrn_hip.partial_runner import save_checkpoint as save_parent_checkpoint
from ptrn_pipeline.branch_checkpoint import (
    BRANCH_FORMAT,
    branch_sample_ids,
    create_branch_checkpoint,
    mapping_sha256,
    restore_branch_checkpoint,
    save_branch_checkpoint,
    sha256_file,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def contract(self):
        return {"tiny": "same geometry and output"}

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
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


@pytest.fixture
def example(tmp_path):
    parent_model = TinyModel()
    parent_optimizer = torch.optim.Adam(parent_model.parameters(), lr=1e-4)
    _adam_update(parent_model, parent_optimizer)
    old_binding = {"effective_batch": 64, "stage": "head", "development": {"fixed": "old-dev"},
                   "code": {"changed_source.py": "0" * 64}}
    parent = tmp_path / "parent-step-2000.pt"
    save_parent_checkpoint(parent, parent_model, parent_optimizer, step=2000,
                           cursor=128000, binding=old_binding)
    code = {"partial_runner.py": "1" * 64, "new_recipe.py": "2" * 64}
    assets = {"manifest.json": "3" * 64, "backgrounds.json": "4" * 64}
    hashes = {"recipe": "5" * 64, "code": mapping_sha256(code),
              "assets": mapping_sha256(assets), "sample_manifest": "6" * 64}
    new_binding = {"effective_batch": 64, "stage": "head", "development": old_binding["development"],
                   "recipe_version": "step2000-unified-v1", "recipe_sha256": hashes["recipe"],
                   "sample_manifest_sha256": hashes["sample_manifest"], "code": code, "assets": assets}
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return parent, parent_model, parent_optimizer, old_binding, new_binding, hashes, model, optimizer


def _create(example, path):
    parent, _, _, old_binding, new_binding, hashes, model, optimizer = example
    return create_branch_checkpoint(parent, path, model, optimizer,
                                    expected_parent_sha256=sha256_file(parent),
                                    expected_parent_binding=old_binding,
                                    branch_binding=new_binding, source_hashes=hashes,
                                    branch_seed=20260925)


def test_branch_loads_exact_parent_adam_and_starts_new_sample_stream(example, tmp_path):
    parent, parent_model, parent_optimizer, old_binding, new_binding, hashes, model, optimizer = example
    branch = tmp_path / "branch-step-0.pt"
    provenance = _create(example, branch)
    assert _equal(model.state_dict(), parent_model.state_dict())
    assert _equal(optimizer.state_dict(), parent_optimizer.state_dict())
    assert provenance["format"] == BRANCH_FORMAT
    assert provenance["parent_sha256"] == sha256_file(parent)
    assert provenance["parent_step"] == 2000 and provenance["parent_cursor"] == 128000
    assert provenance["parent_binding"] == old_binding
    assert provenance["source_hashes"] == hashes
    contents = torch.load(branch, map_location="cpu", weights_only=True)
    assert contents["step"] == contents["branch_step"] == 0
    assert contents["cursor"] == contents["branch_cursor"] == 0
    assert contents["binding"] == new_binding
    assert list(branch_sample_ids(contents["cursor"], 2)) == list(range(128))


def test_branch_resume_preserves_rng_adam_and_next_sample_id(example, tmp_path):
    _, _, _, _, binding, hashes, model, optimizer = example
    provenance = _create(example, tmp_path / "step-0.pt")
    uninterrupted = list(branch_sample_ids(0, 5))
    consumed = list(branch_sample_ids(0, 3))
    for _ in range(3):
        _adam_update(model, optimizer)
    saved_model, saved_adam = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    step3 = tmp_path / "step-3.pt"
    save_branch_checkpoint(step3, model, optimizer, branch_step=3, branch_cursor=192,
                           branch_binding=binding, provenance=provenance)
    expected_random = torch.rand(8)
    _adam_update(model, optimizer)
    torch.rand(17)
    step, cursor, restored_provenance = restore_branch_checkpoint(
        step3, model, optimizer, expected_binding=binding,
        expected_provenance=provenance, expected_source_hashes=hashes)
    assert (step, cursor) == (3, 192)
    assert restored_provenance == provenance
    assert _equal(model.state_dict(), saved_model)
    assert _equal(optimizer.state_dict(), saved_adam)
    assert torch.equal(torch.rand(8), expected_random)
    assert consumed + list(branch_sample_ids(cursor, 5)) == uninterrupted


def test_rejects_wrong_parent_and_new_source_binding(example, tmp_path):
    parent, _, _, old_binding, binding, hashes, model, optimizer = example
    target = tmp_path / "branch.pt"
    with pytest.raises(ValueError, match="Parent checkpoint SHA mismatch"):
        create_branch_checkpoint(parent, target, model, optimizer,
                                 expected_parent_sha256="f" * 64,
                                 expected_parent_binding=old_binding,
                                 branch_binding=binding, source_hashes=hashes,
                                 branch_seed=9)
    assert not target.exists()
    wrong_old = {**old_binding, "code": {"changed_source.py": "9" * 64}}
    with pytest.raises(ValueError, match="stored binding mismatch"):
        create_branch_checkpoint(parent, target, model, optimizer,
                                 expected_parent_sha256=sha256_file(parent),
                                 expected_parent_binding=wrong_old,
                                 branch_binding=binding, source_hashes=hashes,
                                 branch_seed=9)
    assert not target.exists()
    provenance = _create(example, target)
    wrong_hashes = {**hashes, "assets": "f" * 64}
    with pytest.raises(ValueError):
        restore_branch_checkpoint(target, model, optimizer, expected_binding=binding,
                                  expected_provenance=provenance,
                                  expected_source_hashes=wrong_hashes)
    with pytest.raises(FileExistsError):
        _create(example, target)


def test_rejects_cursor_and_development_drift(example, tmp_path):
    parent, _, _, old_binding, binding, hashes, model, optimizer = example
    bad = {**binding, "development": {"fixed": "different"}}
    with pytest.raises(ValueError, match="Frozen development"):
        create_branch_checkpoint(parent, tmp_path / "bad.pt", model, optimizer,
                                 expected_parent_sha256=sha256_file(parent),
                                 expected_parent_binding=old_binding,
                                 branch_binding=bad, source_hashes=hashes,
                                 branch_seed=9)
    provenance = _create(example, tmp_path / "step-0.pt")
    with pytest.raises(ValueError, match="cursor"):
        save_branch_checkpoint(tmp_path / "step-1.pt", model, optimizer,
                               branch_step=1, branch_cursor=63,
                               branch_binding=binding, provenance=provenance)
    with pytest.raises(ValueError, match="cursor"):
        branch_sample_ids(65, 10)
