import numpy as np
import pytest

from ptrn_pipeline.step2000_data import BranchSamples, prepare_sequence


class TinyDataset:
    seed = 20260925
    fingerprint = {"fixture": "tiny", "recipe_version": "ptrn-step2000-unified-v1"}
    round_index = 0

    def __len__(self):
        return 7

    def __getitem__(self, index):
        return (self.round_index, index)


def test_sequence_is_full_rounds_and_restart_exact(tmp_path):
    data = TinyDataset()
    first, digest = prepare_sequence(data, 2, tmp_path)
    assert len(first) == 128
    for start in range(0, 126, 7):
        assert sorted(first[start:start + 7].tolist()) == list(range(7))
    assert BranchSamples(data, first)[7] == (1, int(first[7]))
    second, resumed_digest = prepare_sequence(data, 2, tmp_path)
    assert resumed_digest == digest
    assert np.array_equal(first, second)


def test_sequence_refuses_changed_budget_or_corruption(tmp_path):
    data = TinyDataset()
    prepare_sequence(data, 2, tmp_path)
    with pytest.raises(ValueError, match="specification"):
        prepare_sequence(data, 3, tmp_path)
    array = tmp_path / "sample-sequence.npy"
    with array.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="SHA"):
        prepare_sequence(data, 2, tmp_path)
