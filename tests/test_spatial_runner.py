import numpy as np
import pytest

from ptrn_hip.spatial_runner import fixed_small_schedule, statistics


def test_small_source_schedule_has_fixed_coverage_and_heldout_exclusion():
    records = [{'original_size': [50 + i, 100]} for i in range(80)]
    train = list(range(60))
    a = fixed_small_schedule(records, train)
    assert a == fixed_small_schedule(records, train)
    sources = {r['source_index'] for r in a}
    assert len(a) == 128 and len(sources) == 32
    assert sources <= set(train) and {0, 59} <= sources
    assert all(sum(r['source_index'] == i for r in a) == 4 for i in sources)
    assert {r['kind'] for r in a} == {'digital', 'photo'}


def test_fitting_gate_counts_invalid_predictions_and_signed_border_error():
    truth = np.array([[[0, 0], [223, 0], [223, 223], [0, 223]]], dtype=float)
    info = [{'photo_size': [224, 224], 'kind': 'digital'}]
    inverse = np.eye(3)[None]
    exact = statistics(truth, truth, inverse, info)
    assert exact['fit_pass'] and exact['valid_quads'] == 1
    out = truth.copy(); out[0, 0, 0] = -.02
    near = statistics(out, truth, inverse, info)
    assert near['corner_nme_mean'] < .005
    assert not near['fit_pass'] and near['valid_quads'] == 0
    assert near['digital_inward_side_bias_photo_px'][0] < 0


def test_nonfinite_fit_prediction_fails():
    bad = np.full((1, 4, 2), np.nan)
    with pytest.raises(ValueError, match='Nonfinite'):
        statistics(bad, bad, np.eye(3)[None], [])
