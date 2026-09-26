import json
from pathlib import Path

import numpy as np
import pytest

from ptrn_hip.spatial_policy_audit import (
    PERFECT_D_METADATA_SHA256, PERFECT_D_REGRESSION, _destination,
    _focused, _one, _perfect_prediction, main, verify_perfect_regression)
from ptrn_hip.unified_geometry import decode_prediction, inverse_letterbox_matrix
from ptrn_hip.spatial_geometry import plan_transform
from ptrn_hip.unified_evaluation import prepare_reference
from ptrn_hip.ratio_synthesis import pixel_corners


def test_audit_destination_must_be_distinct_local_run(tmp_path):
    prior = tmp_path / '.local' / 'old'
    prior.mkdir(parents=True)
    assert _destination(prior, tmp_path / '.local' / 'new').name == 'new'
    with pytest.raises(ValueError):
        _destination(prior, prior / 'nested')
    with pytest.raises(ValueError):
        _destination(prior, tmp_path / 'untracked')


def test_perfect_prediction_uses_current_h8_plus_delta_contract():
    q = pixel_corners(401, 301)
    info = {'photo_size': [401, 301], 'quad_photo': q.tolist(), 'log_ratio': float(np.log(401 / 301))}
    prediction = _perfect_prediction(info, 384)
    decoded = decode_prediction(prediction, 384, info['photo_size'],
                                inverse_letterbox_matrix(info['photo_size'], 384))
    assert decoded['valid']
    np.testing.assert_allclose(decoded['quad'], q, atol=.001)
    assert decoded['ratio'] == pytest.approx(401 / 301, rel=1e-6)


def test_audit_evaluation_introduces_truth_only_after_action(monkeypatch):
    q = pixel_corners(401, 301)
    info = {'photo_size': [401, 301], 'quad_photo': q.tolist(),
            'source_size': [401, 301], 'source_to_photo': np.eye(3).tolist(),
            'log_ratio': float(np.log(401 / 301)), 'kind': 'digital', 'mode': 'identity',
            'source_index': 0, 'condition': 'keep'}
    reference = prepare_reference(info['photo_size'], info['source_size'], info['source_to_photo'], q)
    prediction = _perfect_prediction(info, 224)
    seen = []

    def predicted_only(photo_size, quad, ratio, thresholds):
        seen.append((tuple(photo_size), np.asarray(quad).shape, float(ratio)))
        return plan_transform(photo_size, quad, ratio, thresholds)

    row = _one(prediction, info, 224, reference, predicted_only, {'margin_fraction': 0.})
    assert len(seen) == 1 and seen[0][1] == (4, 2)
    assert row['action'] == 'keep' and row['success']


def _synthetic_full_D_aggregate():
    """Only counters, never actual D records or patient images in tests."""
    all_rows = {**PERFECT_D_REGRESSION,
                'automatic_success_rate': 5038 / 5258,
                'undercorrected_rate': 0., 'unnecessary_warp_rate': 0.}
    summary = {'all': all_rows}
    result = {'summary': {'spatial': summary}, 'focused': {'spatial': _focused(summary)}}
    return {'scope': 'D_perfect', 'input_size': 384, 'views': 5258,
            'margin_fraction': 0., 'prior_D_metadata_sha256': PERFECT_D_METADATA_SHA256,
            'settings': {'default': result, 'J384_trial8': {
                'summary': {'spatial': {'all': dict(all_rows)}},
                'focused': {'spatial': dict(result['focused']['spatial'])}}}}


def test_full_D_perfect_regression_accepts_only_bound_counts_and_sha():
    report = _synthetic_full_D_aggregate()
    proof = verify_perfect_regression(report)
    assert proof['passed'] and proof['metadata_sha256'] == PERFECT_D_METADATA_SHA256
    wrong = _synthetic_full_D_aggregate()
    wrong['prior_D_metadata_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='metadata_SHA'):
        verify_perfect_regression(wrong)
    wrong = _synthetic_full_D_aggregate()
    wrong['views'] = 5257
    with pytest.raises(ValueError, match='view_denominator'):
        verify_perfect_regression(wrong)


@pytest.mark.parametrize('key,value', (
    ('undercorrected', 298), ('warp_required_count', 2373),
    ('corner_missing', 1), ('direction_error', 1),
    ('unnecessary_warp', 1), ('automatic_count', 4962),
    ('success_count', 4962), ('action_counts', {'keep': 1434, 'crop': 1399,
                                               'warp': 2206, 'needs_confirmation': 219}),
))
def test_full_D_perfect_regression_rejects_wrong_failure_or_denominator(key, value):
    report = _synthetic_full_D_aggregate()
    report['settings']['default']['summary']['spatial']['all'][key] = value
    with pytest.raises(ValueError, match='perfect_regression_default_'):
        verify_perfect_regression(report)


def test_perfect_regression_refuses_model_only_or_nonzero_margin(tmp_path):
    args = ['--prior-run', str(tmp_path / 'old'), '--run', str(tmp_path / 'new'),
            '--verify-perfect-regression']
    with pytest.raises(SystemExit) as rejected:
        main([*args, '--scope', 'models'])
    assert rejected.value.code == 2
    with pytest.raises(SystemExit) as rejected:
        main([*args, '--margin-fraction', '.005'])
    assert rejected.value.code == 2
    assert not (tmp_path / 'new').exists()
