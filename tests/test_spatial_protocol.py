"""CPU-only evidence checks for the reproducible Stage B decision CLI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ptrn_hip.ratio_data import sha256
from ptrn_hip.spatial_protocol import FOLDERS, summarize_small
from ptrn_hip.spatial_runner import ARMS, statistics


def _dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _fixture(run: Path, folder='B', step=500, passes=True):
    size = 224
    truth = np.tile(np.array([[0., 0.], [223., 0.], [223., 223.], [0., 223.]],
                             dtype=np.float32), (128, 1, 1))
    inverse = np.tile(np.eye(3), (128, 1, 1))
    metadata = [{'kind': 'digital', 'photo_size': [224, 224]} for _ in range(128)]
    data_path = run / 'B/views.npz'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, pixels=np.zeros((128, size, size, 3), np.uint8),
                        quad=truth, inverse=inverse, log_ratio=np.zeros(128))
    _dump(run / 'B/dataset.json', {'pixels_sha256': sha256(data_path), 'views': 128,
                                  'input_size': size, 'metadata': metadata})
    binding = {'sha256': 'source-hash'}
    _dump(run / folder / 'source-binding.json', binding)
    for arm in ARMS:
        base = run / folder / arm
        base.mkdir(parents=True, exist_ok=True)
        prediction = truth.copy()
        if not passes:
            prediction[:, :, 0] += 12
        np.save(base / f'quad-{step:04d}.npy', prediction)
        evaluation = statistics(prediction, truth, inverse, metadata)
        evaluation.update(step=step, arm=arm, stage=FOLDERS[folder], max_reserved_mib=1.)
        config = {'arm': arm, 'stage': FOLDERS[folder], 'size': size,
                  'dataset_sha256': sha256(data_path), 'source_sha256': 'source-hash',
                  'g0_sha256': 'g0-hash', 'b11_sha256': 'b11-hash',
                  'seed': 20260923, 'effective_batch': 16,
                  'contract': {'size': size}, 'GT_ratio': 'diagnostic_only_no_ratio_training',
                  'checkpoints': [500, 2000, 5000]}
        checkpoint = base / f'step-{step:04d}.pt'
        checkpoint.write_bytes(f'{arm} {step}'.encode())
        _dump(base / 'config.json', config)
        _dump(base / f'eval-{step:04d}.json', evaluation)
        _dump(base / f'step-{step:04d}.json', {'sha256': sha256(checkpoint),
                                               'step': step, 'config': config,
                                               'evaluation': evaluation})


@pytest.mark.parametrize('folder,step', [('B', 500), ('B-conv5', 2000),
                                         ('B-conv5-cached', 5000)])
def test_pass_decision_and_idempotence(tmp_path, folder, step):
    _fixture(tmp_path, folder, step)
    decision = summarize_small(tmp_path, folder, step)
    assert decision['pass_arms'] == list(ARMS)
    assert decision['stage'] == FOLDERS[folder]
    assert summarize_small(tmp_path, folder, step) == decision
    saved = tmp_path / f'{folder}-decision-{step:04d}.json'
    assert json.loads(saved.read_text()) == decision


def test_failed_fit_still_gets_faithful_decision(tmp_path):
    _fixture(tmp_path, passes=False)
    decision = summarize_small(tmp_path, 'B', 500)
    assert decision['pass_arms'] == []
    assert all(not record['fit_pass'] for record in decision['arms'].values())


def test_missing_arm_refused_without_decision(tmp_path):
    _fixture(tmp_path)
    (tmp_path / 'B/spatial_heatmap/eval-0500.json').unlink()
    with pytest.raises(RuntimeError, match='Missing B evidence'):
        summarize_small(tmp_path, 'B', 500)
    assert not (tmp_path / 'B-decision-0500.json').exists()


@pytest.mark.parametrize('tamper', ['checkpoint', 'quad', 'eval', 'sidecar'])
def test_tampered_evidence_refused(tmp_path, tamper):
    _fixture(tmp_path)
    base = tmp_path / 'B/spatial_quad'
    if tamper == 'checkpoint':
        (base / 'step-0500.pt').write_bytes(b'tampered')
    elif tamper == 'quad':
        quad = np.load(base / 'quad-0500.npy')
        quad[:, :, 0] += 2
        np.save(base / 'quad-0500.npy', quad)
    elif tamper == 'eval':
        p = base / 'eval-0500.json'
        value = json.loads(p.read_text())
        value['corner_nme_mean'] += .1
        _dump(p, value)
    else:
        p = base / 'step-0500.json'
        value = json.loads(p.read_text())
        value['evaluation']['fit_pass'] = False
        _dump(p, value)
    with pytest.raises(RuntimeError):
        summarize_small(tmp_path, 'B', 500)
    assert not (tmp_path / 'B-decision-0500.json').exists()


@pytest.mark.parametrize('key,value', [('seed', 1), ('effective_batch', 8),
                                       ('g0_sha256', 'different'),
                                       ('dataset_sha256', 'different'),
                                       ('stage', 'last_block')])
def test_mixed_four_arm_binding_refused(tmp_path, key, value):
    _fixture(tmp_path)
    base = tmp_path / 'B/spatial_heatmap'
    for name in ('config.json', 'step-0500.json'):
        path = base / name
        record = json.loads(path.read_text())
        config = record if name == 'config.json' else record['config']
        config[key] = value
        _dump(path, record)
    with pytest.raises(RuntimeError):
        summarize_small(tmp_path, 'B', 500)


def test_existing_different_decision_is_not_overwritten(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / 'B-decision-0500.json'
    _dump(path, {'stage': 'head', 'step': 500, 'arms': {}, 'pass_arms': []})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match='refusing overwrite'):
        summarize_small(tmp_path, 'B', 500)
    assert path.read_bytes() == before


def test_resume_peak_memory_is_not_model_metric(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / 'B/gap_h8/eval-0500.json'
    record = json.loads(path.read_text())
    record['max_reserved_mib'] = 2.
    _dump(path, record)
    assert summarize_small(tmp_path, 'B', 500)['pass_arms'] == list(ARMS)
