"""CPU-only, reproducible shared-checkpoint decision for Stage B small fit.

The decision describes fit on the fixed 128 training views, not generalization.
It reads saved predictions and evidence; it never loads or runs a model.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path

import numpy as np

from .data import write_json
from .ratio_data import sha256
from .spatial_runner import ARMS, CHECKPOINTS, statistics

FOLDERS = {'B': 'head', 'B-conv5': 'last_block',
           'B-conv5-cached': 'last_block'}
_BINDING_KEYS = ('dataset_sha256', 'source_sha256', 'g0_sha256',
                 'b11_sha256', 'seed', 'effective_batch', 'size', 'stage',
                 'contract', 'GT_ratio', 'checkpoints')
_STAT_KEYS = ('count', 'corner_nme_mean', 'corner_nme_p90',
              'worst_corner_nme', 'valid_quads', 'photo_nme_mean',
              'digital_inward_side_bias_photo_px', 'by_kind', 'fit_pass')


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f'Missing B evidence: {path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise RuntimeError(f'Expected JSON object: {path}')
    return value


def _same(expected, actual) -> bool:
    """Compare recomputed metrics with stored JSON, including nested breakdowns."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and expected.keys() == actual.keys()
                and all(_same(value, actual[key]) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(expected) == len(actual)
                and all(_same(a, b) for a, b in zip(expected, actual)))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=1e-10, abs_tol=1e-12)
    return expected == actual


def _small_data(run: Path):
    path = run / 'B' / 'views.npz'
    meta = _read_json(run / 'B' / 'dataset.json')
    if not path.is_file() or sha256(path) != meta.get('pixels_sha256'):
        raise RuntimeError('Fixed B dataset is missing or differs from dataset.json')
    if meta.get('views') != 128 or meta.get('input_size') != 224:
        raise RuntimeError('Unexpected fixed B dataset contract')
    with np.load(path, allow_pickle=False) as data:
        truth = np.asarray(data['quad'])
        inverse = np.asarray(data['inverse'])
        pixels = data['pixels']
        if pixels.shape != (128, 224, 224, 3):
            raise RuntimeError('Unexpected fixed B image tensor shape')
    if truth.shape != (128, 4, 2) or inverse.shape != (128, 3, 3):
        raise RuntimeError('Unexpected fixed B geometry tensor shape')
    if not isinstance(meta.get('metadata'), list) or len(meta['metadata']) != 128:
        raise RuntimeError('Unexpected fixed B metadata count')
    return meta, truth, inverse


def summarize_small(run: Path, folder: str, step: int) -> dict:
    """Validate four-arm evidence and atomically create an idempotent C decision."""
    run = Path(run)
    if folder not in FOLDERS or step not in CHECKPOINTS:
        raise ValueError('Expected an approved Stage B folder and shared checkpoint')
    stage = FOLDERS[folder]
    dataset, truth, inverse = _small_data(run)
    source = _read_json(run / folder / 'source-binding.json')
    common = None
    arms = {}
    for arm in ARMS:
        base = run / folder / arm
        config = _read_json(base / 'config.json')
        sidecar = _read_json(base / f'step-{step:04d}.json')
        evaluation = _read_json(base / f'eval-{step:04d}.json')
        checkpoint = base / f'step-{step:04d}.pt'
        quad_path = base / f'quad-{step:04d}.npy'
        if not checkpoint.is_file() or not quad_path.is_file():
            raise RuntimeError(f'Missing checkpoint or predictions for {arm} at {step}')
        if sidecar.get('sha256') != sha256(checkpoint):
            raise RuntimeError(f'Checkpoint SHA differs for {arm} at {step}')
        if sidecar.get('step') != step or sidecar.get('config') != config:
            raise RuntimeError(f'Checkpoint sidecar config or step differs for {arm}')
        # A later resume re-evaluates the starting step and overwrites eval JSON.
        # Peak CUDA allocation can legitimately change; all model metrics and
        # the step/arm/stage identity must remain identical to the sidecar.
        side_eval = sidecar.get('evaluation')
        if (not isinstance(side_eval, dict)
                or {k: v for k, v in side_eval.items() if k != 'max_reserved_mib'}
                != {k: v for k, v in evaluation.items() if k != 'max_reserved_mib'}):
            raise RuntimeError(f'Checkpoint sidecar evaluation differs for {arm}')
        if (config.get('arm') != arm or config.get('stage') != stage
                or config.get('size') != 224
                or config.get('dataset_sha256') != dataset['pixels_sha256']
                or config.get('source_sha256') != source.get('sha256')):
            raise RuntimeError(f'B data, source, arm, size or stage binding differs for {arm}')
        binding = {key: config.get(key) for key in _BINDING_KEYS}
        if any(value is None for value in binding.values()):
            raise RuntimeError(f'Incomplete B configuration binding for {arm}')
        if common is None:
            common = binding
        elif binding != common:
            raise RuntimeError(f'Four-arm B configuration bindings differ at {arm}')
        if (evaluation.get('arm') != arm or evaluation.get('stage') != stage
                or evaluation.get('step') != step):
            raise RuntimeError(f'B evaluation arm, stage or step differs for {arm}')
        prediction = np.load(quad_path, allow_pickle=False)
        calculated = statistics(prediction, truth, inverse, dataset['metadata'], size=224)
        for key in _STAT_KEYS:
            if key not in evaluation or not _same(calculated[key], evaluation[key]):
                raise RuntimeError(f'B {arm} evaluation {key} differs from saved predictions')
        # The checkpoint sidecar is immutable across resumes.  A resume may
        # overwrite eval JSON with a different peak-memory observation only.
        arms[arm] = side_eval
    decision = {'stage': stage, 'step': step, 'arms': arms,
                'pass_arms': [arm for arm in ARMS if arms[arm]['fit_pass']]}
    path = run / f'{folder}-decision-{step:04d}.json'
    # Serialize the read/check/write cycle so a concurrent invocation cannot
    # replace a different decision between the existence check and rename.
    with (run / '.spatial-protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if _read_json(path) != decision:
                raise RuntimeError(f'Existing B decision differs; refusing overwrite: {path}')
        else:
            write_json(path, decision)
    return decision


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['summarize-small'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--folder', choices=tuple(FOLDERS), required=True)
    parser.add_argument('--step', type=int, choices=CHECKPOINTS, required=True)
    args = parser.parse_args(argv)
    result = summarize_small(args.run, args.folder, args.step)
    print(json.dumps({'folder': args.folder, 'step': args.step,
                      'pass_arms': result['pass_arms'],
                      'decision': str(args.run / f'{args.folder}-decision-{args.step:04d}.json')}))


if __name__ == '__main__':
    main()
