"""CPU-only D audit of frozen unified versus predicted-grid spatial policies.

Reads only the prior run's D metadata and cached predictions. Every artifact is
written to a separate ignored .local run. No source images, T, GPU or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .ratio_synthesis import UNIT, normalization, project
from .spatial_geometry import DEFAULT_THRESHOLDS as SPATIAL_DEFAULTS
from .spatial_geometry import plan_transform as spatial_plan
from .unified_geometry import (DEFAULT_THRESHOLDS as FROZEN_DEFAULTS,
                               decode_prediction, edge_ratio,
                               inverse_letterbox_matrix, plan_transform as frozen_plan)
from .unified_evaluation import (admission_gates, measure_final, measure_geometry,
                                 prepare_reference, summarize_rows)

GROUPS = ('S224', 'F224', 'J224', 'J384')
PERFECT_D_METADATA_SHA256 = 'bb5f1adf2128ac89922497982dd66b44c960289dc1f3daf07c635b07b89999b1'
PERFECT_D_REGRESSION = {
    'count': 5258, 'source_count': 478, 'invalid': 0, 'valid_count': 5258,
    'automatic_count': 5038, 'success_count': 5038,
    'action_counts': {'keep': 1434, 'crop': 1398, 'warp': 2206,
                      'needs_confirmation': 220},
    'final_invalid': 0, 'corner_missing': 0, 'direction_error': 0,
    'crop_allowed_count': 2884, 'warp_required_count': 2374,
    'preserve_count': 1434, 'unnecessary_warp': 0,
    'undercorrected': 0, 'unnecessary_preserve_change': 0,
}


def _load(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _destination(prior, run):
    prior, run = prior.resolve(), run.resolve()
    if '.local' not in prior.parts or '.local' not in run.parts:
        raise ValueError('Both runs must be under .local')
    if run == prior or run in prior.parents or prior in run.parents:
        raise ValueError('New audit destination must be disjoint from frozen prior run')
    run.mkdir(parents=True, exist_ok=True)
    return run


def _write(run, name, value):
    target = run / name
    if target.suffix != '.json' or target.resolve().parent != run.resolve():
        raise ValueError('Audit JSON must be directly inside the new run')
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    return target


def _perfect_prediction(info, size):
    """Make a numerically realistic GT H8+delta; decision receives decoded q/r."""
    photo_size = info['photo_size']
    inverse = inverse_letterbox_matrix(photo_size, size)
    quad = np.asarray(info['quad_photo'], np.float64)
    in_quad = project(quad, np.linalg.inv(inverse))
    normal = project(in_quad, normalization(size, size))
    h = cv2.getPerspectiveTransform(UNIT.astype(np.float32), normal.astype(np.float32))
    delta = float(info['log_ratio']) - np.log(edge_ratio(quad))
    return np.r_[h.reshape(-1)[:8], delta]


def _decorate(row, info):
    kind = info['kind']
    row.update(condition=info.get('condition', 'noisy'),
               geometry=info.get('mode', 'front') if kind == 'photo' else 'preserve',
               kind=kind, mode=info.get('mode', 'front'),
               preserve_kind=kind if kind != 'photo' else None,
               front_subgroup=('rotated' if info.get('pure_roll') else 'upright')
               if kind == 'photo' and info.get('mode') == 'front' else None,
               occupancy=('30-50', '50-75', '75-90')[int(info['occupancy_bin'])]
               if kind == 'photo' else 'full_canvas',
               source_aspect=('portrait' if info['source_size'][0] / info['source_size'][1] < .8
                              else 'landscape' if info['source_size'][0] / info['source_size'][1] > 1.25
                              else 'near_square'),
               source_index=int(info['source_index']))
    return row


def _one(prediction, info, size, reference, planner, thresholds):
    photo_size = tuple(info['photo_size'])
    inverse = inverse_letterbox_matrix(photo_size, size)
    decoded = decode_prediction(prediction, size, photo_size, inverse)
    truth_input = project(np.asarray(info['quad_photo']), np.linalg.inv(inverse))
    row = measure_geometry(decoded, truth_input, float(info['log_ratio']), size)
    if decoded['valid']:
        plan = planner(photo_size, decoded['quad'], decoded['ratio'], thresholds)
    else:
        plan = {'action': 'needs_confirmation', 'matrix': None, 'output_size': photo_size,
                'photo_size': photo_size, 'valid': False, 'reason': ','.join(decoded['reasons']), 'metrics': {}}
    row.update(measure_final(plan, info['quad_photo'], info['source_size'],
                             info['source_to_photo'], reference))
    row['reason'] = plan['reason']
    return _decorate(row, info)


def _evaluate(predictions, metadata, size, references, thresholds, *, margin):
    old, new = [], []
    # The old trial-8 proxy thresholds (.01/.02) are deliberately not reused:
    # this policy's 9x9 max-grid/ratio thresholds are the approved GT crop gates.
    spatial_limits = {'margin_fraction': margin}
    for prediction, info, reference in zip(predictions, metadata, references, strict=True):
        old.append(_one(prediction, info, size, reference, frozen_plan, thresholds))
        new.append(_one(prediction, info, size, reference, spatial_plan, spatial_limits))
    return {'frozen': summarize_rows(old), 'spatial': summarize_rows(new)}


def _focused(summary):
    a = summary['all']
    return {key: a[key] for key in ('count', 'invalid', 'automatic_count', 'success_count',
                                   'automatic_success_rate', 'action_counts', 'corner_missing',
                                   'direction_error', 'warp_required_count', 'undercorrected',
                                   'undercorrected_rate', 'unnecessary_warp',
                                   'unnecessary_warp_rate', 'unnecessary_preserve_change')}


def verify_perfect_regression(report):
    """Fail closed against the frozen, full-D perfect-input spatial result.

    This is a local-data regression, not a CI fixture or a model admission
    result. The frozen D metadata digest binds the exact 5258-view corpus.
    """
    if report.get('scope') != 'D_perfect' or report.get('input_size') != 384:
        raise ValueError('perfect_regression_requires_D_perfect_384')
    if report.get('margin_fraction') != 0:
        raise ValueError('perfect_regression_requires_zero_margin')
    if report.get('views') != 5258:
        raise ValueError('perfect_regression_wrong_view_denominator')
    if report.get('prior_D_metadata_sha256') != PERFECT_D_METADATA_SHA256:
        raise ValueError('perfect_regression_D_metadata_SHA_mismatch')
    settings = report.get('settings', {})
    for setting in ('default', 'J384_trial8'):
        result = settings.get(setting, {})
        summary = result.get('summary', {}).get('spatial', {})
        all_rows = summary.get('all', {})
        for key, expected in PERFECT_D_REGRESSION.items():
            if all_rows.get(key) != expected:
                raise ValueError(f'perfect_regression_{setting}_{key}_mismatch')
        if result.get('focused', {}).get('spatial') != _focused(summary):
            raise ValueError(f'perfect_regression_{setting}_focused_mismatch')
        if all_rows.get('undercorrected_rate') != 0 or all_rows.get('unnecessary_warp_rate') != 0:
            raise ValueError(f'perfect_regression_{setting}_failure_rate_mismatch')
        if all_rows.get('automatic_success_rate') != 5038 / 5258:
            raise ValueError(f'perfect_regression_{setting}_success_denominator_mismatch')
        if sum(all_rows['action_counts'].values()) != 5258:
            raise ValueError(f'perfect_regression_{setting}_action_denominator_mismatch')
    return {'passed': True, 'metadata_sha256': PERFECT_D_METADATA_SHA256,
            'expectation': 'full_D_perfect_spatial_margin0_v1',
            'settings_checked': ['default', 'J384_trial8']}


def _run_set(prior, metadata, references, baseline, *, scope, margin):
    if scope == 'perfect':
        size = 384
        predictions = np.asarray([_perfect_prediction(info, size) for info in metadata])
        configurations = {'default': None,
                          'J384_trial8': _load(prior / 'final/D-calibration.json')['groups']['J384']['thresholds']}
    else:
        group = scope
        best = _load(prior / group / 'best.json')
        size = 384 if group == 'J384' else 224
        predictions_path = prior / group / f"predictions-{best['step']:04d}.npy"
        predictions = np.load(predictions_path, allow_pickle=False)
        if predictions.shape != (5258, 9):
            raise RuntimeError(f'{group} frozen D predictions have wrong shape')
        configurations = {'selected': _load(prior / 'final/D-calibration.json')['groups'][group]['thresholds']}
    result = {'scope': f'D_{scope}', 'input_size': size, 'views': len(metadata),
              'margin_fraction': margin, 'margin_calibrated': False,
              'prediction_sha256': _sha(predictions_path) if scope != 'perfect' else None,
              'settings': {}, 'GT_use': 'evaluation_and_perfect_input_only_not_spatial_decision'}
    for name, thresholds in configurations.items():
        values = _evaluate(predictions, metadata, size, references, thresholds, margin=margin)
        result['settings'][name] = {
            'thresholds': {'frozen': thresholds or FROZEN_DEFAULTS,
                           'spatial': {**SPATIAL_DEFAULTS, 'margin_fraction': margin}},
            'focused': {policy: _focused(summary) for policy, summary in values.items()},
            'gates': {policy: admission_gates(summary, baseline) for policy, summary in values.items()},
            'summary': values}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, required=True,
                        help='frozen unified-v1 run, read only')
    parser.add_argument('--run', type=Path, required=True,
                        help='new spatial-v2 .local run, write destination')
    parser.add_argument('--scope', choices=('perfect', 'models', 'all'), default='perfect')
    parser.add_argument('--margin-fraction', type=float, default=0.,
                        help='uncalibrated preset margin; default 0')
    parser.add_argument('--verify-perfect-regression', action='store_true',
                        help='require exact frozen 5258-view D perfect-input spatial regression')
    args = parser.parse_args(argv)
    if args.verify_perfect_regression and (args.scope == 'models' or args.margin_fraction != 0):
        parser.error('--verify-perfect-regression requires scope perfect/all and margin 0')
    prior = args.prior_run.resolve()
    run = _destination(prior, args.run)
    metadata_path = prior / 'cache/dev/metadata.json'
    metadata = _load(metadata_path)
    if len(metadata) != 5258:
        raise RuntimeError('Full D requires all 5258 frozen views')
    metadata_sha256 = _sha(metadata_path)
    if args.verify_perfect_regression and metadata_sha256 != PERFECT_D_METADATA_SHA256:
        raise RuntimeError('Frozen perfect-D metadata SHA mismatch')
    baseline = _load(prior / 'G0-D.json')
    references = [prepare_reference(row['photo_size'], row['source_size'],
                                    row['source_to_photo'], row['quad_photo'])
                  for row in metadata]
    scopes = (('perfect',) if args.scope == 'perfect' else
              GROUPS if args.scope == 'models' else ('perfect', *GROUPS))
    for scope in scopes:
        report = _run_set(prior, metadata, references, baseline,
                          scope=scope, margin=args.margin_fraction)
        report.update(prior_D_metadata_sha256=metadata_sha256,
                      prior_run_read_only=True, T_accessed=False)
        if args.verify_perfect_regression and scope == 'perfect':
            report['perfect_regression'] = verify_perfect_regression(report)
        path = _write(run, f'{scope}-D-policy-audit.local.json', report)
        print(json.dumps({'scope': scope, 'saved': str(path),
                          'counts': {setting: data['focused'] for setting, data in report['settings'].items()}},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
