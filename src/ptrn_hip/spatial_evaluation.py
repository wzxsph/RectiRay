"""Four-corner prediction bridge to the frozen final-output evaluator.

All rows, including invalid predictions and confirmations, remain in the
denominator. Oracle-ratio variants are explicitly diagnostic and must never be
reported as end-to-end model results. No ground truth enters action selection.
"""
from __future__ import annotations

import math

import numpy as np

from .ratio_synthesis import project
from .spatial_geometry import plan_transform
from .unified_evaluation import (FAILURE, GEOMETRY_METRICS, _failure_final,
                                 measure_final, measure_geometry,
                                 prepare_reference, summarize_rows)
from .unified_geometry import edge_ratio, inverse_letterbox_matrix, quad_reasons

RATIO_METHODS = frozenset({'prediction', 'edge', 'oracle_gt_diagnostic'})
MARGINS = (0., .0025, .005, .01)


def _truth_input(info, inverse):
    """Derive truth at requested size; cached quad_input may have another size."""
    return project(np.asarray(info['quad_photo'], np.float64), np.linalg.inv(inverse))


def _strata(info):
    kind = info.get('kind')
    size = info.get('source_size')
    row = {'condition': info.get('condition', 'noisy'), 'kind': kind,
           'mode': info.get('mode', 'front'),
           'geometry': info.get('mode', 'front') if kind == 'photo' else 'preserve',
           'preserve_kind': kind if kind != 'photo' else None,
           'front_subgroup': ('rotated' if info.get('pure_roll') else 'upright')
           if kind == 'photo' and info.get('mode') == 'front' else None,
           'occupancy': ('30-50', '50-75', '75-90')[int(info['occupancy_bin'])]
           if kind == 'photo' and info.get('occupancy_bin') is not None else 'full_canvas'}
    if size is not None:
        aspect = size[0] / size[1]
        row['source_aspect'] = 'portrait' if aspect < .8 else ('landscape' if aspect > 1.25 else 'near_square')
    if info.get('source_index') is not None:
        row['source_index'] = int(info['source_index'])
    if info.get('split') is not None:
        row['split'] = info['split']
    return row


def _decode(quad_input, supplied_log_ratio, photo_size, inverse, ratio_method, gt_log_ratio):
    decoded = {'quad_input': np.asarray(quad_input, np.float64), 'quad': None,
               'log_ratio': None, 'ratio': None, 'r_edge': None,
               'valid': False, 'reasons': []}
    try:
        if decoded['quad_input'].shape != (4, 2) or not np.isfinite(decoded['quad_input']).all():
            raise ValueError('invalid_coordinates')
        quad = project(decoded['quad_input'], inverse)
        edge = edge_ratio(quad)
        if ratio_method == 'edge':
            log_ratio = math.log(edge)
        elif ratio_method == 'oracle_gt_diagnostic':
            log_ratio = float(gt_log_ratio)
        else:
            log_ratio = float(supplied_log_ratio)
        decoded.update(quad=quad, r_edge=edge, log_ratio=log_ratio)
        if not np.isfinite(log_ratio) or abs(log_ratio) > 30 or not .01 <= math.exp(log_ratio) <= 100:
            raise ValueError('invalid_ratio')
        decoded['ratio'] = math.exp(log_ratio)
        decoded['reasons'] = quad_reasons(quad, photo_size)
        decoded['valid'] = not decoded['reasons']
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        decoded['reasons'].append(str(error))
    return decoded


def evaluate_quad_predictions(quad_input, log_ratios, metadata, size, thresholds=None, *,
                              ratio_method='prediction', margin_fraction=0., references=None,
                              source_kind='unknown'):
    """Return ``(legacy_compatible_summary, rows)`` for four-corner inputs.

    ``quad_input`` is [N,4,2] TL/TR/BR/BL in size-specific pixel-centre
    coordinates. ``log_ratios`` is the *full* log(W/H), not a delta; it may be
    None for edge and oracle diagnostics. ``metadata`` supplies photo_size,
    quad_photo, log_ratio and source geometry. The action policy sees only the
    predicted photo quad, selected ratio and provenance; ground truth is used
    only by the legacy scorer after the action is fixed.
    """
    if ratio_method not in RATIO_METHODS:
        raise ValueError('unknown_ratio_method')
    if not np.isfinite(margin_fraction) or not 0 <= margin_fraction < .25:
        raise ValueError('invalid_uncalibrated_margin')
    quad_input = np.asarray(quad_input, np.float64)
    metadata = list(metadata)
    if quad_input.shape != (len(metadata), 4, 2):
        raise ValueError('quad_input_must_be_N_4_2')
    if ratio_method == 'prediction':
        log_ratios = np.asarray(log_ratios, np.float64)
        if log_ratios.shape != (len(metadata),):
            raise ValueError('log_ratios_must_be_N_full_log_ratios')
    elif log_ratios is not None and np.asarray(log_ratios).shape != (len(metadata),):
        raise ValueError('log_ratios_length_mismatch')
    if references is not None and len(references) != len(metadata):
        raise ValueError('references_length_mismatch')
    policy_thresholds = dict(thresholds or {})
    policy_thresholds['margin_fraction'] = float(margin_fraction)
    rows = []
    for index, (q_input, info) in enumerate(zip(quad_input, metadata, strict=True)):
        photo_size = tuple(info['photo_size'])
        inverse = inverse_letterbox_matrix(photo_size, size)
        truth_photo = np.asarray(info['quad_photo'], np.float64)
        gt_log_ratio = float(info['log_ratio'])
        supplied = None if log_ratios is None else log_ratios[index]
        decoded = _decode(q_input, supplied, photo_size, inverse, ratio_method, gt_log_ratio)
        try:
            geometry = measure_geometry(decoded, _truth_input(info, inverse), gt_log_ratio, size)
        except (ValueError, TypeError, OverflowError, FloatingPointError):
            geometry = {'invalid': 1, **{key: FAILURE for key in GEOMETRY_METRICS}}
        row = {**geometry, **_strata(info), 'ratio_method': ratio_method,
               'ratio_uses_gt': ratio_method == 'oracle_gt_diagnostic',
               'margin_fraction': float(margin_fraction), 'margin_calibrated': False}
        if decoded['quad'] is not None and decoded['ratio'] is not None:
            plan = plan_transform(photo_size, decoded['quad'], decoded['ratio'],
                                  policy_thresholds, source_kind)
        else:
            plan = {'action': 'needs_confirmation', 'matrix': None, 'output_size': photo_size,
                    'valid': False, 'reason': ','.join(decoded['reasons']), 'metrics': {}}
        if info.get('source_size') is None or info.get('source_to_photo') is None:
            row.update(_failure_final(), final_reason='missing_GT_source_geometry')
        else:
            reference = references[index] if references is not None else prepare_reference(
                photo_size, info['source_size'], info['source_to_photo'], truth_photo)
            row.update(measure_final(plan, truth_photo, info['source_size'],
                                     info['source_to_photo'], reference))
        row['reason'] = plan['reason']
        row['transform_metrics'] = plan['metrics']
        rows.append(row)
    summary = summarize_rows(rows)
    summary['evaluation_method'] = {'corner_method': 'predicted_quad',
                                    'ratio_method': ratio_method,
                                    'ratio_uses_gt': ratio_method == 'oracle_gt_diagnostic',
                                    'margin_fraction': float(margin_fraction),
                                    'margin_calibrated': False,
                                    'include_invalid': True,
                                    'scope': 'synthetic_geometry_only'}
    return summary, rows


def evaluate_ratio_margin_grid(predicted_quad, predicted_log_ratios, metadata, size,
                               thresholds=None, *, margins=MARGINS, references=None,
                               split='dev'):
    """D-only, predeclared four-way ratio/corner counterfactual and margin grid.

    GT variants are diagnostics, never a model score. This helper refuses test
    selection; evaluate_quad_predictions itself may score a frozen test set.
    """
    metadata = list(metadata)
    if split != 'dev' or any(info.get('split', 'dev') not in ('dev', 'development') for info in metadata):
        raise ValueError('margin_grid_is_development_only')
    predicted_quad = np.asarray(predicted_quad, np.float64)
    gt_quad = np.stack([_truth_input(info, inverse_letterbox_matrix(info['photo_size'], size))
                        for info in metadata])
    variants = {
        'predicted_quad_predicted_ratio': (predicted_quad, predicted_log_ratios, 'prediction'),
        'predicted_quad_edge_ratio': (predicted_quad, None, 'edge'),
        'predicted_quad_GT_ratio_diagnostic': (predicted_quad, None, 'oracle_gt_diagnostic'),
        'GT_quad_GT_ratio_perfect_diagnostic': (gt_quad, None, 'oracle_gt_diagnostic'),
    }
    result = {}
    for name, (q, ratios, method) in variants.items():
        result[name] = {}
        for margin in margins:
            summary, _ = evaluate_quad_predictions(
                q, ratios, metadata, size, thresholds, ratio_method=method,
                margin_fraction=margin, references=references)
            summary['evaluation_method']['corner_method'] = ('GT_quad_diagnostic' if name.startswith('GT_')
                                                               else 'predicted_quad')
            result[name][str(margin)] = summary
    return result
