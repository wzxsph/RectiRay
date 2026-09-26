"""Source-coordinate and actual-output acceptance for unified PTRN geometry.

All invalid/abstained cases stay in the total denominator. Similarity alignment
never removes anisotropic distortion, clipping, reflection or direction failures.
No source paths, patient identifiers or image content are persisted by this API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import cv2
import numpy as np

from .ratio_synthesis import pixel_corners, project
from .unified_geometry import (decode_prediction, edge_ratio, plan_transform,
                               similarity_crop, fit_rotation_rectangle,
                               inverse_letterbox_matrix)

FAILURE = 1e6
GEOMETRY_METRICS = ('corner_nme', 'ratio_error', 'edge_ratio_error')
FINAL_METRICS = ('grid_error_mean', 'grid_error_p90', 'grid_error_max',
                 'final_ratio_error', 'lost_area', 'background_fraction', 'orientation_error_deg')


def measure_geometry(decoded, truth_quad, log_ratio, size):
    result = {'invalid': int(not decoded['valid']), **{key: FAILURE for key in GEOMETRY_METRICS}}
    try:
        q = np.asarray(decoded['quad_input'], np.float64)
        result['corner_nme'] = float(np.linalg.norm(q - np.asarray(truth_quad), axis=1).mean() / np.hypot(size - 1, size - 1))
        result['ratio_error'] = float(np.expm1(abs(decoded['log_ratio'] - log_ratio)))
        result['edge_ratio_error'] = float(np.expm1(abs(np.log(decoded['r_edge']) - log_ratio)))
        if not all(np.isfinite(result[key]) for key in GEOMETRY_METRICS):
            raise ValueError('nonfinite_prediction_metric')
    except (ValueError, TypeError, FloatingPointError, OverflowError):
        result.update(invalid=1, **{key: FAILURE for key in GEOMETRY_METRICS})
    return result


def _failure_final(action='needs_confirmation'):
    return {'action': action, 'automatic': False, 'success': False, 'final_invalid': 1,
            **{key: FAILURE for key in FINAL_METRICS}, 'corner_missing': 1,
            'direction_error': 1, 'grid_count': 81, 'grid_inside_count': 0}


def _signed_area(points):
    # Centre first to avoid cancellation for polygons translated far from zero.
    points = points - points.mean(axis=0)
    return float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1)) / 2)


def _intersection(a, b):
    """Float64 Sutherland-Hodgman clipping of two convex polygons.

    OpenCV's float32 intersection can report half the area for coincident
    rectangles when one boundary differs by 1e-12 px. Analytic clipping retains
    subpixel geometry, with only a scale-relative machine-precision tolerance;
    it does not relax the 0.5% missing-area or 5% background acceptance gates.
    Either winding is accepted here; output direction is checked separately.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    for polygon in (a, b):
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all()):
            raise ValueError('invalid_intersection_polygon')
        edges = np.roll(polygon, -1, axis=0) - polygon
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        tolerance = 64 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(edges)))) ** 2
        if (not np.isfinite(cross).all() or abs(_signed_area(polygon)) <= tolerance
                or not ((cross >= -tolerance).all() or (cross <= tolerance).all())):
            raise ValueError('nonconvex_intersection')
    if _signed_area(b) < 0:
        b = b[::-1]
    scale = max(1., float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    coordinate_tolerance = 64 * np.finfo(np.float64).eps * scale
    points = a
    for start, end in zip(b, np.roll(b, -1, axis=0), strict=True):
        if not len(points):
            return 0.
        edge = end - start
        cross_tolerance = coordinate_tolerance * float(np.linalg.norm(edge))
        def distance(point):
            return float(edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0]))
        previous = points[-1]
        previous_distance = distance(previous)
        previous_inside = previous_distance >= -cross_tolerance
        output = []
        for current in points:
            current_distance = distance(current)
            current_inside = current_distance >= -cross_tolerance
            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                if denominator == 0:
                    raise ValueError('degenerate_intersection_segment')
                fraction = previous_distance / denominator
                # A tolerant endpoint can be infinitesimally outside the line.
                # Keep that endpoint instead of extrapolating beyond the segment.
                crossing = (previous if fraction <= 0 else current if fraction >= 1 else
                            previous + fraction * (current - previous))
                output.append(crossing)
            if current_inside:
                output.append(current)
            previous, previous_distance, previous_inside = current, current_distance, current_inside
        points = np.asarray(output, np.float64)
    if len(points) < 3:
        return 0.
    area = abs(_signed_area(points))
    if not np.isfinite(area):
        raise ValueError('nonfinite_intersection_area')
    return area


def _final_raw(plan, truth_quad_photo, source_size, source_to_photo, reference=None):
    if plan['action'] == 'needs_confirmation' or not plan['valid'] or plan['matrix'] is None:
        return _failure_final(plan['action'])
    result = _failure_final(plan['action'])
    result['automatic'] = True
    try:
        source_width, source_height = map(float, source_size)
        output_size = tuple(map(int, plan['output_size']))
        if min(*source_size, *output_size) < 2:
            raise ValueError('invalid_image_size')
        if reference is None:
            x, y = np.meshgrid(np.linspace(0, source_width - 1, 9), np.linspace(0, source_height - 1, 9))
            grid = np.c_[x.ravel(), y.ravel()]
            gt_scale = 1024 / max(source_width - 1, source_height - 1)
            target = (grid - np.array([source_width - 1, source_height - 1]) / 2) * gt_scale
            diagonal = np.hypot(source_width - 1, source_height - 1) * gt_scale
            photo_grid = project(grid, source_to_photo)
        else:
            target, diagonal, photo_grid = reference['target_grid'], reference['gt_diagonal'], reference['photo_grid']
        actual = project(photo_grid, np.asarray(plan['matrix']))
        actual_corners = project(np.asarray(truth_quad_photo), np.asarray(plan['matrix']))
        output_bounds = np.asarray(output_size, np.float64) - 1
        # Both grids use a common 1024 long-edge coordinate convention first.
        observed = (actual - output_bounds / 2) * (1024 / max(output_bounds))
        ac, tc = observed - observed.mean(axis=0), target - target.mean(axis=0)
        u, singular, vt = np.linalg.svd(ac.T @ tc)
        signs = np.ones(2)
        if np.linalg.det(u @ vt) < 0:
            signs[-1] = -1
        rotation = (u * signs) @ vt
        energy = float(np.sum(ac * ac))
        if energy < 1e-12:
            raise ValueError('collapsed_output_grid')
        scale = float(np.sum(singular * signs) / energy)
        aligned = ac @ rotation * scale + target.mean(axis=0)
        errors = np.linalg.norm(aligned - target, axis=1) / diagonal
        source_quad = pixel_corners(*source_size)
        composite = np.asarray(plan['matrix']) @ np.asarray(source_to_photo)
        output_source = project(pixel_corners(*output_size), np.linalg.inv(composite))
        source_area = (source_width - 1) * (source_height - 1)
        source_covered = _intersection(source_quad, output_source)
        lost = max(0., 1 - min(1., source_covered / source_area))
        output_area = float(np.prod(output_bounds))
        film_output = _intersection(actual_corners, pixel_corners(*output_size))
        background = max(0., 1 - min(1., film_output / output_area))
        # Half a pixel tolerates centre-rounding without hiding a missing strip.
        inside = ((actual >= -.5) & (actual <= output_bounds + .5)).all(axis=1)
        corners_inside = ((actual_corners >= -.5) & (actual_corners <= output_bounds + .5)).all(axis=1)
        signed_area = cv2.contourArea(actual_corners.astype(np.float32), oriented=True)
        orientation = abs(fit_rotation_rectangle(actual_corners)['angle_deg']) if signed_area > 0 else 180.
        direction_error = int(signed_area <= 0 or orientation > .5 + 1e-7)
        final_ratio = float(np.expm1(abs(np.log(edge_ratio(actual_corners) / ((source_width - 1) / (source_height - 1))))))
        result.update(grid_error_mean=float(errors.mean()), grid_error_p90=float(np.percentile(errors, 90)),
                      grid_error_max=float(errors.max()), final_ratio_error=final_ratio,
                      lost_area=lost, background_fraction=background, orientation_error_deg=orientation,
                      corner_missing=int(not corners_inside.all()), direction_error=direction_error,
                      grid_count=81, grid_inside_count=int(inside.sum()), final_invalid=0)
        # Approved final-output gate: P90 grid residual and content integrity.
        # The 1% ratio bound belongs to the GT crop oracle, not a new output gate.
        result['success'] = bool(np.percentile(errors, 90) <= .01 and lost <= .005
                                 and background <= .05 and corners_inside.all() and not direction_error)
    except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError, FloatingPointError, OverflowError):
        pass
    return result


def prepare_reference(photo_size, source_size, source_to_photo, truth_quad_photo=None):
    """Cache GT-only geometry; never call with predicted corners as truth."""
    source_quad = pixel_corners(*source_size)
    matrix = np.asarray(source_to_photo, np.float64)
    q = project(source_quad, matrix) if truth_quad_photo is None else np.asarray(truth_quad_photo, np.float64)
    # Reject contradictory geometry instead of selecting one truth silently.
    if not np.allclose(q, project(source_quad, matrix), atol=2e-3, rtol=1e-6):
        raise ValueError('GT_quad_and_source_projection_disagree')
    x, y = np.meshgrid(np.linspace(0, source_size[0] - 1, 9), np.linspace(0, source_size[1] - 1, 9))
    grid = np.c_[x.ravel(), y.ravel()]
    spans = np.asarray(source_size, np.float64) - 1
    gt_scale = 1024 / max(spans)
    reference = {'truth_quad_photo': q, 'source_size': tuple(source_size), 'photo_size': tuple(photo_size),
                 'source_to_photo': matrix, 'photo_grid': project(grid, matrix),
                 'target_grid': (grid - spans / 2) * gt_scale,
                 'gt_diagonal': float(np.linalg.norm(spans) * gt_scale)}
    crop_matrix, output_size, _ = similarity_crop(photo_size, q)
    oracle_plan = {'action': 'crop', 'matrix': crop_matrix, 'output_size': output_size, 'valid': True}
    final = _final_raw(oracle_plan, q, source_size, matrix, reference)
    allowed = (not final['final_invalid'] and final['grid_error_max'] <= .005
               and final['final_ratio_error'] <= .01 and final['lost_area'] <= .005
               and not final['corner_missing'] and not final['direction_error'])
    keep_plan = {'action': 'keep', 'matrix': np.eye(3), 'output_size': tuple(photo_size), 'valid': True}
    keep_result = _final_raw(keep_plan, q, source_size, matrix, reference)
    is_full_canvas = bool(np.allclose(q, pixel_corners(*photo_size), atol=2e-3, rtol=0))
    preserve = bool(is_full_canvas and keep_result['grid_error_max'] <= .005
                    and keep_result['final_ratio_error'] <= .01 and not keep_result['direction_error'])
    reference.update(oracle_action='keep' if preserve else ('crop' if allowed else 'warp'),
                     oracle_crop_acceptable=bool(allowed), oracle_crop_metrics=final,
                     preserve_input=preserve)
    return reference


def measure_final(plan, truth_quad_photo, source_size, source_to_photo, reference=None):
    reference = reference or prepare_reference(plan.get('photo_size', plan['output_size']), source_size, source_to_photo, truth_quad_photo)
    result = _final_raw(plan, truth_quad_photo, source_size, source_to_photo, reference)
    oracle = reference['oracle_action']
    result.update(oracle_action=oracle, preserve_input=reference['preserve_input'],
                  oracle_crop_acceptable=reference['oracle_crop_acceptable'])
    auto = result['automatic']
    result['unnecessary_warp'] = int(auto and reference['oracle_crop_acceptable'] and plan['action'] == 'warp')
    result['undercorrected'] = int(auto and oracle == 'warp' and plan['action'] in ('keep', 'crop')
                                   and (result['grid_error_max'] > .005 or result['final_ratio_error'] > .01))
    result['unnecessary_preserve_change'] = int(auto and reference['preserve_input'] and plan['action'] != 'keep')
    return result


def measure(prediction, truth_quad, log_ratio, size, photo_size=None, source_size=None,
            source_to_photo=None, inverse_letterbox=None, thresholds=None, source_kind='unknown',
            confirmed=False, reference=None, **metadata):
    """Evaluate prediction and actual transform; metadata is explicit stratum data.

    Required for final-output metrics: original source_size and source_to_photo.
    No fabricated source geometry is inferred from a model prediction.
    """
    photo_size = tuple(photo_size or (size, size))
    inverse = inverse_letterbox_matrix(photo_size, size) if inverse_letterbox is None else np.asarray(inverse_letterbox)
    decoded = decode_prediction(prediction, size, photo_size, inverse)
    row = measure_geometry(decoded, truth_quad, log_ratio, size)
    row.update({key: value for key, value in metadata.items()
                if key in ('condition', 'geometry', 'mode', 'kind', 'occupancy', 'source_aspect', 'front_subgroup', 'preserve_kind', 'split')})
    if 'geometry' not in row and 'mode' in row:
        row['geometry'] = row['mode']
    if row.get('kind') in ('digital', 'appearance', 'rectified'):
        row['preserve_kind'] = row['kind']
        row['geometry'] = 'preserve'
        row.pop('front_subgroup', None)
    if 'source_index' in metadata:
        # Ephemeral ordinal enables source-grouped uncertainty; summarize never emits it.
        row['source_index'] = int(metadata['source_index'])
    if 'source_aspect' not in row and source_size is not None:
        aspect = source_size[0] / source_size[1]
        row['source_aspect'] = 'portrait' if aspect < .8 else ('landscape' if aspect > 1.25 else 'near_square')
    if source_size is None or source_to_photo is None:
        row.update(_failure_final(), final_reason='missing_GT_source_geometry')
        return row
    truth_photo = project(np.asarray(truth_quad), inverse)
    reference = reference or prepare_reference(photo_size, source_size, source_to_photo, truth_photo)
    if decoded['valid'] or confirmed or source_kind in ('reliable_digital', 'verified_digital', 'reliable_dicom', 'verified_dicom', 'confirmed_working', 'already_rectified'):
        plan = plan_transform(photo_size, decoded['quad'], decoded['ratio'], thresholds, source_kind, confirmed)
    else:
        plan = {'action': 'needs_confirmation', 'matrix': None, 'output_size': photo_size, 'valid': False,
                'reason': ','.join(decoded['reasons']), 'metrics': {}}
    row.update(measure_final(plan, truth_photo, source_size, source_to_photo, reference))
    row['reason'] = plan['reason']
    row['transform_metrics'] = plan['metrics']
    return row


def _stats(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {'count': 0, 'mean': None, 'median': None, 'p90': None, 'max': None}
    if not np.isfinite(values).all():
        raise FloatingPointError('nonfinite_evaluation_metric')
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'max': float(values.max())}


def _summarize(rows):
    n = len(rows)
    automatic = [r for r in rows if r.get('automatic', False)]
    crop_allowed = [r for r in rows if r.get('oracle_crop_acceptable', False)]
    warp_required = [r for r in rows if r.get('oracle_action') == 'warp']
    preserve = [r for r in rows if r.get('preserve_input', False)]
    result = {'count': n, 'source_count': len({r['source_index'] for r in rows if 'source_index' in r}),
              'invalid': sum(r.get('invalid', 1) for r in rows),
              'valid_count': sum(not r.get('invalid', 1) for r in rows),
              'action_counts': dict(Counter(r.get('action', 'needs_confirmation') for r in rows)),
              'automatic_count': len(automatic), 'success_count': sum(r.get('success', False) for r in rows),
              'automatic_rate': len(automatic) / n if n else 0.,
              'automatic_success_rate': sum(r.get('success', False) for r in rows) / n if n else 0.,
              'final_invalid': sum(r.get('final_invalid', 1) for r in automatic),
              'corner_missing': sum(r.get('corner_missing', 1) for r in automatic),
              'direction_error': sum(r.get('direction_error', 1) for r in automatic),
              'crop_allowed_count': len(crop_allowed), 'warp_required_count': len(warp_required), 'preserve_count': len(preserve)}
    for key, subset in [('unnecessary_warp', crop_allowed), ('undercorrected', warp_required), ('unnecessary_preserve_change', preserve)]:
        result[key] = sum(r.get(key, 0) for r in subset)
        result[key + '_rate'] = result[key] / len(subset) if subset else None
    # Prediction aggregates retain failed cases and also expose valid-only values.
    for key in GEOMETRY_METRICS:
        result[key] = _stats([r.get(key, FAILURE) for r in rows])
    result['valid_geometry'] = {key: _stats([r.get(key, FAILURE) for r in rows if not r.get('invalid', 1)]) for key in GEOMETRY_METRICS}
    # Final-output statistics describe attempted automatic outputs, with explicit n.
    for key in FINAL_METRICS:
        result[key] = _stats([r.get(key, FAILURE) for r in automatic])
    return result


def summarize_rows(rows):
    """Stratified aggregates; condition_geometry keys are 'clear|mild', etc."""
    rows = list(rows)
    result = {'all': _summarize(rows)}
    for field in ('condition', 'geometry', 'kind', 'occupancy', 'source_aspect', 'action', 'front_subgroup', 'preserve_kind'):
        grouped = defaultdict(list)
        for row in rows:
            if row.get(field) is not None:
                grouped[str(row[field])].append(row)
        result[field] = {key: _summarize(value) for key, value in sorted(grouped.items())}
    grouped = defaultdict(list)
    for row in rows:
        if row.get('condition') is not None and row.get('geometry') is not None:
            grouped[f"{row['condition']}|{row['geometry']}"].append(row)
    result['condition_geometry'] = {key: _summarize(value) for key, value in sorted(grouped.items())}
    return result


def _number(summary, metric, statistic, default=FAILURE):
    value = summary.get(metric, {}).get(statistic)
    return default if value is None else value


def _set_gates(candidate, baseline=None):
    checks = {}
    overall = candidate['all']
    checks['automatic_completion'] = overall['automatic_success_rate'] >= .90
    strata = candidate.get('condition_geometry', {})
    # Require both clear/clean and noisy mild groups; aliases match old protocols.
    mild = {key: value for key, value in strata.items() if key.split('|')[-1] == 'mild'}
    conditions = {key.split('|')[0] for key in mild}
    checks['mild_strata_present'] = ('clear' in conditions or 'clean' in conditions) and 'noisy' in conditions
    for key, value in mild.items():
        checks[f'ratio:{key}'] = (_number(value, 'ratio_error', 'median') <= .03
                                  and _number(value, 'ratio_error', 'p90') <= .10
                                  and _number(value, 'ratio_error', 'median') <= .8 * _number(value, 'edge_ratio_error', 'median'))
    clean_name = 'clear' if any(key.startswith('clear|') for key in strata) else 'clean'
    checks['evaluation_strata_present'] = all(f'{condition}|{mode}' in strata
        for condition in (clean_name, 'noisy') for mode in ('front', 'mild', 'medium', 'hard'))
    for key, value in strata.items():
        if value['count'] == 0:
            checks[f'nonempty:{key}'] = False
            continue
        if key.split('|')[-1] not in ('preserve', 'original', 'noise_original', 'rectified'):
            checks[f'final_grid:{key}'] = value['automatic_count'] > 0 and _number(value, 'grid_error_p90', 'p90') <= .01
        checks[f'content:{key}'] = (value['final_invalid'] == 0 and value['corner_missing'] == 0 and value['direction_error'] == 0
                                  and _number(value, 'lost_area', 'max') <= .005
                                  and _number(value, 'background_fraction', 'p90') <= .05)
        if baseline is not None:
            reference = baseline.get('condition_geometry', {}).get(key)
            checks[f'corner:{key}'] = bool(reference is not None and
                _number(value, 'corner_nme', 'mean') <= 1.05 * _number(reference, 'corner_nme', 'mean') + 1e-12 and
                _number(value, 'corner_nme', 'p90') <= 1.10 * _number(reference, 'corner_nme', 'p90') + 1e-12 and
                value['invalid'] <= reference['invalid'])
    checks['baseline_present'] = baseline is not None
    for field, limit in [('unnecessary_warp_rate', .02), ('undercorrected_rate', .02), ('unnecessary_preserve_change_rate', .01)]:
        value = overall.get(field)
        checks[field] = (value is None and field == 'undercorrected_rate' and overall['warp_required_count'] == 0) or (value is not None and value <= limit)
    preserve_groups = candidate.get('preserve_kind', {})
    for key, value in preserve_groups.items():
        rate = value.get('unnecessary_preserve_change_rate')
        checks[f'preserve:{key}'] = rate is not None and rate <= .01
    front = candidate.get('front_subgroup', {})
    checks['front_subgroups_present'] = 'upright' in front and ('rotated' in front or 'rotation' in front)
    for key, value in front.items():
        checks[f'front:{key}'] = value['automatic_success_rate'] >= .95
    return {'eligible': bool(checks) and all(checks.values()), 'checks': checks,
            'failed': [key for key, value in checks.items() if not value]}


def _regression_gates(candidate, baseline):
    checks = {'baseline_present': baseline is not None}
    for key, value in candidate.get('condition_geometry', {}).items():
        reference = (baseline or {}).get('condition_geometry', {}).get(key)
        checks[f'corner:{key}'] = bool(reference is not None and
            _number(value, 'corner_nme', 'mean') <= 1.05 * _number(reference, 'corner_nme', 'mean') + 1e-12 and
            _number(value, 'corner_nme', 'p90') <= 1.10 * _number(reference, 'corner_nme', 'p90') + 1e-12 and
            value['invalid'] <= reference['invalid'])
    expected = {f'{condition}|{mode}' for condition in ('clean', 'noisy') for mode in ('identity', 'mild', 'hard')}
    actual = candidate.get('condition_geometry', {})
    checks['regression_strata_present'] = expected.issubset(actual) and all(actual[key]['count'] == 478 for key in expected)
    baseline_strata = (baseline or {}).get('condition_geometry', {})
    checks['baseline_strata_complete'] = expected.issubset(baseline_strata) and all(baseline_strata[key]['count'] == 478 for key in expected)
    return {'eligible': all(checks.values()), 'checks': checks,
            'failed': [key for key, value in checks.items() if not value]}


def admission_gates(candidate, baseline=None, **external_checks):
    """Evaluate each D/T/R stratum; absent required evidence cannot pass.

    For one set pass summarize_rows output. For multiple sets pass dictionaries
    keyed D/T (optional R); baseline must have matching structure. Data, real-photo
    review, export and CPU gates remain explicit external evidence supplied by caller.
    """
    if 'all' in candidate:
        result = _set_gates(candidate, baseline)
    else:
        sets = {key: (_regression_gates(value, (baseline or {}).get(key)) if key == 'R' else
                     _set_gates(value, (baseline or {}).get(key))) for key, value in candidate.items()}
        result = {'sets': sets, 'eligible': bool(sets) and all(value['eligible'] for value in sets.values()),
                  'failed': [f'{key}:{failure}' for key, value in sets.items() for failure in value['failed']]}
    if external_checks:
        result['external_checks'] = {key: bool(value) for key, value in external_checks.items()}
        result['eligible'] = result['eligible'] and all(result['external_checks'].values())
        result['failed'].extend(key for key, value in result['external_checks'].items() if not value)
    result['scope'] = 'synthetic_geometry_only_unless_external_evidence_supplied'
    return result


def selection_score(summary, baseline=None):
    """Lower is better: protect corners/preserve first, then mild tail and median.

    Append update number in the caller to resolve ties toward earlier checkpoints.
    This research score does not confer admission when no candidate passes gates.
    """
    all_rows = summary['all']
    mild = [value for key, value in summary.get('condition_geometry', {}).items() if key.endswith('|mild')]
    protection_failures = all_rows['invalid'] + all_rows.get('unnecessary_preserve_change', 0)
    if baseline:
        for key, value in summary.get('condition_geometry', {}).items():
            reference = baseline.get('condition_geometry', {}).get(key)
            if reference is not None:
                protection_failures += int(_number(value, 'corner_nme', 'mean') > 1.05 * _number(reference, 'corner_nme', 'mean') + 1e-12)
                protection_failures += int(_number(value, 'corner_nme', 'p90') > 1.10 * _number(reference, 'corner_nme', 'p90') + 1e-12)
    return (protection_failures, max((_number(value, 'ratio_error', 'p90') for value in mild), default=FAILURE),
            max((_number(value, 'ratio_error', 'median') for value in mild), default=FAILURE))
