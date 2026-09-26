import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import UNIT, project, normalization, pixel_corners
from ptrn_hip.unified_geometry import inverse_letterbox_matrix, edge_ratio
from ptrn_hip.unified_evaluation import (prepare_reference, measure, measure_final,
    summarize_rows, admission_gates, selection_score)


def setup_case(source_size=(401, 301), photo_size=(700, 600), quad=None):
    q = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], float) if quad is None else np.asarray(quad, float)
    matrix = cv2.getPerspectiveTransform(pixel_corners(*source_size).astype(np.float32), q.astype(np.float32))
    lb = np.linalg.inv(inverse_letterbox_matrix(photo_size, 224))
    q_input = project(q, lb)
    h = cv2.getPerspectiveTransform(UNIT.astype(np.float32), project(q_input, normalization(224, 224)).astype(np.float32))
    ratio = source_size[0] / source_size[1]
    p = np.r_[h.ravel()[:8], np.log(ratio / edge_ratio(q))]
    return q, q_input, matrix, p


def test_perfect_crop_uses_source_pixels_and_accepts_without_resampling_gt():
    q, qi, matrix, p = setup_case()
    reference = prepare_reference((700, 600), (401, 301), matrix, q)
    row = measure(p, qi, np.log(401 / 301), 224, photo_size=(700, 600),
                  source_size=(401, 301), source_to_photo=matrix, reference=reference)
    assert row['action'] == 'crop'
    assert row['oracle_action'] == 'crop'
    assert row['success']
    assert row['grid_count'] == row['grid_inside_count'] == 81
    assert row['grid_error_max'] < 1e-7
    assert row['lost_area'] < 1e-6


def test_grid_alignment_does_not_hide_wrong_aspect_ratio():
    q, _, matrix, _ = setup_case((401, 401), (700, 700), [[100, 100], [500, 100], [500, 500], [100, 500]])
    reference = prepare_reference((700, 700), (401, 401), matrix, q)
    plan = {'action': 'warp', 'valid': True, 'matrix': np.array([[2., 0, -200], [0, 1, -100], [0, 0, 1]]), 'output_size': (801, 401)}
    row = measure_final(plan, q, (401, 401), matrix, reference)
    assert row['grid_error_max'] > .1
    assert row['final_ratio_error'] == pytest.approx(1)
    assert not row['success']


def test_similarity_alignment_cannot_excuse_upside_down_output():
    q, _, matrix, _ = setup_case()
    reference = prepare_reference((700, 600), (401, 301), matrix, q)
    plan = {'action': 'warp', 'valid': True, 'matrix': np.array([[-1., 0, 500], [0, -1, 400], [0, 0, 1]]), 'output_size': (401, 301)}
    row = measure_final(plan, q, (401, 301), matrix, reference)
    assert row['grid_error_max'] < 1e-10
    assert row['direction_error']
    assert not row['success']


def test_missing_edges_are_failures_even_with_perfect_internal_grid_shape():
    q, _, matrix, _ = setup_case()
    reference = prepare_reference((700, 600), (401, 301), matrix, q)
    plan = {'action': 'crop', 'valid': True, 'matrix': np.array([[1., 0, -110], [0, 1, -100], [0, 0, 1]]), 'output_size': (381, 301)}
    row = measure_final(plan, q, (401, 301), matrix, reference)
    assert row['grid_error_max'] < 1e-10
    assert row['lost_area'] == pytest.approx(.05)
    assert row['corner_missing']
    assert row['grid_count'] == 81 and row['grid_inside_count'] < 81
    assert not row['success']


def test_gt_affine_compression_requires_warp_even_though_quad_is_rectangle():
    q, _, matrix, _ = setup_case((401, 401), (700, 600), [[100, 100], [300, 100], [300, 500], [100, 500]])
    reference = prepare_reference((700, 600), (401, 401), matrix, q)
    assert reference['oracle_action'] == 'warp'
    assert reference['oracle_crop_metrics']['final_ratio_error'] > .9


def test_invalid_and_abstained_samples_remain_in_success_denominator():
    q, qi, matrix, p = setup_case()
    good = measure(p, qi, np.log(401 / 301), 224, photo_size=(700, 600), source_size=(401, 301), source_to_photo=matrix,
                   condition='clear', geometry='mild')
    bad = measure(np.full(9, np.nan), qi, np.log(401 / 301), 224, photo_size=(700, 600), source_size=(401, 301), source_to_photo=matrix,
                  condition='noisy', geometry='mild')
    result = summarize_rows([good, bad])
    assert result['all']['count'] == 2
    assert result['all']['invalid'] == 1
    assert result['all']['valid_count'] == 1
    assert result['all']['automatic_success_rate'] == .5
    assert result['all']['corner_nme']['count'] == 2
    assert result['all']['grid_error_max']['count'] == 1
    assert not admission_gates(result, result)['eligible']
    assert np.isfinite(selection_score(result, result)).all()


def test_missing_or_contradictory_ground_truth_never_receives_final_pass():
    q, qi, matrix, p = setup_case()
    row = measure(p, qi, np.log(401 / 301), 224, photo_size=(700, 600))
    assert not row['success'] and row['final_invalid']
    with pytest.raises(ValueError, match='disagree'):
        prepare_reference((700, 600), (401, 301), matrix, q + 5)


def test_source_resolution_does_not_change_normalized_shape_error():
    outputs = []
    for source_size in [(401, 301), (1601, 1201)]:
        q, _, matrix, _ = setup_case(source_size)
        reference = prepare_reference((700, 600), source_size, matrix, q)
        plan = {'action': 'warp', 'valid': True, 'matrix': np.array([[1.1, 0, -110], [0, 1, -100], [0, 0, 1]]), 'output_size': (441, 301)}
        outputs.append(measure_final(plan, q, source_size, matrix, reference)['grid_error_max'])
    assert outputs[0] == pytest.approx(outputs[1], abs=1e-10)


def test_admission_cannot_hide_one_bad_stratum_in_good_overall_results():
    q, qi, matrix, p = setup_case()
    base = measure(p, qi, np.log(401 / 301), 224, photo_size=(700, 600),
                   source_size=(401, 301), source_to_photo=matrix)
    rows = []
    for condition in ('clean', 'noisy'):
        for mode in ('front', 'mild', 'medium', 'hard'):
            for subgroup in (('upright', 'rotated') if mode == 'front' else (None,)):
                row = dict(base, condition=condition, geometry=mode,
                           ratio_error=.005, edge_ratio_error=.01)
                if subgroup:
                    row['front_subgroup'] = subgroup
                if mode == 'hard':
                    row.update(oracle_action='warp', oracle_crop_acceptable=False,
                               action='warp', unnecessary_warp=0, undercorrected=0)
                rows.append(row)
    for kind in ('digital', 'appearance', 'rectified'):
        rows.append(dict(base, condition='clean', geometry='preserve', preserve_kind=kind,
                         preserve_input=True, oracle_action='keep', action='keep',
                         unnecessary_preserve_change=0, ratio_error=.005, edge_ratio_error=.01))
    baseline = summarize_rows(rows)
    assert admission_gates(baseline, baseline)['eligible']
    changed = [dict(row) for row in rows]
    next(row for row in changed if row.get('condition') == 'noisy' and row.get('geometry') == 'hard')['lost_area'] = .01
    gates = admission_gates(summarize_rows(changed), baseline)
    assert not gates['eligible']
    assert 'content:noisy|hard' in gates['failed']


def test_temporary_source_ordinals_only_produce_aggregate_source_count():
    q, qi, matrix, p = setup_case()
    row = measure(p, qi, np.log(401 / 301), 224, photo_size=(700, 600),
                  source_size=(401, 301), source_to_photo=matrix,
                  source_index=731, kind='digital', mode='identity', condition='clean')
    summary = summarize_rows([row, row])
    assert row['source_index'] == 731
    assert summary['all']['source_count'] == 1
    assert summary['preserve_kind']['digital']['count'] == 2
    assert 'source_index' not in str(summary)






def test_historical_gate_requires_six_complete_strata_and_only_corner_regression():
    from copy import deepcopy
    from ptrn_hip.unified_evaluation import admission_gates
    cell = {'count': 478, 'invalid': 0, 'corner_nme': {'mean': .01, 'p90': .02}}
    summary = {'condition_geometry': {f'{condition}|{mode}': deepcopy(cell)
                for condition in ('clean', 'noisy') for mode in ('identity', 'mild', 'hard')}}
    assert admission_gates({'R': summary}, {'R': summary})['eligible']
    missing = deepcopy(summary)
    del missing['condition_geometry']['noisy|hard']
    assert not admission_gates({'R': missing}, {'R': summary})['eligible']
    partial = deepcopy(summary)
    partial['condition_geometry']['noisy|hard']['count'] = 477
    assert not admission_gates({'R': partial}, {'R': summary})['eligible']




def test_float64_intersection_preserves_coincident_rectangle_with_tiny_boundary_shift():
    from ptrn_hip.unified_evaluation import _intersection
    rectangle = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], np.float32)
    perturbed = rectangle.copy()
    perturbed[0, 0] = -1e-12
    # OpenCV float32 has returned 4000 for this 8000-area intersection.
    assert _intersection(rectangle, perturbed) == pytest.approx(8000., abs=1e-8)
    assert _intersection(perturbed, rectangle) == pytest.approx(8000., abs=1e-8)
    assert _intersection(rectangle[::-1], perturbed[::-1]) == pytest.approx(8000., abs=1e-8)


def test_float64_intersection_handles_partial_disjoint_and_rotated_convex_regions():
    from ptrn_hip.unified_evaluation import _intersection
    rectangle = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    assert _intersection(rectangle, rectangle + [50, 0]) == pytest.approx(4000.)
    assert _intersection(rectangle, rectangle + [150, 0]) == 0.
    assert _intersection(rectangle, rectangle + [100, 0]) == 0.
    diamond = np.array([[50, -10], [100, 40], [50, 90], [0, 40]], float)
    # Diamond area 5000 minus two 10-high, 20-wide triangles = 4800.
    assert _intersection(rectangle, diamond) == pytest.approx(4800., abs=1e-8)
    assert _intersection(rectangle + [1e6, -2e6], diamond + [1e6, -2e6]) == pytest.approx(4800., abs=1e-6)
    contained = np.array([[50, 10], [80, 40], [50, 70], [20, 40]], float)
    assert _intersection(rectangle, contained) == pytest.approx(1800., abs=1e-8)


def test_exact_projective_inverse_has_full_coverage_and_no_background():
    # Purely geometric fixture: neither pixels nor case-specific coordinates.
    source_size = (1001, 801)
    source_quad = pixel_corners(*source_size)
    transform = np.array([[.63, -.07, 91.], [.04, .51, 77.], [.00017, -.00009, 1.]])
    photo_quad = project(source_quad, transform)
    reference = prepare_reference((900, 700), source_size, transform, photo_quad)
    plan = {'action': 'warp', 'valid': True, 'matrix': np.linalg.inv(transform),
            'output_size': source_size, 'photo_size': (900, 700)}
    row = measure_final(plan, photo_quad, source_size, transform, reference)
    assert row['success']
    assert row['grid_error_max'] < 1e-12
    assert row['lost_area'] < 1e-12
    assert row['background_fraction'] < 1e-12
    assert row['corner_missing'] == row['direction_error'] == 0


def test_stable_area_does_not_hide_real_missing_content():
    from ptrn_hip.unified_evaluation import _intersection
    rectangle = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], float)
    inset = np.array([[1, 0], [99, 0], [99, 80], [1, 80]], float)
    lost = 1 - _intersection(rectangle, inset) / 8000
    assert lost == pytest.approx(.02)
    assert lost > .005
    invalid = np.array([[0, 0], [100, 80], [100, 0], [0, 80]], float)
    with pytest.raises(ValueError, match='nonconvex'):
        _intersection(rectangle, invalid)
