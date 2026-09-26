import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.spatial_evaluation import evaluate_quad_predictions, evaluate_ratio_margin_grid
from ptrn_hip.unified_geometry import inverse_letterbox_matrix


def case():
    source_size, photo_size = (401, 301), (700, 600)
    quad = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], float)
    matrix = cv2.getPerspectiveTransform(pixel_corners(*source_size).astype(np.float32), quad.astype(np.float32))
    info = {'photo_size': photo_size, 'source_size': source_size, 'source_to_photo': matrix,
            'quad_photo': quad, 'log_ratio': np.log(source_size[0] / source_size[1]),
            'kind': 'photo', 'mode': 'mild', 'condition': 'clear', 'occupancy_bin': 1,
            'source_index': 7, 'split': 'dev'}
    inverse = inverse_letterbox_matrix(photo_size, 224)
    return info, project(quad, np.linalg.inv(inverse))


def test_perfect_quad_and_full_ratio_bridge_preserves_legacy_fields():
    info, q = case()
    summary, rows = evaluate_quad_predictions(q[None], [info['log_ratio']], [info], 224)
    row = rows[0]
    assert summary['all']['count'] == 1
    assert row['corner_nme'] < 1e-12
    assert row['ratio_error'] == pytest.approx(0)
    # The label is W/H while photo corner edges reflect (W-1)/(H-1).
    assert row['edge_ratio_error'] == pytest.approx(
        abs((400 / 300) / (401 / 301) - 1), rel=1e-5)
    assert row['action'] == 'crop' and row['success']
    assert row['grid_error_p90'] < 1e-6
    assert row['ratio_method'] == 'prediction' and not row['ratio_uses_gt']
    assert summary['evaluation_method']['margin_calibrated'] is False
    assert summary['condition_geometry']['clear|mild']['count'] == 1


def test_invalid_corner_remains_in_denominator_and_does_not_clamp_to_keep():
    info, q = case()
    bad = q.copy()
    bad[0] = [-100, -100]
    summary, rows = evaluate_quad_predictions(np.stack([q, bad]),
                                               [info['log_ratio']] * 2, [info, info], 224)
    assert summary['all']['count'] == 2
    assert summary['all']['automatic_count'] == 1
    assert rows[1]['invalid'] == 1
    assert rows[1]['action'] == 'needs_confirmation'
    assert not rows[1]['success']


def test_ratio_counterfactuals_are_explicitly_diagnostic_and_dev_only():
    info, q = case()
    wrong = np.log(1.8)
    predicted, rows = evaluate_quad_predictions(q[None], [wrong], [info], 224)
    oracle, oracle_rows = evaluate_quad_predictions(q[None], None, [info], 224,
                                                      ratio_method='oracle_gt_diagnostic')
    edge, _ = evaluate_quad_predictions(q[None], None, [info], 224, ratio_method='edge')
    assert rows[0]['ratio_error'] > .2
    assert oracle_rows[0]['ratio_error'] == 0
    assert oracle['evaluation_method']['ratio_uses_gt']
    assert not edge['evaluation_method']['ratio_uses_gt']
    assert predicted['all']['count'] == oracle['all']['count'] == edge['all']['count'] == 1
    grid = evaluate_ratio_margin_grid(q[None], [wrong], [info], 224, margins=(0., .005))
    assert len(grid) == 4
    assert all(len(value) == 2 for value in grid.values())
    assert grid['GT_quad_GT_ratio_perfect_diagnostic']['0.0']['evaluation_method']['corner_method'] == 'GT_quad_diagnostic'
    with pytest.raises(ValueError, match='development_only'):
        evaluate_ratio_margin_grid(q[None], [wrong], [info], 224, split='test')


def test_full_log_ratio_shape_is_checked_and_margin_is_unmodified():
    info, q = case()
    supplied = {'shape': .005}
    with pytest.raises(ValueError, match='full_log_ratios'):
        evaluate_quad_predictions(q[None], [[info['log_ratio']]], [info], 224)
    evaluate_quad_predictions(q[None], [info['log_ratio']], [info], 224, supplied,
                              margin_fraction=.0025)
    assert supplied == {'shape': .005}
