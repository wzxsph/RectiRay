import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import UNIT, pixel_corners, project, normalization
from ptrn_hip.unified_geometry import (decode_prediction, plan_transform, apply_transform,
    inverse_letterbox_matrix, fit_rotation_rectangle, similarity_crop, edge_ratio)


def prediction(quad, ratio, photo_size, size=224):
    lb = np.linalg.inv(inverse_letterbox_matrix(photo_size, size))
    normalized = project(project(quad, lb), normalization(size, size))
    h = cv2.getPerspectiveTransform(UNIT.astype(np.float32), normalized.astype(np.float32))
    return np.r_[h.ravel()[:8], np.log(ratio / edge_ratio(quad))]


def test_delta_decode_undoes_letterbox_and_preserves_portrait_width_axis():
    q = np.array([[200, 100], [450, 100], [450, 700], [200, 700]], float)
    p = prediction(q, .45, (800, 900))
    decoded = decode_prediction(p, 224, (800, 900))
    assert decoded['valid']
    np.testing.assert_allclose(decoded['quad'], q, atol=5e-5)
    assert decoded['ratio'] == pytest.approx(.45, abs=1e-7)
    assert decoded['r_edge'] == pytest.approx(250 / 600, abs=1e-7)
    assert p[8] != pytest.approx(np.log(.45))


def test_reliable_original_is_returned_identically_even_when_prediction_invalid():
    image = np.arange(120 * 80 * 3, dtype=np.uint8).reshape(120, 80, 3)
    plan = plan_transform((80, 120), None, None, source_kind='reliable_digital')
    assert plan['action'] == 'keep'
    assert apply_transform(image, plan) is image
    np.testing.assert_array_equal(apply_transform(image, plan), image)


def test_unknown_full_canvas_keeps_but_affine_compression_is_corrected():
    q = pixel_corners(401, 301)
    assert plan_transform((401, 301), q, 400 / 300)['action'] == 'keep'
    compressed = plan_transform((401, 301), q, 1.)
    assert compressed['action'] == 'warp'
    assert compressed['metrics']['d_ratio'] > .3


def test_rotated_portrait_crop_encloses_every_corner_without_axis_swap():
    theta = np.deg2rad(12)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q = np.array([[-100, -250], [100, -250], [100, 250], [-100, 250]], float) @ rotation.T + [450, 450]
    plan = plan_transform((900, 900), q, .4)
    assert plan['action'] == 'crop'
    assert plan['output_size'][0] < plan['output_size'][1]
    transformed = project(q, plan['matrix'])
    assert (transformed >= -1e-6).all()
    assert (transformed <= np.asarray(plan['output_size']) - 1 + 1e-6).all()
    np.testing.assert_allclose(plan['matrix'][:2, :2].T @ plan['matrix'][:2, :2], np.eye(2), atol=1e-8)
    assert transformed[1, 0] > transformed[0, 0]
    assert transformed[3, 1] > transformed[0, 1]


def test_small_nonrectangular_crop_uses_enclosing_rectangle_not_fitted_corners():
    q = np.array([[100, 100], [500, 102], [502, 500], [98, 498]], float)
    fit = fit_rotation_rectangle(q)
    plan = plan_transform((700, 700), q, fit['ratio'])
    assert plan['action'] == 'crop'
    transformed = project(q, plan['matrix'])
    assert (transformed >= -1e-6).all()
    assert (transformed <= np.asarray(plan['output_size']) - 1 + 1e-6).all()
    assert plan['metrics']['d_shape'] > 0


def test_ratio_comparison_uses_fitted_rectangle_not_delta_or_crop_canvas():
    q = np.array([[100, 100], [500, 100], [510, 490], [90, 500]], float)
    plan = plan_transform((700, 700), q, edge_ratio(q))
    assert plan['metrics']['d_ratio'] == pytest.approx(np.expm1(abs(np.log(edge_ratio(q) / fit_rotation_rectangle(q)['fitted_ratio']))))
    assert plan['metrics']['d_ratio'] > .0001
    assert plan['metrics']['r_box'] == pytest.approx(fit_rotation_rectangle(q)['fitted_ratio'])
    assert plan['metrics']['r_crop'] == pytest.approx(fit_rotation_rectangle(q)['ratio'])
    assert plan['metrics']['r_box'] != pytest.approx(plan['metrics']['r_crop'])


@pytest.mark.parametrize('relative', [.009, .010, .011])
def test_ten_percent_threshold_band_abstains(relative):
    q = np.array([[100, 100], [500, 100], [500, 500], [100, 500]], float)
    plan = plan_transform((700, 700), q, 1 + relative)
    assert plan['action'] == 'needs_confirmation'
    assert plan['valid']
    assert apply_transform(np.zeros((700, 700, 3), np.uint8), plan) is None


def test_illegal_quad_and_nonfinite_output_cannot_be_sampled():
    q = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], float)
    assert plan_transform((200, 200), q, 1)['action'] == 'needs_confirmation'
    assert not decode_prediction(np.full(9, np.nan), 224, (400, 500))['valid']
    q = pixel_corners(200, 200) - 1
    assert not plan_transform((200, 200), q, 1)['valid']


def test_output_pixel_limit_bounds_similarity_and_perspective():
    q = np.array([[50, 50], [950, 50], [950, 1950], [50, 1950]], float)
    for ratio in (900 / 1900, .8):
        plan = plan_transform((1000, 2000), q, ratio, {'max_output_pixels': 10_000})
        assert plan['action'] in ('crop', 'warp')
        assert np.prod(plan['output_size']) <= 10_000
