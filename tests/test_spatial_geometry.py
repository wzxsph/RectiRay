import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.spatial_geometry import plan_transform
from ptrn_hip.unified_geometry import apply_transform
from ptrn_hip.unified_evaluation import measure_final


def test_reliable_provenance_bypasses_bad_prediction_without_resampling():
    image = np.arange(120 * 80 * 3, dtype=np.uint8).reshape(120, 80, 3)
    plan = plan_transform((80, 120), None, None, source_kind='reliable_digital')
    assert plan['action'] == 'keep'
    assert plan['metrics']['provenance_bypass']
    assert apply_transform(image, plan) is image


def test_unknown_full_canvas_requires_valid_in_bounds_geometry():
    q = pixel_corners(401, 301)
    keep = plan_transform((401, 301), q, 401 / 301)
    assert keep['action'] == 'keep'
    assert keep['metrics']['predicted_keep']['predicted_background_fraction'] == pytest.approx(0, abs=1e-6)
    outside = q.copy()
    outside[0, 0] = -.1
    rejected = plan_transform((401, 301), outside, 401 / 301)
    assert rejected['action'] == 'needs_confirmation'
    assert 'outside_photo' in rejected['reason']
    np.testing.assert_array_equal(outside[0], [-.1, 0])


def test_unknown_compressed_full_canvas_requires_warp():
    plan = plan_transform((401, 301), pixel_corners(401, 301), 1.)
    assert plan['action'] == 'warp'
    assert plan['metrics']['crop_candidate']['ratio_error'] > .01


def test_roll_crop_preserves_ordered_corners_without_axis_swap():
    theta = np.deg2rad(8)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    q = np.array([[-120, -170], [120, -170], [120, 170], [-120, 170]], float) @ rotation.T + 450
    plan = plan_transform((900, 900), q, 240 / 340)
    assert plan['action'] == 'crop'
    assert plan['output_size'][0] < plan['output_size'][1]
    assert plan['metrics']['grid_error_max'] < .005
    transformed = project(q, plan['matrix'])
    assert (transformed >= -.5).all()
    assert (transformed <= np.asarray(plan['output_size']) - .5).all()


def test_mirror_degenerate_and_nonfinite_quads_confirm():
    mirrored = np.array([[50, 50], [50, 150], [150, 150], [150, 50]], float)
    collapsed = np.array([[50, 50], [50, 50], [150, 150], [50, 150]], float)
    for q in (mirrored, collapsed, np.full((4, 2), np.nan)):
        plan = plan_transform((200, 200), q, 1.)
        assert plan['action'] == 'needs_confirmation'
        assert not plan['valid']


def test_margin_is_explicit_uncalibrated_and_increases_crop_exterior():
    q = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], float)
    base = plan_transform((700, 600), q, 400 / 300)
    padded = plan_transform((700, 600), q, 400 / 300, {'margin_fraction': .005})
    assert base['action'] == padded['action'] == 'crop'
    assert padded['metrics']['margin_fraction'] == .005
    assert padded['metrics']['margin_calibrated'] is False
    assert np.prod(padded['output_size']) > np.prod(base['output_size'])
    assert padded['metrics']['predicted_background_fraction'] > base['metrics']['predicted_background_fraction']


def test_margin_cannot_invent_pixels_outside_captured_photo():
    q = np.array([[1., 20.], [150., 20.], [150., 170.], [1., 170.]])
    plan = plan_transform((200, 200), q, 1., {'margin_fraction': .02})
    assert plan['action'] == 'needs_confirmation'
    assert plan['reason'] == 'margin_exceeds_captured_photo'


def test_warp_margin_checks_actual_inverse_output_corners():
    # Radial expansion of this legal quad stays in bounds, yet the warp's
    # output upper-right corner maps to y=-3.56 in the 500x500 input photo.
    q = np.array([[8.238564352888574, 83.68134372353992],
                  [469.45618348803663, 5.865933488756834],
                  [402.130605405585, 489.62211996456756],
                  [78.5041260631282, 388.5014089835029]])
    unpadded = plan_transform((500, 500), q, 1.)
    assert unpadded['action'] == 'warp'
    padded = plan_transform((500, 500), q, 1., {'margin_fraction': .01})
    assert padded['action'] == 'needs_confirmation'
    assert padded['reason'] == 'margin_exceeds_captured_photo'
    np.testing.assert_array_equal(q[0], [8.238564352888574, 83.68134372353992])


def test_near_perspective_and_ratio_thresholds_abstain():
    q = np.array([[100., 100.], [500., 104.5], [500., 400.], [100., 400.]])
    perspective = plan_transform((700, 600), q, 400 / 300)
    assert perspective['action'] == 'needs_confirmation'
    assert perspective['reason'] == 'correction_threshold_boundary'
    assert .0045 <= perspective['metrics']['grid_error_max'] <= .0055
    rectangle = np.array([[100., 100.], [500., 100.], [500., 400.], [100., 400.]])
    ratio = plan_transform((700, 600), rectangle, 400 / 300 * 1.01)
    assert ratio['action'] == 'needs_confirmation'
    assert ratio['reason'] == 'correction_threshold_boundary'
    assert ratio['metrics']['ratio_error'] == pytest.approx(.01)


def test_crop_and_warp_sample_original_once(monkeypatch):
    original = np.zeros((600, 700, 3), dtype=np.uint8)
    q = np.array([[100., 100.], [500., 100.], [500., 400.], [100., 400.]])
    crop = plan_transform((700, 600), q, 400 / 300)
    warp = plan_transform((700, 600), q, 1.)
    assert (crop['action'], warp['action']) == ('crop', 'warp')
    calls = []

    def sample(image, matrix, size, **options):
        calls.append((image, np.asarray(matrix).copy(), size, options))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(cv2, 'warpPerspective', sample)
    assert apply_transform(original, crop).shape[:2] == crop['output_size'][::-1]
    assert apply_transform(original, warp).shape[:2] == warp['output_size'][::-1]
    assert len(calls) == 2 and all(call[0] is original for call in calls)


def test_plan_is_compatible_with_frozen_final_evaluator():
    source_size = (401, 301)
    photo_size = (700, 600)
    source_to_photo = cv2.getPerspectiveTransform(
        pixel_corners(*source_size).astype(np.float32),
        np.array([[120, 100], [520, 100], [520, 400], [120, 400]], np.float32))
    q = project(pixel_corners(*source_size), source_to_photo)
    plan = plan_transform(photo_size, q, 401 / 301)
    result = measure_final(plan, q, source_size, source_to_photo)
    assert plan['action'] == 'crop'
    assert result['automatic'] and result['success']
    assert not result['corner_missing'] and not result['direction_error']


def test_only_prediction_and_photo_size_are_needed_for_action():
    q = np.array([[90, 70], [430, 72], [428, 330], [92, 329]], float)
    a = plan_transform((600, 450), q, 340 / 260)
    b = plan_transform((600, 450), q.copy(), 340 / 260)
    assert a['action'] == b['action']
    assert a['reason'] == b['reason']
    assert set(a).issuperset({'action', 'matrix', 'output_size', 'photo_size', 'metrics', 'valid', 'reason'})
