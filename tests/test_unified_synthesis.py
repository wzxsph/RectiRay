import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.unified_synthesis import (
    MODES, OCCUPANCY_BOUNDS, ROLL_BOUNDS, TILT_BOUNDS, _rotation, quad_area, sample,
)


def fixture(width=240, height=180):
    """An abstract grid and colored disks, not a medical image."""
    result = np.full((height, width, 3), 65, np.uint8)
    result[::12, :, :] = 95
    result[:, ::12, :] = 95
    for color, location in zip(((255, 0, 0), (0, 255, 0), (0, 0, 255)),
                               ((.25, .30), (.72, .32), (.55, .72))):
        cv2.circle(result, (round(location[0] * (width - 1)), round(location[1] * (height - 1))),
                   max(3, min(width, height) // 24), color, -1)
    return result


@pytest.mark.parametrize('shape', [(174, 438), (700, 253), (240, 180)])
def test_actual_projected_occupancy_complete_corners_and_physical_pose(shape):
    # The first two shapes cover the audited approximately .3977--2.7668 range.
    source = fixture(*shape)
    source.setflags(write=False)
    for mode in MODES:
        for occupancy_bin in range(3):
            view = sample(source, 12, occupancy_bin + 3 * MODES.index(mode),
                          kind='photo', mode=mode, occupancy_bin=occupancy_bin,
                          pure_roll=mode == 'front')
            width, height = view.info['common_size']
            lo, hi = OCCUPANCY_BOUNDS[occupancy_bin]
            measured = quad_area(view.quad) / (width * height)
            assert lo <= measured <= hi
            if occupancy_bin:
                assert measured > lo
            assert measured == pytest.approx(view.info['occupancy'], abs=1e-12)
            assert (view.quad >= 0).all()
            assert (view.quad <= [width - 1, height - 1]).all()
            assert cv2.isContourConvex(view.quad.astype(np.float32))
            assert cv2.contourArea(view.quad.astype(np.float32), oriented=True) > 0
            assert TILT_BOUNDS[mode][0] <= view.info['tilt'] <= TILT_BOUNDS[mode][1]
            assert abs(view.info['roll']) <= ROLL_BOUNDS[mode]
            if mode == 'front':
                assert abs(view.info['roll']) >= 1.
            rotation = _rotation(view.info['tilt'], view.info['tilt_azimuth'], view.info['roll'])
            normal_tilt = np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1)))
            assert normal_tilt == pytest.approx(view.info['tilt'], abs=1e-6)
            assert np.linalg.det(rotation) == pytest.approx(1.)
            np.testing.assert_allclose(project(pixel_corners(*shape), np.array(view.info['source_to_photo'])),
                                       view.quad, atol=1e-9)
            assert view.log_ratio == pytest.approx(np.log(shape[0] / shape[1]))
            assert 1 <= view.info['camera_attempts'] <= 100


def test_appearance_pairs_share_geometry_background_and_seeds_are_domain_isolated():
    source = fixture()
    a = sample(source, 7, 4, kind='photo', mode='mild', level='clear', domain='dev')
    b = sample(source, 7, 4, kind='photo', mode='mild', level='medium', domain='dev')
    again = sample(source, 7, 4, kind='photo', mode='mild', level='medium', domain='dev')
    other = sample(source, 7, 4, kind='photo', mode='mild', level='clear', domain='test')
    np.testing.assert_array_equal(a.quad, b.quad)
    np.testing.assert_array_equal(a.clean, b.clean)
    np.testing.assert_array_equal(a.screen_mask, b.screen_mask)
    np.testing.assert_array_equal(b.noisy, again.noisy)
    assert a.info['source_to_photo'] == b.info['source_to_photo']
    assert not np.array_equal(a.quad, other.quad)
    assert not np.array_equal(b.clean, b.noisy)
    assert np.count_nonzero(b.screen_mask) > quad_area(b.quad)


def test_visible_landmarks_follow_pixel_transform_and_letterbox_labels():
    source = fixture(320, 200)
    view = sample(source, 0, 102, kind='photo', mode='medium', occupancy_bin=2)
    points = np.array([[round(.25 * 319), round(.30 * 199)],
                       [round(.72 * 319), round(.32 * 199)],
                       [round(.55 * 319), round(.72 * 199)]])
    projected = project(points, np.array(view.info['source_to_photo']))
    for channel, (x, y) in enumerate(np.rint(projected).astype(int)):
        pixel = view.clean[y, x].astype(int)
        assert pixel[channel] > 220
        assert np.delete(pixel, channel).sum() < 30
    for size in (224, 384):
        image, label, q = view.input(size)
        assert image.shape == (size, size, 3)
        h = np.r_[label[:8], 1.].reshape(3, 3)
        unit = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        np.testing.assert_allclose((project(unit, h) + 1) * (size - 1) / 2, q, atol=2e-4)
        assert label[-1] == pytest.approx(view.log_ratio)


@pytest.mark.parametrize('kind,level', [('digital', 'clear'), ('appearance', 'medium'), ('rectified', 'light')])
def test_keep_views_have_full_original_geometry_and_never_mutate_source(kind, level):
    source = fixture(247, 179)
    before = source.copy()
    source.setflags(write=False)
    view = sample(source, 1, 15, kind=kind, level=level)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(view.quad, pixel_corners(247, 179))
    np.testing.assert_array_equal(view.info['source_to_photo'], np.eye(3))
    assert view.clean.shape == source.shape == view.noisy.shape
    assert view.log_ratio == pytest.approx(np.log(247 / 179))
    if kind == 'digital':
        np.testing.assert_array_equal(view.noisy, source)
    if kind == 'rectified':
        assert 0 < view.info['rectified_from_tilt'] <= 8


def test_invalid_strata_and_unattainable_pose_fail_without_downgrading():
    source = fixture()
    with pytest.raises(ValueError, match='Hard geometry'):
        sample(source, 1, 1, kind='photo', mode='hard', level='strong')
    with pytest.raises(ValueError, match='Digital-original'):
        sample(source, 1, 1, kind='digital', level='light')
    with pytest.raises(ValueError, match='pure-rotation'):
        sample(source, 1, 1, kind='photo', mode='mild', pure_roll=True)
    with pytest.raises(RuntimeError, match='100 attempts'):
        sample(fixture(1000, 10), 1, 1, kind='photo', pure_roll=True, occupancy_bin=2)
