import cv2
import numpy as np
import pytest

from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.spatial_synthesis import TOPOLOGIES, quad_area, sample


def fixture(width=240, height=180):
    pixels = np.full((height, width, 3), 80, np.uint8)
    pixels[::13] = (200, 30, 20)
    pixels[:, ::17] = (10, 180, 40)
    return pixels


@pytest.mark.parametrize('width,height', [(174, 438), (700, 253), (240, 180)])
def test_photo_mask_geometry_occupancy_and_source_immutability(width, height):
    source = fixture(width, height)
    original = source.copy()
    source.setflags(write=False)
    for mode in ('front', 'mild', 'medium', 'hard'):
        for occupancy_bin in range(3):
            view = sample(source, 43, 100 + occupancy_bin + 3 * ('front', 'mild', 'medium', 'hard').index(mode),
                          kind='photo', mode=mode, occupancy_bin=occupancy_bin,
                          level='light', pure_roll=mode == 'front')
            w, h = view.info['photo_size']
            lo, hi = ((.30, .50), (.50, .75), (.75, .90))[occupancy_bin]
            actual = quad_area(view.quad) / (w * h)
            assert lo <= actual <= hi
            assert actual == pytest.approx(view.info['occupancy'], abs=1e-12)
            assert (view.quad >= 0).all() and (view.quad <= [w - 1, h - 1]).all()
            assert cv2.isContourConvex(view.quad.astype(np.float32))
            assert view.clean.shape == view.noisy.shape == (h, w, 3)
            assert view.film_mask.shape == view.screen_mask.shape == (h, w)
            assert np.count_nonzero(view.film_mask > .5) > 0
            assert np.count_nonzero((view.screen_mask > .5) & (view.film_mask < .1)) > 0
            np.testing.assert_allclose(project(pixel_corners(width, height),
                                               np.asarray(view.info['source_to_photo'])), view.quad, atol=1e-9)
            np.testing.assert_array_equal(source, original)
            assert view.info['topology'] in TOPOLOGIES


def test_clear_noisy_pair_keeps_same_film_screen_and_geometry():
    source = fixture()
    clear = sample(source, 1, 7, kind='photo', mode='medium', occupancy_bin=1,
                   level='clear', domain='dev')
    noisy = sample(source, 1, 7, kind='photo', mode='medium', occupancy_bin=1,
                   level='medium', domain='dev')
    repeat = sample(source, 1, 7, kind='photo', mode='medium', occupancy_bin=1,
                    level='medium', domain='dev')
    np.testing.assert_array_equal(clear.quad, noisy.quad)
    np.testing.assert_array_equal(clear.clean, noisy.clean)
    np.testing.assert_array_equal(clear.film_mask, noisy.film_mask)
    np.testing.assert_array_equal(clear.screen_mask, noisy.screen_mask)
    np.testing.assert_array_equal(noisy.noisy, repeat.noisy)
    assert clear.info['source_to_photo'] == noisy.info['source_to_photo']
    assert not np.array_equal(noisy.noisy, clear.clean)


def test_independent_roll_and_canvas_proposals_leave_auditable_realized_distribution():
    source = fixture(320, 200)
    views = [sample(source, 3, i, kind='photo', mode='mild', occupancy_bin=i % 2,
                    level='clear') for i in range(50)]
    assert len({view.info['requested_canvas_anchor'] for view in views}) >= 4
    assert len({view.info['topology'] for view in views}) == len(TOPOLOGIES)
    assert any(abs(view.info['roll']) > 10 for view in views)
    assert any(view.info['roll'] > 0 for view in views)
    assert any(view.info['roll'] < 0 for view in views)
    assert all(view.info['requested_roll_band'] == view.info['roll_band'] for view in views)
    assert all(view.info['roll_band'] != 2 for view in views if view.info['occupancy'] > .5)
    assert all(view.info['camera_attempts'] == 1 + sum(view.info['camera_rejections'].values())
               for view in views)
    assert all(sum(view.info['proposed_roll_bands'].values()) == view.info['camera_attempts']
               for view in views)


@pytest.mark.parametrize('kind,level', [('digital', 'clear'), ('appearance', 'medium'),
                                         ('rectified', 'light')])
def test_keep_masks_full_film_and_no_screen(kind, level):
    source = fixture()
    view = sample(source, 2, 8, kind=kind, level=level)
    assert np.all(view.film_mask == 1)
    assert not np.any(view.screen_mask)
    np.testing.assert_array_equal(view.quad, pixel_corners(240, 180))
    np.testing.assert_array_equal(view.info['source_to_photo'], np.eye(3))
    assert view.info['photo_size'] == [240, 180]


def test_infeasible_film_is_rejected_not_shrunk_or_relabeled():
    with pytest.raises(RuntimeError, match='no quota was downgraded'):
        sample(fixture(1000, 10), 1, 2, kind='photo', occupancy_bin=2)
