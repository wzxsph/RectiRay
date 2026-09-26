"""Technical capture invariants; these tests do not judge visual realism."""
import copy

import cv2
import numpy as np
import pytest

from ptrn_hip.capture_synthesis import SURFACE, CAMERA, recipe, render
from ptrn_hip.capture_review import reconstruct, assignments
from ptrn_hip.partial_synthesis import sample


@pytest.fixture
def scene():
    y, x = np.indices((320, 400))
    image = np.stack([50+x%160, 40+y%160, 80+(x+y)%120], -1).astype(np.uint8)
    mask = np.zeros(image.shape[:2], np.float32); mask[30:290, 40:360] = 1
    return image, mask


@pytest.mark.parametrize('effect', SURFACE+CAMERA)
def test_reproducible_active_and_supported(scene, effect):
    photo, surface = scene
    original = photo.copy(); mask_before = surface.copy()
    spec = recipe('screen', 'light', 55, [effect]); before = copy.deepcopy(spec)
    out, info = render(photo, surface, np.eye(3), [400, 320], spec)
    repeated, again = render(photo, surface, np.eye(3), [400, 320], spec)
    np.testing.assert_array_equal(out, repeated)
    np.testing.assert_array_equal(photo, original)
    np.testing.assert_array_equal(surface, mask_before)
    assert spec == before and info == again
    assert info['effects'][0]['native_delta']['max_255'] > 0
    assert info['effects'][0]['input224_delta']['max_255'] > 0
    if effect in SURFACE: np.testing.assert_array_equal(out[surface == 0], photo[surface == 0])
    else: assert np.any(out[surface == 0] != photo[surface == 0])


def test_original_and_clear_are_exact(scene):
    photo, surface = scene
    for material in ['keep', 'screen', 'film']:
        out, info = render(photo, surface, np.eye(3), [400, 320], recipe(material, 'clear', 2))
        np.testing.assert_array_equal(out, photo)
        assert not info['effects']
    with pytest.raises(ValueError): recipe('keep', 'light', 2, ['noise'])
    with pytest.raises(ValueError): recipe('film', 'light', 2, ['moire_color'])
    with pytest.raises(ValueError): recipe('screen', 'light', 2, ['noise', 'motion', 'haze'])
    with pytest.raises(ValueError): recipe('screen', 'clear', 2, ['noise'])


def test_empty_surface_is_reported_not_silent(scene):
    photo, surface = scene
    with pytest.raises(RuntimeError, match='Inactive'):
        render(photo, surface*0, np.eye(3), [400, 320], recipe('screen', 'light', 4, ['moire_gray']))


@pytest.mark.parametrize('family', ['viewer_screen', 'near_color', 'mixed_color', 'coco_context'])
@pytest.mark.parametrize('material', ['screen', 'film'])
def test_reconstruct_preserves_postcrop_and_labels(scene, family, material):
    source, _ = scene
    opts = dict(kind='photo', integrity='moderate', family=family, source_crop=True,
                high_occupancy=True, mode='mild', level='light', pure_roll=False)
    bg = np.flip(source, 0).copy()
    old = sample(source, 11, 7, seed=20260925, domain='dev', background=bg, **opts)
    row = {'id': 'syn-008', 'source_index': 11, 'scene_seed': 20260925, 'options': opts,
           'quad_gt': old.quad.tolist(), 'info': copy.deepcopy(old.info)}
    clean, restored, surface, _ = reconstruct(source, row, bg, material)
    assert clean.shape == old.photo.shape and surface.max() > 0
    assert restored.info == old.info and restored.ratio == old.ratio
    np.testing.assert_array_equal(restored.quad, old.quad)
    np.testing.assert_array_equal(restored.film_mask, old.film_mask)
    effect = 'moire_color' if material == 'screen' else 'window'
    out, info = render(clean, surface, restored.transform, restored.info['plane_size'], recipe(material, 'light', 3, [effect]))
    assert info['input224_delta']['max_255'] > 0
    np.testing.assert_array_equal(out[surface == 0], clean[surface == 0])
    assert row['info'] == old.info


def test_quotas_and_effect_caps():
    from collections import Counter
    rows = [{'id': f'syn-{i+1:03}', 'options': {'kind': 'keep', 'family': 'viewer_screen'}} for i in range(20)]
    for family, count in [('viewer_screen', 32), ('coco_context', 20), ('near_color', 20), ('mixed_color', 8)]:
        for _ in range(count): rows.append({'id': f'syn-{len(rows)+1:03}', 'options': {'kind': 'photo', 'family': family}})
    assigned = assignments(rows)
    counts = Counter((s['material'], s['severity']) for s in assigned.values())
    assert len(assigned) == 100 and counts['keep', 'clear'] == 20
    for material in ['screen', 'film']:
        assert [counts[material, s] for s in ['clear', 'light', 'medium', 'strong']] == [8, 24, 7, 1]
        for s in assigned.values():
            if s['material'] == material:
                assert len(s['effects']) <= (0 if s['severity']=='clear' else 2 if s['severity']=='light' else 3)
