"""New train-only recipe contracts, using synthetic sources and backgrounds."""
from collections import Counter
import json

import cv2
import numpy as np
import pytest

from ptrn_hip.partial_data import STEP2000_RECIPE, PartialDataset, recipe, sha256
from ptrn_hip.partial_synthesis import sample


def fixture():
    y, x = np.mgrid[:180, :220]
    return np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)


def test_train_distribution_is_versioned_and_dev_is_frozen():
    rows = [recipe(i, v, round_index=0, seed=314, domain='train', version=STEP2000_RECIPE)
            for i in range(500) for v in range(16)]
    assert rows == [recipe(i, v, round_index=0, seed=314, domain='train', version=STEP2000_RECIPE)
                    for i in range(500) for v in range(16)]
    photos = [r for r in rows if r['kind'] == 'photo']
    assert len(photos) == 6400
    counts = Counter(r['occupancy_bin_v2'] for r in photos)
    for bin_number, expected in enumerate((.20, .30, .35, .10, .05)):
        assert abs(counts[bin_number]/len(photos) - expected) < .025
    integrity = Counter(r['integrity'] for r in photos)
    for name, expected in zip(('complete', 'light', 'moderate', 'severe'),
                              (.80, .12, .06, .02)):
        assert abs(integrity[name]/len(photos) - expected) < .02
    assert .10 <= sum(r['source_crop'] for r in photos)/len(photos) <= .15
    family = Counter(r['family'] for r in photos)
    for name, expected in zip(('viewer_screen', 'coco_context', 'near_color', 'mixed_color'),
                              (.40, .25, .25, .10)):
        assert abs(family[name]/len(photos) - expected) < .025
    assert .47 <= sum(r['capture_material'] == 'screen' for r in photos)/len(photos) <= .53
    assert recipe(2, 7, round_index=0, seed=314, domain='dev') == recipe(
        2, 7, round_index=0, seed=314, domain='dev', version='partial-v3')
    with pytest.raises(ValueError, match='train-only'):
        recipe(2, 7, round_index=0, seed=314, domain='dev', version=STEP2000_RECIPE)


def test_actual_geometry_and_capture_respect_requested_bins():
    source = fixture()
    original = source.copy()
    background = np.flipud(source).copy()
    completed = set()
    for i in range(80):
        options = recipe(i//16, i % 16, round_index=0, seed=614,
                         domain='train', version=STEP2000_RECIPE)
        if options['kind'] == 'keep':
            continue
        for attempt in range(20):
            try:
                view = sample(source, i//16, i, seed=614 + attempt*1000003,
                              domain='train', background=background, **options)
                break
            except RuntimeError:
                if attempt == 19:
                    raise
        lo, hi = options['occupancy_range']
        assert lo <= view.info['occupancy'] < hi
        assert view.info['scene_version'] == STEP2000_RECIPE
        assert view.info['capture']['material'] == options['capture_material']
        assert view.info['capture']['severity'] == options['capture_severity']
        assert view.info['capture_seed'] == options['capture_seed']
        assert view.input(224)['mask'].shape == (224, 224)
        if options['integrity'] == 'complete':
            assert view.info['missing_fraction'] < 1e-5
        else:
            ml, mh = options['missing_range']
            assert ml < view.info['missing_fraction'] <= mh
        if options['source_crop']:
            assert .05 <= view.info['source_pre_crop_area_loss'] <= .10
        completed.add(options['occupancy_bin_v2'])
    assert completed == set(range(5))
    np.testing.assert_array_equal(source, original)


def test_dataset_entry_point_keeps_dev_binding_and_reproducibility(tmp_path):
    source = fixture()
    bundle = tmp_path/'bundle'
    background_dir = tmp_path/'backgrounds'
    (bundle/'data/films').mkdir(parents=True)
    background_dir.mkdir()
    splits = {'train': [0, 1, 2], 'dev': [3, 4], 'quarantine': [5]}
    records = []
    for i in range(6):
        path = bundle/f'data/films/{i}.png'
        assert cv2.imwrite(str(path), np.clip(source.astype(np.int16)+i, 0, 255).astype(np.uint8))
        records.append({'source_path': str(path.relative_to(bundle)),
                        'source_sha256': sha256(path),
                        'split': next(name for name, ids in splits.items() if i in ids)})
    backgrounds = []
    for index, name in enumerate(('train', 'dev')):
        path = background_dir/f'{name}.png'
        assert cv2.imwrite(str(path), np.clip(source.astype(np.int16)+index*10, 0, 255).astype(np.uint8))
        backgrounds.append({'id': name, 'path': path.name, 'sha256': sha256(path),
                            'split': name, 'source_kind': 'real_photo'})
    (bundle/'data/manifest.json').write_text(json.dumps({'records': records}))
    (bundle/'data/split.json').write_text(json.dumps(splits))
    (bundle/'data/backgrounds-manifest.json').write_text(json.dumps({'rows': backgrounds}))
    old_train = PartialDataset(bundle, background_dir)
    new_train = PartialDataset(bundle, background_dir, recipe_version=STEP2000_RECIPE)
    frozen_dev = PartialDataset(bundle, background_dir, split='dev')
    assert old_train.fingerprint.keys() == frozen_dev.fingerprint.keys()
    assert 'recipe_version' not in frozen_dev.fingerprint
    assert new_train.fingerprint['recipe_version'] == STEP2000_RECIPE
    assert new_train.fingerprint != old_train.fingerprint
    with pytest.raises(ValueError, match='Frozen development'):
        PartialDataset(bundle, background_dir, split='dev', recipe_version=STEP2000_RECIPE)
    original = new_train.render(0)
    assert original.info['scene_version'] == STEP2000_RECIPE
    np.testing.assert_array_equal(original.photo, cv2.cvtColor(
        cv2.imread(str(bundle/'data/films/0.png')), cv2.COLOR_BGR2RGB))
    a = new_train.render(1)
    b = new_train.render(1)
    np.testing.assert_array_equal(a.photo, b.photo)
    np.testing.assert_array_equal(a.quad, b.quad)
