from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from ptrn_hip.unified_data import audit_schedule, build_schedule
from ptrn_hip.unified_synthesis import sample


def test_training_full_source_coverage_balanced_exposure_and_all_joint_quotas():
    rows = build_schedule(range(4114))
    audit = audit_schedule(rows, indices=range(4114), split='train')
    assert audit['views'] == 80000
    assert audit['sources'] == audit['sources_with_keep'] == audit['sources_with_photo'] == 4114
    exposures = Counter(row['source_index'] for row in rows)
    assert Counter(exposures.values()) == {19: 2280, 20: 1834}
    assert audit['kind_counts'] == {'digital': 8000, 'appearance': 4000, 'rectified': 4000, 'photo': 64000}
    for counts in audit['photo_occupancy_by_mode'].values():
        n = sum(counts.values())
        assert counts == {'0': n // 5, '1': n * 3 // 5, '2': n // 5}
    assert audit['photo_appearance_counts'] == {'clear': 28800, 'light': 22400, 'medium': 9600, 'strong': 3200}
    assert audit['front_pure_rotation_views'] == 14400
    assert rows == build_schedule(reversed(range(4114)))


@pytest.mark.parametrize('split,number', [('dev', 5258), ('test', 4780)])
def test_development_and_sealed_schedules_have_paired_geometry_without_rendering_sources(split, number):
    rows = build_schedule(range(478), split=split)
    audit = audit_schedule(rows, indices=range(478), split=split)
    assert audit['views'] == number
    assert audit['geometry_pair_count'] == 478 * 4
    assert audit['front_pure_rotation_views'] == 478
    for counts in audit['photo_occupancy_by_mode'].values():
        assert counts == {'0': 190, '1': 574, '2': 192}
        assert (counts['1'] + counts['2']) / sum(counts.values()) >= .8
    if split == 'test':
        assert not any(row['kind'] == 'digital' for row in rows)


def test_schedule_pairs_produce_identical_geometry_for_abstract_fixture():
    rows = build_schedule([23], split='dev')
    # Only an abstract fixture is rendered. This does not open sealed source data.
    source = np.full((80, 130, 3), 100, np.uint8)
    photo = [row for row in rows if row['kind'] == 'photo' and row['mode'] == 'medium']
    views = [sample(source, **{key: value for key, value in row.items() if key != 'condition'}) for row in photo]
    np.testing.assert_array_equal(views[0].quad, views[1].quad)
    np.testing.assert_array_equal(views[0].clean, views[1].clean)


def test_audit_rejects_coverage_pairing_and_quota_corruption():
    rows = build_schedule(range(478), split='dev')
    with pytest.raises(ValueError, match='coverage'):
        audit_schedule(rows, indices=range(479))
    broken = deepcopy(rows)
    pair = [row for row in broken if row['kind'] == 'photo' and row['mode'] == 'mild'][:2]
    pair[1]['occupancy_bin'] = (pair[0]['occupancy_bin'] + 1) % 3
    with pytest.raises(ValueError, match='geometry differs'):
        audit_schedule(broken)
    with pytest.raises(ValueError, match='duplicate'):
        audit_schedule(rows + [dict(rows[0])])


def test_domains_and_seed_changes_do_not_reuse_scene_allocation():
    a = build_schedule(range(478), split='dev')
    b = build_schedule(range(478), split='test')
    c = build_schedule(range(478), split='dev', seed=20260923)
    assert {row['domain'] for row in a} == {'dev'}
    assert {row['domain'] for row in b} == {'test'}
    assert a != c
    with pytest.raises(ValueError, match='unique'):
        build_schedule([1, 1], split='dev')
