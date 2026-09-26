from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from ptrn_hip.spatial_data import RealizedAudit, audit_schedule, build_schedule, validate_partition
from ptrn_hip.spatial_synthesis import sample


def test_inherited_partition_counts_overlap_and_patient_scope():
    splits = {'train': list(range(4114)), 'dev': list(range(4114, 4592)),
              'quarantine': list(range(4592, 4620))}
    result = validate_partition(splits)
    assert result['counts'] == {'train': 4114, 'dev': 478, 'quarantine': 28}
    assert result['patient_independent'] is False
    altered = deepcopy(splits)
    altered['dev'][0] = 0
    with pytest.raises(ValueError, match='overlap'):
        validate_partition(altered)


def test_train_schedule_preserves_full_recipe_and_front_loads_every_source():
    rows = build_schedule(range(4114))
    audit = audit_schedule(rows, range(4114), 'train')
    assert len(rows) == 80000
    assert audit['kind_counts'] == {'digital': 8000, 'appearance': 4000,
                                    'rectified': 4000, 'photo': 64000}
    assert audit['photo_mode_counts'] == {'front': 28800, 'mild': 22400,
                                          'medium': 9600, 'hard': 3200}
    assert audit['first_16000_sources_with_keep'] == 4114
    assert audit['first_16000_sources_with_photo'] == 4114
    assert Counter(Counter(r['source_index'] for r in rows).values()) == {19: 2280, 20: 1834}
    assert rows == build_schedule(reversed(range(4114)))


def test_schedule_rejects_corrupted_first_window_and_keeps_dev_pairs():
    rows = build_schedule(range(4114))
    broken = deepcopy(rows)
    first = broken[0]
    replacement = next(r for r in reversed(broken) if r['kind'] == first['kind']
                       and r['source_index'] != first['source_index'])
    first['source_index'] = replacement['source_index']
    with pytest.raises(ValueError):
        audit_schedule(broken, range(4114), 'train')
    dev = build_schedule(range(478), 'dev')
    assert audit_schedule(dev, range(478), 'dev')['geometry_pair_count'] == 478 * 4


def test_realized_audit_checks_actual_photo_area_masks_and_histograms():
    source = np.full((180, 240, 3), 100, np.uint8)
    rows = [dict(source_index=1, view_index=i, kind='photo', mode=mode,
                 occupancy_bin=occupancy_bin, level='clear', pure_roll=False,
                 domain='dev', condition='clear')
            for i, (mode, occupancy_bin) in enumerate(
                [('front', 0), ('mild', 1), ('medium', 2), ('hard', 1)])]
    audit = RealizedAudit('dev')
    for row in rows:
        view = sample(source, **{k: v for k, v in row.items() if k != 'condition'})
        audit.add(view, row)
    result = audit.finish(expected_rows=4, expected_sources=1)
    assert result['photo_count'] == 4
    assert result['photo_above_half'] == 3
    assert result['film_occupancy_min'] >= .30
    assert sum(result['topologies'].values()) == 4
    assert sum(result['requested_canvas_anchors_realized'].values()) == 4
    assert sum(result['roll_bands_realized'].values()) == 4
    with pytest.raises(ValueError, match='count'):
        audit.finish(expected_rows=5)
