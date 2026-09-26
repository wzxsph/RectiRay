"""Spatial-v2 schedules and aggregate audits; no source pixels or paths here."""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from .spatial_synthesis import SEED, TOPOLOGIES, quad_area
from .unified_data import (KEEP_COUNTS, PHOTO_COUNTS, audit_schedule as old_audit,
                           build_schedule as old_build)


def validate_partition(splits):
    """Check the inherited source-level partition without persisting any IDs."""
    required = {'train': 4114, 'dev': 478, 'quarantine': 28}
    if set(splits) != set(required):
        raise ValueError('Expected train/dev/quarantine source partition')
    values = {name: [int(i) for i in splits[name]] for name in required}
    if {name: len(v) for name, v in values.items()} != required:
        raise ValueError('Inherited source counts differ')
    if any(len(set(v)) != len(v) or min(v) < 0 for v in values.values()):
        raise ValueError('Duplicate or negative source ordinal')
    if any(set(values[a]) & set(values[b]) for a, b in
           [('train', 'dev'), ('train', 'quarantine'), ('dev', 'quarantine')]):
        raise ValueError('Inherited source partitions overlap')
    return {'counts': required, 'source_overlap': 0,
            'patient_independent': False, 'patient_mapping_available': False}


def build_schedule(indices, split='train', seed=SEED):
    """Reuse the exact 80k kind/mode/occupancy quotas and balance all sources.

    Train rows are reordered so the first 16k exposures include one keep and
    one photo for every source. New view ordinals are assigned after ordering.
    This never removes a source or changes the old 45/35/15/5 photo counts.
    """
    indices = list(indices)
    rows = old_build(indices, split=split, seed=seed)
    if split == 'train':
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 991]))
        by_source = defaultdict(lambda: {'keep': [], 'photo': []})
        for row in rows:
            by_source[row['source_index']]['photo' if row['kind'] == 'photo' else 'keep'].append(row)
        front = []
        chosen = set()
        for source_index in sorted(by_source):
            for kind in ('keep', 'photo'):
                choices = by_source[source_index][kind]
                if not choices:
                    raise ValueError('Every train source requires keep and photo')
                row = choices[int(rng.integers(len(choices)))]
                front.append(row)
                chosen.add(id(row))
        others = [row for row in rows if id(row) not in chosen]
        rng.shuffle(front)
        rng.shuffle(others)
        rows = front + others
        for i, row in enumerate(rows):
            row['view_index'] = i
    audit_schedule(rows, indices=indices, split=split)
    return rows


def audit_schedule(rows, indices=None, split=None):
    audit = old_audit(rows, indices=indices, split=split)
    if audit['split'] == 'train':
        if (audit['views'] != 80000 or audit['kind_counts'] !=
                {**KEEP_COUNTS, 'photo': 64000} or
                audit['photo_mode_counts'] != PHOTO_COUNTS):
            raise ValueError('Spatial train recipe differs from approved 80k budget')
        prefix = rows[:16000]
        keep_sources = {r['source_index'] for r in prefix if r['kind'] != 'photo'}
        photo_sources = {r['source_index'] for r in prefix if r['kind'] == 'photo'}
        if len(keep_sources) != audit['sources'] or len(photo_sources) != audit['sources']:
            raise ValueError('First 16k views do not cover every source in both roles')
        audit['first_16000_sources_with_keep'] = len(keep_sources)
        audit['first_16000_sources_with_photo'] = len(photo_sources)
    audit['renderer'] = 'spatial-v2; independent canvas/roll proposals with explicit feasibility audit'
    audit['old_test_scope'] = 'same_exposed_source_scene_only; never patient-independent'
    return audit


class RealizedAudit:
    """Stream realized photo geometry without retaining per-source metadata."""

    def __init__(self, split='train'):
        if split not in ('train', 'dev', 'test'):
            raise ValueError('Unknown split')
        self.split = split
        self.count = 0
        self.kinds = Counter()
        self.modes = Counter()
        self.occupancy_bins = Counter()
        self.actual_occupancy_005 = Counter()
        self.topologies = Counter()
        self.canvas_anchors = Counter()
        self.roll_bands = Counter()
        self.roll_signs = Counter()
        self.proposed_roll_bands = Counter()
        self.proposed_canvas_anchors = Counter()
        self.rejections = Counter()
        self.source_keep = set()
        self.source_photo = set()
        self.occupancy_min = float('inf')
        self.occupancy_max = 0.
        self.photo_above_half = 0
        self.photo_count = 0

    def add(self, view, row):
        if row['domain'] != self.split or view.info['domain'] != self.split:
            raise ValueError('Mixed realized split')
        if row['kind'] != view.info['kind']:
            raise ValueError('Rendered kind differs from schedule')
        self.count += 1
        kind = row['kind']
        self.kinds[kind] += 1
        if kind != 'photo':
            self.source_keep.add(row['source_index'])
            if not np.all(view.film_mask == 1) or np.any(view.screen_mask):
                raise ValueError('Keep masks do not describe a full source canvas')
            return
        self.source_photo.add(row['source_index'])
        self.photo_count += 1
        self.modes[row['mode']] += 1
        self.occupancy_bins[row['occupancy_bin']] += 1
        width, height = view.info['photo_size']
        actual = quad_area(view.quad) / (width * height)
        low, high = ((.30, .50), (.50, .75), (.75, .90))[row['occupancy_bin']]
        if (not math.isfinite(actual) or actual < low - 1e-9 or actual > high + 1e-9 or
                (row['occupancy_bin'] > 0 and actual <= low) or
                abs(actual - view.info['occupancy']) > 1e-9 or
                np.any(view.quad < 0) or np.any(view.quad > [width - 1, height - 1])):
            raise ValueError('Actual film occupancy/corner containment violates schedule')
        if view.film_mask.shape != (height, width) or view.screen_mask.shape != (height, width):
            raise ValueError('Film/screen masks differ from photo size')
        if (np.any(view.film_mask < 0) or np.any(view.film_mask > 1) or
                np.any(view.screen_mask < 0) or np.any(view.screen_mask > 1)):
            raise ValueError('Film/screen mask range invalid')
        self.occupancy_min = min(self.occupancy_min, actual)
        self.occupancy_max = max(self.occupancy_max, actual)
        self.actual_occupancy_005[f'{math.floor(actual / .05) * .05:.2f}'] += 1
        self.photo_above_half += actual > .5
        self.topologies[view.info['topology']] += 1
        self.canvas_anchors[str(view.info['requested_canvas_anchor'])] += 1
        self.roll_bands[view.info['roll_band']] += 1
        self.roll_signs['positive' if view.info['roll'] > 0 else
                        'negative' if view.info['roll'] < 0 else 'zero'] += 1
        self.proposed_roll_bands.update(view.info['proposed_roll_bands'])
        self.proposed_canvas_anchors.update(view.info['proposed_canvas_anchors'])
        self.rejections.update(view.info['camera_rejections'])

    def finish(self, expected_rows=None, expected_sources=None):
        if expected_rows is not None and self.count != expected_rows:
            raise ValueError('Realized view count differs from schedule')
        if self.split == 'train':
            if self.count != 80000 or self.photo_count != 64000 or self.modes != PHOTO_COUNTS:
                raise ValueError('Full train recipe not realized')
            if self.photo_above_half < math.ceil(.8 * self.photo_count):
                raise ValueError('Less than 80% of photo views have film occupancy above 50%')
            if expected_sources is not None and (len(self.source_keep) != expected_sources or
                                                 len(self.source_photo) != expected_sources):
                raise ValueError('Actual rendering omitted a train source')
        if self.photo_count and self.occupancy_min < .30 - 1e-9:
            raise ValueError('Film occupancy below 30%')
        return {'split': self.split, 'views': self.count, 'kind_counts': dict(self.kinds),
                'photo_mode_counts': dict(self.modes), 'photo_count': self.photo_count,
                'photo_above_half': self.photo_above_half,
                'photo_above_half_fraction': self.photo_above_half / self.photo_count if self.photo_count else None,
                'film_occupancy_min': self.occupancy_min if self.photo_count else None,
                'film_occupancy_max': self.occupancy_max if self.photo_count else None,
                'actual_film_occupancy_005': dict(sorted(self.actual_occupancy_005.items())),
                'topologies': dict(self.topologies), 'requested_canvas_anchors_realized': dict(self.canvas_anchors),
                'roll_bands_realized': {str(k): v for k, v in self.roll_bands.items()},
                'roll_bands_proposed': dict(self.proposed_roll_bands),
                'canvas_anchors_proposed': dict(self.proposed_canvas_anchors),
                'roll_signs': dict(self.roll_signs), 'rejected_proposals': dict(self.rejections),
                'sources_with_keep': len(self.source_keep), 'sources_with_photo': len(self.source_photo),
                'patient_independent': False}
