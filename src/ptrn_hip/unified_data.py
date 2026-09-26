"""Exact, reproducible view schedules; no source pixels or private paths are read."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import numpy as np

from .unified_synthesis import DOMAINS, KINDS, LEVELS, MODES, SEED

PHOTO_COUNTS = {'front': 28800, 'mild': 22400, 'medium': 9600, 'hard': 3200}
KEEP_COUNTS = {'digital': 8000, 'appearance': 4000, 'rectified': 4000}
PHOTO_LEVEL_COUNTS = {'clear': 28800, 'light': 22400, 'medium': 9600, 'strong': 3200}


def _quotas(total):
    """Largest remainders, with at least ceil(80% * total) above 50%."""
    ideals = np.array([.2, .6, .2]) * total
    counts = np.floor(ideals).astype(int)
    for i in np.argsort(-(ideals - counts), kind='stable')[:total - counts.sum()]:
        counts[i] += 1
    if counts[0] > total - math.ceil(.8 * total):
        counts[0] -= 1
        deficit = ideals[1:] - counts[1:]
        counts[1 + int(np.argmax(deficit))] += 1
    return counts.tolist()


def _expand(counts, rng):
    values = [key for key, count in counts.items() for _ in range(count)]
    rng.shuffle(values)
    return values


def _row(source_index, view_index, kind, *, mode='front', occupancy_bin=1,
         level='clear', pure_roll=False, domain='train', condition='train'):
    return {'source_index': int(source_index), 'view_index': int(view_index), 'kind': kind,
            'mode': mode, 'occupancy_bin': int(occupancy_bin), 'level': level,
            'pure_roll': bool(pure_roll), 'domain': domain, 'condition': condition}


def build_schedule(indices, split='train', seed=SEED):
    """Construct the approved train/D/T schedule without rendering any images.

    Generic source counts are accepted for synthetic fixture tests; production
    callers retain the audited 4114/478 split and check those aggregate counts.
    Each source has floor/ceil(80000/N) training slots, including keep and photo.
    """
    if split not in DOMAINS:
        raise ValueError('Expected train, dev, or test split')
    indices = [int(i) for i in indices]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError('Source ordinals must be nonempty, unique and nonnegative')
    indices.sort()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), DOMAINS[split], 900]))
    if split != 'train':
        rows = []
        n = len(indices)
        modes = {}
        for mode in MODES:
            bins = _expand(dict(enumerate(_quotas(n))), rng)
            rolls = [False] * ((n + 1) // 2) + [True] * (n // 2)
            rng.shuffle(rolls)
            noisy = (['light'] * n if mode == 'hard' else
                     _expand({'light': n - n // 3 - n // 12,
                              'medium': n // 3, 'strong': n // 12}, rng))
            modes[mode] = list(zip(bins, rolls, noisy))
        for i, source_index in enumerate(indices):
            if split == 'dev':
                rows.append(_row(source_index, 0, 'digital', domain=split, condition='keep'))
            rows.append(_row(source_index, 1, 'appearance', level='light', domain=split, condition='keep'))
            rows.append(_row(source_index, 2, 'rectified', level='light', domain=split, condition='keep'))
            for j, mode in enumerate(MODES):
                occupancy, roll, noisy = modes[mode][i]
                for condition, level in [('clear', 'clear'), ('noisy', noisy)]:
                    rows.append(_row(source_index, 10 + j, 'photo', mode=mode,
                                     occupancy_bin=occupancy, level=level,
                                     pure_roll=roll if mode == 'front' else False,
                                     domain=split, condition=condition))
        audit_schedule(rows, indices=indices, split=split)
        return rows
    n = len(indices)
    if n > sum(KEEP_COUNTS.values()):
        raise ValueError('Keep budget cannot cover every requested source')
    order = rng.permutation(n)
    totals = np.full(n, 80000 // n, dtype=int)
    totals[order[:80000 % n]] += 1
    keeps = np.full(n, 16000 // n, dtype=int)
    keeps[rng.permutation(n)[:16000 % n]] += 1
    if np.any(totals - keeps < 1):
        raise ValueError('Photo budget cannot cover every requested source')
    keep_sources = np.repeat(indices, keeps).tolist()
    photo_sources = np.repeat(indices, totals - keeps).tolist()
    rng.shuffle(keep_sources)
    rng.shuffle(photo_sources)
    keep_descriptors = []
    for kind, count in KEEP_COUNTS.items():
        level_counts = ({'clear': count} if kind == 'digital' else
                        {'light': 2800, 'medium': 1200} if kind == 'appearance' else
                        {'clear': 2000, 'light': 2000})
        keep_descriptors.extend((kind, level) for level in _expand(level_counts, rng))
    rng.shuffle(keep_descriptors)
    rows = [_row(index, 0, kind, level=level) for index, (kind, level) in zip(keep_sources, keep_descriptors)]
    hard_levels = _expand({'clear': 1800, 'light': 1400}, rng)
    other_levels = _expand({'clear': 27000, 'light': 21000, 'medium': 9600, 'strong': 3200}, rng)
    photo_descriptors = []
    for mode, count in PHOTO_COUNTS.items():
        for occupancy, number in enumerate(_quotas(count)):
            for i in range(number):
                level = (hard_levels if mode == 'hard' else other_levels).pop()
                photo_descriptors.append((mode, occupancy, level, mode == 'front' and i < number // 2))
    rng.shuffle(photo_descriptors)
    rows.extend(_row(index, 0, 'photo', mode=mode, occupancy_bin=occupancy,
                     level=level, pure_roll=roll)
                for index, (mode, occupancy, level, roll) in zip(photo_sources, photo_descriptors))
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row['view_index'] = i
    audit_schedule(rows, indices=indices, split=split)
    return rows


def audit_schedule(rows, indices=None, split=None):
    """Validate counts/coverage/pairing and return aggregate-only statistics.

    This audits intended sampling quotas; a separate renderer audit must check
    actual tilt, film area, complete corners, and effective network resolution.
    """
    if not rows:
        raise ValueError('Empty schedule')
    split = split or rows[0]['domain']
    if split not in DOMAINS:
        raise ValueError('Unknown schedule domain')
    observed = set(row['source_index'] for row in rows)
    expected = observed if indices is None else set(map(int, indices))
    errors = []
    if observed != expected:
        errors.append('source coverage differs from supplied split')
    kinds, modes, levels = Counter(), Counter(), Counter()
    source_totals, keeps, photos = Counter(), Counter(), Counter()
    joint, pairs = Counter(), defaultdict(list)
    signatures = set()
    for row in rows:
        kind, mode, level = row['kind'], row['mode'], row['level']
        if kind not in KINDS or mode not in MODES or level not in LEVELS or row['domain'] != split:
            errors.append('invalid or mixed stratum')
            continue
        if row['occupancy_bin'] not in range(3) or (row['pure_roll'] and mode != 'front'):
            errors.append('invalid occupancy or pure-roll assignment')
        signature = (row['source_index'], row['view_index'], row['condition'])
        if signature in signatures:
            errors.append('duplicate schedule entry')
        signatures.add(signature)
        kinds[kind] += 1
        source_totals[row['source_index']] += 1
        if kind != 'photo':
            keeps[row['source_index']] += 1
            if kind == 'digital' and level != 'clear':
                errors.append('digital originals must stay clear')
            continue
        photos[row['source_index']] += 1
        modes[mode] += 1
        levels[level] += 1
        joint[mode, row['occupancy_bin']] += 1
        if mode == 'hard' and level not in ('clear', 'light'):
            errors.append('hard geometry combined with strong appearance')
        pairs[row['source_index'], row['view_index']].append(row)
    n = len(expected)
    if split == 'train':
        if len(rows) != 80000 or dict(kinds) != {**KEEP_COUNTS, 'photo': 64000}:
            errors.append('training kind budget differs from 80000-view protocol')
        if dict(modes) != PHOTO_COUNTS or dict(levels) != PHOTO_LEVEL_COUNTS:
            errors.append('training geometry/appearance quotas changed')
        if any(keeps[i] < 1 or photos[i] < 1 for i in expected):
            errors.append('source missing keep or photo coverage')
        if source_totals and (max(source_totals.values()) - min(source_totals.values()) > 1):
            errors.append('training source exposure is not balanced')
        if len({row['view_index'] for row in rows}) != len(rows):
            errors.append('training view ordinals must be unique')
    else:
        per_source = 11 if split == 'dev' else 10
        expected_kinds = {'appearance': n, 'rectified': n, 'photo': n * 8}
        if split == 'dev':
            expected_kinds['digital'] = n
        if dict(kinds) != expected_kinds or any(source_totals[i] != per_source for i in expected):
            errors.append('evaluation source/view budget changed')
        if any(modes[mode] != 2 * n for mode in MODES):
            errors.append('evaluation geometry strata are not balanced')
        for pair in pairs.values():
            if len(pair) != 2 or {r['condition'] for r in pair} != {'clear', 'noisy'}:
                errors.append('evaluation photograph missing clear/noisy pair')
                continue
            a, b = pair
            if any(a[key] != b[key] for key in ('source_index', 'view_index', 'mode', 'occupancy_bin', 'pure_roll', 'domain')):
                errors.append('paired evaluation geometry differs')
            if any((r['condition'] == 'clear') != (r['level'] == 'clear') for r in pair):
                errors.append('paired evaluation appearance mislabeled')
    for mode in MODES:
        divisor = 1 if split == 'train' else 2
        target = _quotas(modes[mode] // divisor)
        if [joint[mode, b] for b in range(3)] != [divisor * q for q in target]:
            errors.append('geometry/occupancy joint quota changed')
    front = [row for row in rows if row['kind'] == 'photo' and row['mode'] == 'front']
    expected_rotated = len(front) // 2 if split == 'train' else 2 * (n // 2)
    if sum(row['pure_roll'] for row in front) != expected_rotated:
        errors.append('front/rotation subquota changed')
    if errors:
        raise ValueError('Invalid unified schedule: ' + '; '.join(sorted(set(errors))))
    return {'split': split, 'sources': n, 'views': len(rows),
            'kind_counts': dict(kinds), 'photo_mode_counts': dict(modes),
            'photo_appearance_counts': dict(levels),
            'photo_occupancy_by_mode': {mode: {str(b): joint[mode, b] for b in range(3)} for mode in MODES},
            'source_exposures_min': min(source_totals.values()),
            'source_exposures_max': max(source_totals.values()),
            'sources_with_keep': len(keeps), 'sources_with_photo': len(photos),
            'front_pure_rotation_views': expected_rotated,
            'geometry_pair_count': len(pairs) if split != 'train' else 0,
            'actual_rendering_verified': False}
