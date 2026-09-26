"""Source-complete round sampling and split-bound portable background assets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .data import read_rgb
from .partial_synthesis import sample

STEP2000_RECIPE = 'ptrn-step2000-unified-v1'


def sha256(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def recipe(source_position, view, *, round_index, seed, domain, version='partial-v3'):
    """Independent streams for background, integrity, pre-crop and geometry.

    A balanced keep/occupancy schedule applies across each five-source block.
    Remaining recipe proportions are probabilistic and audited, never relabelled.
    """
    if min(source_position, view, round_index, seed) < 0 or domain not in ('train', 'dev'):
        raise ValueError('Invalid sampling key')
    if version not in ('partial-v3', STEP2000_RECIPE):
        raise ValueError('Unknown recipe version')
    if version == STEP2000_RECIPE and domain != 'train':
        raise ValueError('New recipe is train-only; frozen development scenes use partial-v3')
    n = source_position*16+view
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_index, source_position, view, domain == 'dev']))
    if n % 5 == 0:
        return ({'kind': 'keep', 'scene_version': STEP2000_RECIPE}
                if version == STEP2000_RECIPE else {'kind': 'keep'})
    if version == STEP2000_RECIPE:
        # Independent named streams prevent retries/appearance changes from
        # changing the requested occupancy, defects, background or medium.
        def stream(channel):
            return np.random.default_rng(np.random.SeedSequence(
                [seed, round_index, source_position, view, 20260925, channel]))
        occupancy_index = int(stream(1).choice(5, p=[.20, .30, .35, .10, .05]))
        occupancy_ranges = ((.90, .98), (.75, .90), (.50, .75), (.30, .50), (.10, .30))
        modes = ((('front',), (1.,)),
                 (('front', 'mild'), (.75, .25)),
                 (('front', 'mild', 'medium'), (.40, .50, .10)),
                 (('mild', 'medium', 'hard'), (.50, .40, .10)),
                 (('medium', 'hard'), (.50, .50)))
        mode = str(stream(2).choice(modes[occupancy_index][0],
                                    p=modes[occupancy_index][1]))
        missing_index = int(stream(3).choice(4, p=[.80, .12, .06, .02]))
        missing_ranges = (None, (.01, .03), (.03, .08), (.08, .15))
        missing_range = missing_ranges[missing_index]
        missing_target = (None if missing_range is None else
                          float(stream(11).uniform(missing_range[0]+.004,
                                                   missing_range[1]-.004)))
        family = str(stream(4).choice(
            ['viewer_screen', 'coco_context', 'near_color', 'mixed_color'],
            p=[.40, .25, .25, .10]))
        material = str(stream(5).choice(['screen', 'film']))
        severity = str(stream(6).choice(
            ['clear', 'light', 'medium', 'strong'], p=[.20, .60, .175, .025]))
        blend_tier = int(stream(7).choice(3, p=[.70, .25, .05]))
        edge_blend = (0. if blend_tier == 0 else
                      float(stream(8).uniform(.5, 2.) if blend_tier == 1 else
                            stream(8).uniform(2., 3.)))
        return {'kind': 'photo', 'scene_version': STEP2000_RECIPE,
                'occupancy_range': occupancy_ranges[occupancy_index],
                'occupancy_bin_v2': occupancy_index,
                'integrity': ('complete', 'light', 'moderate', 'severe')[missing_index],
                'missing_range': missing_range,
                'missing_target': missing_target,
                'family': family, 'source_crop': bool(stream(9).random() < .125),
                'source_crop_area_loss': (.05, .10),
                'mode': mode, 'pure_roll': False, 'level': 'clear',
                'capture_material': material, 'capture_severity': severity,
                'capture_seed': int(stream(10).integers(0, 2**32)),
                'edge_blend_px224': edge_blend}
    mode = str(rng.choice(['front', 'mild', 'medium', 'hard'], p=[.35, .45, .15, .05]))
    return {'kind': 'photo', 'integrity': str(rng.choice(['complete', 'light', 'moderate'], p=[.6, .3, .1])),
            'family': str(rng.choice(['viewer_screen', 'coco_context', 'near_color', 'mixed_color'], p=[.4, .25, .25, .1])),
            'source_crop': bool(rng.random() < .25), 'high_occupancy': n % 20 != 1,
            'mode': mode, 'pure_roll': bool(mode == 'front' and rng.random() < .7),
            'level': str(rng.choice(['clear', 'light'], p=[.35, .65]))}


class PartialDataset(Dataset):
    def __init__(self, bundle, backgrounds, *, split='train', size=224, seed=20260925,
                 round_index=0, views=None, verify=True, recipe_version='partial-v3'):
        self.bundle, self.backgrounds = Path(bundle), Path(backgrounds)
        self.split, self.size, self.seed, self.round_index = split, size, seed, round_index
        if recipe_version not in ('partial-v3', STEP2000_RECIPE):
            raise ValueError('Unknown recipe version')
        if split == 'dev' and recipe_version != 'partial-v3':
            raise ValueError('Frozen development scenes require partial-v3')
        self.recipe_version = recipe_version
        if split not in ('train', 'dev') or size not in (224, 384) or (split == 'dev' and round_index != 0):
            raise ValueError('Unsupported split/input; development scenes must stay fixed')
        self.views = (16 if split == 'train' else 12) if views is None else int(views)
        if self.views < 1: raise ValueError('Positive views required')
        manifest = self.bundle/'data/manifest.json'
        splitting = self.bundle/'data/split.json'
        bg_manifest = self.bundle/'data/backgrounds-manifest.json'
        self.records = json.loads(manifest.read_text())['records']
        splits = json.loads(splitting.read_text())
        flattened = [i for name in ('train', 'dev', 'quarantine') for i in splits[name]]
        if (len(flattened) != len(set(flattened)) or sorted(flattened) != list(range(len(self.records)))):
            raise ValueError('Film split overlap, duplicate or missing source')
        for name, ids in splits.items():
            for i in ids:
                if self.records[i]['split'] != name: raise ValueError('Manifest/split membership mismatch')
        self.ids = splits[split]
        all_bg = json.loads(bg_manifest.read_text())['rows']
        seen = {}
        for row in all_bg:
            for key in ('id', 'sha256', 'pixel_sha256'):
                value = row.get(key)
                if value:
                    if (key, value) in seen and seen[key, value] != row['split']:
                        raise ValueError('Cross-split background identity/content overlap')
                    seen[key, value] = row['split']
        self.bg = [r for r in all_bg if r['split'] == split and r['source_kind'] == 'real_photo']
        if not self.ids or not self.bg: raise ValueError('Missing film/background split')
        if verify:
            for i in self.ids:
                row = self.records[i]
                if sha256(self.safe_path(self.bundle, row['source_path'])) != row['source_sha256']:
                    raise ValueError('Film asset SHA mismatch')
            for row in self.bg:
                if sha256(self.safe_path(self.backgrounds, row['path'])) != row['sha256']:
                    raise ValueError('Background asset SHA mismatch')
        self.fingerprint = {'manifest': sha256(manifest), 'split': sha256(splitting),
                            'background_manifest': sha256(bg_manifest), 'split_name': split,
                            'size': size, 'seed': seed, 'views': self.views}
        if recipe_version == STEP2000_RECIPE:
            self.fingerprint['recipe_version'] = STEP2000_RECIPE

    @staticmethod
    def safe_path(root, relative):
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()): raise ValueError('Asset path escapes bundle')
        return path

    def __len__(self): return len(self.ids)*self.views

    def render(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        position, view = divmod(index, self.views)
        source_index = self.ids[position]
        options = recipe(position, view, round_index=self.round_index, seed=self.seed,
                         domain=self.split, version=self.recipe_version)
        background, bg_id = None, None
        if options.get('family') == 'coco_context':
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.round_index, source_index, view, 891]))
            row = self.bg[int(rng.integers(len(self.bg)))]
            background = read_rgb(self.safe_path(self.backgrounds, row['path']))
            bg_id = row['id']
        source = read_rgb(self.safe_path(self.bundle, self.records[source_index]['source_path']))
        # Full source is never cached/transformed in place. Retry only geometry,
        # preserving all requested strata; failures are counted in the sample.
        for attempt in range(20):
            try:
                result = sample(source, source_index, view+self.round_index*self.views,
                                seed=self.seed+attempt*1000003, domain=self.split,
                                background=background, **options)
                result.info.update({'background_id': bg_id, 'round': self.round_index,
                                    'geometry_retries': attempt, 'dataset_index': index})
                return result
            except RuntimeError:
                if attempt == 19: raise

    def __getitem__(self, index):
        return self.render(index).input(self.size)
