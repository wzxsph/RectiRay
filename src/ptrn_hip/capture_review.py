"""Build the local 100-scene capture review without loading any model."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .capture_synthesis import VERSION, LABELS, recipe, render
from .data import read_rgb
from .partial_synthesis import sample, pre_crop, _background
from .spatial_synthesis import sample as complete_sample
from .ratio_synthesis import letterbox, project, pixel_corners


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def assignments(rows, seed=2026092519):
    """Balanced background counts; all assignment RNG is independent of geometry."""
    rng = np.random.default_rng(seed)
    result = {r['id']: recipe('keep', 'clear', seed+i) for i, r in enumerate(rows) if r['options']['kind'] == 'keep'}
    groups = {'screen': [], 'film': []}
    photos = [r for r in rows if r['options']['kind'] == 'photo']
    for family in sorted({r['options']['family'] for r in photos}):
        ids = [r['id'] for r in photos if r['options']['family'] == family]
        rng.shuffle(ids)
        if len(ids) % 2: raise ValueError('Expected even background-family counts')
        groups['screen'].extend(ids[:len(ids)//2]); groups['film'].extend(ids[len(ids)//2:])
    for material, ids in groups.items():
        if len(ids) != 40: raise ValueError('Expected 40 photos per material')
        rng.shuffle(ids)
        surface = (['moire_gray', 'moire_color', 'bands', 'window', 'lamp', 'highlight'] if material == 'screen'
                   else ['lightbox', 'haze', 'window', 'lamp', 'highlight', 'lightbox'])
        camera = ['defocus', 'motion', 'noise', 'exposure', 'white_balance', 'jpeg']
        medium = [[surface[i % 6], camera[i % 6], camera[(i+2) % 6]] for i in range(7)]
        strong = ['moire_color', 'highlight', 'motion'] if material == 'screen' else ['haze', 'highlight', 'motion']
        for i, ident in enumerate(ids):
            severity = 'clear' if i < 8 else 'light' if i < 32 else 'medium' if i < 39 else 'strong'
            if severity == 'clear': effects = []
            elif severity == 'light':
                j = i-8
                effects = [surface[j % 6]] if j < 12 else [surface[j % 6], camera[(j//6+j) % 6]]
            elif severity == 'medium': effects = medium[i-32]
            else: effects = strong
            result[ident] = recipe(material, severity, seed+int(ident.split('-')[1])*101, effects)
    return result


def reconstruct(source, row, background, material):
    """Regenerate legacy labels, then remove appearance without resampling crops."""
    idx, ordinal, seed = row['source_index'], int(row['id'].split('-')[1])-1, row['scene_seed']
    opts = row['options']
    old = sample(source, idx, ordinal, seed=seed, domain='dev', background=background, **opts)
    np.testing.assert_array_equal(old.quad, row['quad_gt'])
    if old.info != row['info']: raise AssertionError('Saved scene metadata drift')
    if opts['kind'] == 'keep':
        return source.copy(), old, np.ones(source.shape[:2], np.float32), 'original'
    rng = np.random.default_rng(np.random.SeedSequence([seed, idx, ordinal, 913, True]))
    plane, crop = pre_crop(source, rng, opts['source_crop'])
    assert crop == old.info['source_pre_crop']
    base = complete_sample(plane, idx, ordinal, seed=seed, domain='dev', kind='photo', mode=opts['mode'],
                           pure_roll=opts['pure_roll'], occupancy_bin=1 if opts['high_occupancy'] else 0, level='clear')
    photo = base.clean.copy()
    if opts['family'] != 'viewer_screen':
        exterior = _background(plane, rng, photo.shape[:2], opts['family'], background)
        h, w = photo.shape[:2]
        warped = cv2.warpPerspective(plane, np.array(base.info['source_to_photo']), (w, h), flags=cv2.INTER_LINEAR)
        photo = np.clip(warped.astype(np.float32)+exterior*(1-base.film_mask[..., None]), 0, 255).astype(np.uint8)
        # Image-filling display: only its visible image surface is modeled.
        surface = base.film_mask.copy()
        surface_kind = 'image_filling_display' if material == 'screen' else 'film'
    elif material == 'screen':
        surface = base.screen_mask.copy(); surface_kind = 'display_with_viewer'
    else:
        # Keep the scene geometry, but a physical film must not inherit toolbar
        # buttons. Reinterpret the surrounding display rectangle as a lightbox.
        frame = np.clip(base.screen_mask-base.film_mask, 0, 1)[..., None]
        photo = np.rint(photo*(1-frame)+np.array([224, 224, 219])*frame).astype(np.uint8)
        surface = base.film_mask.copy(); surface_kind = 'film_on_lightbox'
    x, y, right, bottom = old.info['post_photo_crop']
    photo = photo[y:bottom, x:right].copy()
    surface = surface[y:bottom, x:right].copy()
    np.testing.assert_array_equal(base.film_mask[y:bottom, x:right], old.film_mask)
    assert photo.shape == old.photo.shape
    np.testing.assert_allclose(project(pixel_corners(*old.info['plane_size']), old.transform), old.quad, atol=1e-5)
    return photo, old, surface, surface_kind


def build(args):
    cv2.setNumThreads(1)
    out = args.output.resolve()
    if out.exists(): raise FileExistsError('Choose a fresh output directory; old galleries are immutable')
    rows = [r for r in json.loads(args.report.read_text())['rows'] if r['group'] == 'synthetic']
    assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
    specs = assignments(rows)
    records = json.loads((args.bundle/'data/manifest.json').read_text())['records']
    backgrounds = {r['id']: r for r in json.loads((args.bundle/'data/backgrounds-manifest.json').read_text())['rows']}
    protected = [args.report, args.report.parent/'index.html', args.report.parent/'gallery-data.js',
                 args.bundle/'data/manifest.json', args.bundle/'data/backgrounds-manifest.json',
                 *[Path(__file__).with_name(n+'.py') for n in ['partial_synthesis', 'spatial_synthesis', 'ratio_synthesis']]]
    before = {str(p): sha(p) for p in protected}
    (out/'images').mkdir(parents=True); (out/'geometry').mkdir()

    def save_image(path, pixels):
        assert cv2.imwrite(str(out/path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        np.testing.assert_array_equal(read_rgb(out/path), pixels)

    def prepare(row):
        ident = row['id']; spec = specs[ident]
        rec = records[row['source_index']]
        assert rec['split'] == 'dev', 'These reused scenes must remain exposed development data'
        source_path = args.bundle/rec['source_path']
        assert sha(source_path) == row['source_sha256']
        source = read_rgb(source_path); bg = None; bg_path = None
        if row['background_id']:
            b = backgrounds[row['background_id']]; bg_path = args.backgrounds/b['path']
            assert b['split'] == 'dev' and sha(bg_path) == b['sha256']
            bg = read_rgb(bg_path)
        clean, old, surface, surface_kind = reconstruct(source, row, bg, spec['material'])
        # Verify the old gallery's own pixels still regenerate exactly.
        h, w = old.photo.shape[:2]; scale = min(1, 1400/max(h, w))
        preview = cv2.resize(old.photo, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else old.photo
        np.testing.assert_array_equal(preview, read_rgb(args.report.parent/row['input']))
        plane_size = old.info.get('plane_size', [w, h])
        changed, appearance = render(clean, surface, old.transform, plane_size, spec)
        assert np.array_equal(old.quad, row['quad_gt'])
        files = {k: f'images/{ident}-{k}.png' for k in ['clean', 'capture', 'input224', 'clean224', 'surface']}
        input224, lb = letterbox(changed, 224)
        for key, pixels in [('clean', clean), ('capture', changed), ('input224', input224),
                            ('clean224', letterbox(clean, 224)[0]), ('surface', np.repeat(np.rint(surface[..., None]*255).astype(np.uint8), 3, axis=2))]:
            save_image(files[key], pixels)
        geometry_file = f'geometry/{ident}.npz'
        np.savez_compressed(out/geometry_file, film_mask=old.film_mask, surface_mask=surface, quad=old.quad,
                            transform=old.transform, ratio=np.float64(old.ratio), letterbox=lb)
        if spec['material'] == 'keep':
            np.testing.assert_array_equal(changed, source)
            np.testing.assert_array_equal(clean, source)
        assert sha(source_path) == row['source_sha256']
        if bg_path: assert sha(bg_path) == b['sha256']
        return {**{k: row[k] for k in ['id', 'options', 'source_index', 'source_sha256', 'background_id', 'scene_seed', 'quad_gt', 'info']},
                'material': spec['material'], 'severity': spec['severity'], 'appearance': appearance,
                'surface_kind': surface_kind, 'width': w, 'height': h, 'ratio': old.ratio,
                'occupancy': old.info['occupancy'], 'missing': old.info['missing_fraction'],
                'files': files, 'geometry_file': geometry_file,
                'asset_sha256': {p: sha(out/p) for p in [*files.values(), geometry_file]}}

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(prepare, rows):
            completed.append(r)
            if len(completed) % 10 == 0: print(f'Validated {len(completed)}/100', flush=True)
    assert before == {str(p): sha(p) for p in protected}
    counts = Counter((r['material'], r['severity']) for r in completed)
    assert counts[('keep', 'clear')] == 20
    for material in ['screen', 'film']:
        assert [counts[(material, s)] for s in ['clear', 'light', 'medium', 'strong']] == [8, 24, 7, 1]
    photos = [r for r in completed if r['material'] != 'keep']
    assert min(r['occupancy'] for r in photos) >= .3 and sum(r['occupancy'] > .5 for r in photos) >= 64
    audit = {'version': VERSION, 'count': 100, 'old_pixels_and_geometry_verified': 100,
             'unchanged_protected_sha256': before, 'originals_pixel_exact': 20,
             'png_roundtrips_exact': 500, 'native_and_224_active_effects': sum(len(r['appearance']['effects']) for r in completed),
             'quotas': {f'{m}/{s}': n for (m, s), n in counts.items()},
             'photo_occupancy_above_half': sum(r['occupancy'] > .5 for r in photos),
             'photo_occupancy_min': min(r['occupancy'] for r in photos),
             'effect_background_coverage': {e: dict(Counter(r['options']['family'] for r in completed if any(x['name'] == e for x in r['appearance']['effects']))) for e in LABELS},
             'no_training_or_model_inference': True, 'no_visual_quality_assessment': True,
             'renderer_sha256': sha(Path(__file__).with_name('capture_synthesis.py')),
             'builder_sha256': sha(Path(__file__)),
             'dependencies': {'numpy': np.__version__, 'opencv': cv2.__version__}}
    assert all(sum(audit['effect_background_coverage'][e].get(f, 0) for e in ['moire_gray', 'moire_color']) > 0
               for f in ['coco_context', 'near_color', 'mixed_color'])
    data = {'version': VERSION, 'labels': LABELS, 'rows': completed, 'audit': audit}
    (out/'manifest.local.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    (out/'technical-check.local.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    (out/'gallery-data.js').write_text('window.CAPTURE='+json.dumps(data, ensure_ascii=False).replace('</', '<\\/')+';\n')
    (out/'index.html').write_text(Path(__file__).with_name('capture_review.html').read_text())
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--backgrounds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('Use 1–8 bounded workers')
    build(args)


if __name__ == '__main__': main()
