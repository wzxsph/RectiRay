"""Frozen-candidate evaluation, local export and anonymous real-photo review.

No application integration, uploads or original-image modifications occur here.
D may choose a model and tolerances; T is opened only after selection-lock.json.
All artifacts are restricted to an ignored .local run directory. Imports of GPU
model code are lazy so development calibration and report refresh work on CPU.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .data import read_rgb, write_json as _write_json
from .ratio_data import Sources, sha256
from .ratio_synthesis import letterbox, pixel_corners, project
from .unified_geometry import (decode_prediction, edge_ratio, plan_transform,
                               apply_transform, inverse_letterbox_matrix)
from .unified_evaluation import (prepare_reference, measure_geometry, summarize_rows,
                                 admission_gates, selection_score)

GROUPS = ('S224', 'F224', 'J224', 'J384')
THRESHOLD_GRID = tuple({'shape': shape, 'ratio': ratio, 'border': .005,
                        'rotation_degrees': .5, 'ambiguity_fraction': .1}
                       for shape in (.0025, .005, .01) for ratio in (.005, .01, .02))
PHOTO_WEIGHTS = {'front': .45, 'mild': .35, 'medium': .15, 'hard': .05}


def write_json(path, value):
    """Fail closed on any attempted JSON write to an image or outside .local."""
    destination = Path(path)
    if destination.suffix != '.json' or '.local' not in destination.resolve().parts:
        raise ValueError('JSON artifact writes require a .local JSON destination')
    _write_json(destination, value)


def _directory(args):
    run = Path(args.run)
    if '.local' not in run.resolve().parts:
        raise ValueError('Finalization artifacts must remain under .local')
    output = run / 'final'
    output.mkdir(parents=True, exist_ok=True)
    return output


def _load(path):
    return json.loads(Path(path).read_text())


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _load(path) != value:
            raise RuntimeError('Frozen finalization record differs; no overwrite allowed')
    else:
        write_json(path, value)
    return value


def _validate_lock(args, selection):
    from .unified_runner import assert_source
    output = _directory(args)
    saved = _load(output / 'selection-lock.json')
    if selection != saved:
        raise RuntimeError('Selection arguments differ from the immutable lock')
    if sha256(Path(args.run) / selection['checkpoint']) != selection['checkpoint_sha256']:
        raise RuntimeError('Frozen selected checkpoint changed')
    if sha256(output / 'D-calibration.json') != selection['development_calibration_sha256']:
        raise RuntimeError('Development calibration changed after selection freeze')
    if sha256(Path(args.run) / 'sealed/test.json') != selection['test_schedule_sha256']:
        raise RuntimeError('Sealed test schedule changed after selection freeze')
    if sha256(args.manifest) != selection['manifest_sha256']:
        raise RuntimeError('Source manifest changed after selection freeze')
    if assert_source(args.run)['tree_sha256'] != selection['source_tree_sha256']:
        raise RuntimeError('Source snapshot changed after selection freeze')


def _status(args, phase, **values):
    from .unified_runner import stamp
    stamp(Path(args.run) / 'status.json', phase=phase, **values)
    print(json.dumps({'phase': phase, **values}), flush=True)


def _verify_training(args):
    from .unified_runner import assert_source
    inventory = assert_source(args.run)
    for group in GROUPS:
        config = _load(Path(args.run) / group / 'config.json')
        status = _load(Path(args.run) / group / 'status.json')
        if status.get('phase') != 'complete' or status.get('optimizer_step') != config['updates_this_group']:
            raise RuntimeError('All four approved training budgets must finish before finalization')
    return inventory


def _metadata_size(metadata, size):
    result = []
    for record in metadata:
        inverse = inverse_letterbox_matrix(record['photo_size'], size)
        result.append({**record, 'inverse_letterbox': inverse.tolist(),
                       'quad_input': project(np.asarray(record['quad_photo']), np.linalg.inv(inverse)).tolist()})
    return result


def _references(metadata):
    return [prepare_reference(row['photo_size'], row['source_size'], row['source_to_photo'], row['quad_photo'])
            for row in metadata]


def _stats(values):
    a = np.asarray(values, float)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())} if len(a) else {'count': 0}


def _baselines(args, metadata):
    splits = _load(Path(args.run) / 'split.local.json')
    sources = Sources(args.manifest, cache_size=1)
    ratios = [sources.records[index]['original_size'][0] / sources.records[index]['original_size'][1]
              for index in splits['train']]
    mean_log = float(np.log(ratios).mean())
    groups = defaultdict(lambda: {'gt_corner_edge_ratio_error': [], 'training_mean_ratio_error': []})
    for row in metadata:
        mode = row['mode'] if row['kind'] == 'photo' else 'preserve_' + row['kind']
        key = f"{row.get('condition', 'noisy')}|{mode}"
        groups[key]['gt_corner_edge_ratio_error'].append(float(np.expm1(abs(np.log(edge_ratio(np.asarray(row['quad_photo']))) - row['log_ratio']))))
        groups[key]['training_mean_ratio_error'].append(float(np.expm1(abs(mean_log - row['log_ratio']))))
    return {'training_sources': len(ratios), 'training_mean_log_ratio': mean_log,
            'training_mean_ratio': float(np.exp(mean_log)),
            'target': 'full_source_canvas_W/H_including_original_borders',
            'strata': {key: {metric: _stats(values) for metric, values in entry.items()} for key, entry in groups.items()}}


def _cluster_intervals(rows, repetitions=400):
    """Source-cluster bootstrap of means; never persist source identifiers."""
    groups = defaultdict(list)
    for row in rows:
        if 'source_index' in row:
            groups[row['source_index']].append(row)
    if not groups:
        return {'source_count': 0, 'method': 'unavailable'}
    keys = ('corner_nme', 'ratio_error', 'success')
    sums = np.array([[sum(float(row.get(key, 0)) for row in group) for key in keys] for group in groups.values()])
    counts = np.array([len(group) for group in groups.values()])
    rng = np.random.default_rng(20260922)
    estimates = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), len(groups))
        estimates.append(sums[selected].sum(axis=0) / counts[selected].sum())
    bounds = np.percentile(estimates, [2.5, 97.5], axis=0)
    return {'source_count': len(groups), 'repetitions': repetitions,
            'method': 'source_cluster_bootstrap_95pct_interval_of_row_mean_not_patient_independent',
            'intervals': {key: [float(bounds[0, index]), float(bounds[1, index])] for index, key in enumerate(keys)}}


def _photo_weighted(rows):
    output = {}
    for condition in sorted({row.get('condition') for row in rows if row.get('condition')}):
        groups = {mode: [row for row in rows if row.get('condition') == condition and row.get('geometry') == mode]
                  for mode in PHOTO_WEIGHTS}
        if not all(groups.values()):
            continue
        output[condition] = {'photo_only': True, 'weights': PHOTO_WEIGHTS,
            'success_rate': sum(PHOTO_WEIGHTS[mode] * np.mean([row.get('success', False) for row in group]) for mode, group in groups.items()),
            'corner_nme_mean': sum(PHOTO_WEIGHTS[mode] * np.mean([row['corner_nme'] for row in group]) for mode, group in groups.items()),
            'ratio_error_mean': sum(PHOTO_WEIGHTS[mode] * np.mean([row['ratio_error'] for row in group]) for mode, group in groups.items())}
    return output


def calibrate_development(args):
    """Nine tolerances per already-selected group checkpoint; D only, no GPU."""
    from .unified_runner import measure_predictions
    _verify_training(args)
    output = _directory(args)
    if (output / 'selection-lock.json').exists():
        return _load(output / 'D-calibration.json')
    metadata = _load(Path(args.run) / 'cache/dev/metadata.json')
    if len(metadata) != 5258:
        raise RuntimeError('Development calibration requires all 5258 views')
    refs = _references(metadata)
    baseline = _load(Path(args.run) / 'G0-D.json')
    report = {'scope': 'D_only_exposed_source_development', 'groups': {},
              'tolerance_grid': THRESHOLD_GRID, 'baselines': _baselines(args, metadata)}
    for group in GROUPS:
        best = _load(Path(args.run) / group / 'best.json')
        checkpoint = Path(args.run) / group / best['checkpoint']
        sidecar = _load(checkpoint.with_suffix('.json'))
        if sha256(checkpoint) != sidecar['sha256']:
            raise RuntimeError('Selected development checkpoint SHA mismatch')
        size = 384 if group == 'J384' else 224
        predictions_path = Path(args.run) / group / f"predictions-{best['step']:04d}.npy"
        predictions = np.load(predictions_path)
        if predictions.shape != (5258, 9):
            raise RuntimeError('Development prediction cache has wrong shape')
        sized = _metadata_size(metadata, size)
        trials = []
        for index, thresholds in enumerate(THRESHOLD_GRID):
            _status(args, 'D_threshold_calibration', group=group, trial=index + 1, total=9)
            summary, rows = measure_predictions(predictions, sized, size, thresholds, refs)
            gates = admission_gates(summary, baseline)
            # Hard content/geometry failures precede completion for research-only fallbacks.
            content_failed = sum(key.startswith(('content:', 'final_grid:', 'preserve:', 'front:', 'unnecessary_', 'undercorrected')) for key in gates['failed'])
            rank = [int(not gates['eligible']), content_failed,
                    -summary['all']['automatic_success_rate'], -summary['all']['automatic_rate'], index]
            trial = {'index': index, 'thresholds': thresholds, 'summary': summary,
                     'gates': gates, 'rank': rank,
                     'source_cluster_intervals': _cluster_intervals(rows), 'photo_weighted': _photo_weighted(rows)}
            trials.append(trial)
        selected = min(trials, key=lambda row: tuple(row['rank']))
        report['groups'][group] = {'group': group, 'step': best['step'], 'input_size': size,
            'checkpoint': f"{group}/{best['checkpoint']}", 'checkpoint_sha256': sha256(checkpoint),
            'predictions_sha256': sha256(predictions_path), 'selected_trial': selected['index'],
            'thresholds': selected['thresholds'], 'summary': selected['summary'], 'gates': selected['gates'],
            'selection_score': list(selection_score(selected['summary'], baseline)), 'trials': trials}
    write_json(output / 'D-calibration.json', report)
    return report


def _mild_tail(candidate):
    values = [value['ratio_error']['p90'] for key, value in candidate['summary']['condition_geometry'].items() if key.endswith('|mild')]
    return max(values) if len(values) == 2 and all(value is not None for value in values) else 1e6


def _real_files(args):
    directory = getattr(args, 'real_dir', None)
    if directory is None:
        raise ValueError('The explicit authorized 47-photo directory is required')
    directory = Path(directory).resolve()
    run = Path(args.run).resolve()
    if directory == run or directory in run.parents or run in directory.parents:
        raise ValueError('Read-only photo directory and artifact run must be disjoint')
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if len(paths) != 47:
        raise ValueError('Frozen real-photo observation requires exactly 47 images')
    return paths


def _export(args, candidate):
    from .unified_export import export_candidate, load_export
    output = _directory(args) / 'exports' / candidate['group']
    report_path = output / 'export-report.json'
    if report_path.exists():
        report = _load(report_path)
        if report['metadata']['checkpoint_sha256'] != candidate['checkpoint_sha256']:
            raise RuntimeError('Existing export belongs to a different checkpoint')
        load_export(output / report['artifact'], expected_sha256=report['artifact_sha256'], size=candidate['input_size'])
        return report
    if output.exists():
        if output.is_symlink() or _directory(args).resolve() not in output.resolve().parents:
            raise RuntimeError('Unsafe incomplete export destination')
        # Preserve interrupted local artifacts and create a fresh attempt; never
        # delete or rename source/checkpoint/real-photo directories.
        output.rename(output.with_name(output.name + f'.incomplete-{time.time_ns()}'))
    inventory = _load(Path(args.run) / 'source-inventory.json')
    _status(args, 'CPU_export', group=candidate['group'])
    return export_candidate(args.weights, args.g0_checkpoint,
                            Path(args.run) / candidate['checkpoint'], output,
                            inventory['tree_sha256'], getattr(args, 'real_dir', None))


def _predict_script(scripted, pixels):
    from .model import torch
    with torch.no_grad():
        return scripted(torch.tensor(pixels[None], dtype=torch.float32)).detach().cpu().numpy()[0]


def _runtime(scripted, image, size, thresholds):
    started = time.perf_counter()
    from .model import torch
    pixels, lb = letterbox(image, size)
    tensor = torch.tensor(pixels[None], dtype=torch.float32)
    with torch.no_grad():
        before_forward = time.perf_counter()
        raw = scripted(tensor)
        after_forward = time.perf_counter()
    prediction = raw.detach().cpu().numpy()[0]
    photo_size = (image.shape[1], image.shape[0])
    decoded = decode_prediction(prediction, size, photo_size, np.linalg.inv(lb))
    plan = (plan_transform(photo_size, decoded['quad'], decoded['ratio'], thresholds)
            if decoded['valid'] else {'action': 'needs_confirmation', 'valid': False,
                'matrix': None, 'output_size': photo_size, 'reason': ','.join(decoded['reasons'])})
    result = apply_transform(image, plan)
    return result, decoded, plan, (after_forward - before_forward) * 1000, (time.perf_counter() - started) * 1000


def _benchmark(args, candidate):
    """Same 47 photos, 3 warmups then one pass at each thread count."""
    from .model import torch
    from .unified_export import load_export
    folder = _directory(args) / 'exports' / candidate['group']
    report_path = folder / 'cpu47.json'
    if report_path.exists():
        saved = _load(report_path)
        if saved.get('thresholds') != candidate['thresholds'] or saved.get('checkpoint_sha256') != candidate['checkpoint_sha256']:
            raise RuntimeError('Existing CPU benchmark has different model or tolerances')
        return saved
    report = _export(args, candidate)
    model, _ = load_export(folder / report['artifact'])
    files = _real_files(args)
    originals = [(photo_path, sha256(photo_path)) for photo_path in files]
    previous_threads, previous_cv = torch.get_num_threads(), cv2.getNumThreads()
    result = {'checkpoint_sha256': candidate['checkpoint_sha256'], 'thresholds': candidate['thresholds'],
              'same_real_photo_count': 47, 'warmups': 3, 'disk_read_included': False,
              'full_pipeline': 'original_RGB_letterbox_tensor_forward_decode_plan_single_resample', 'threads': {}}
    try:
        cv2.setNumThreads(1)
        for threads in (2, 4):
            torch.set_num_threads(threads)
            _status(args, 'CPU47_benchmark', group=candidate['group'], threads=threads)
            probe = read_rgb(files[0])
            for _ in range(3):
                _runtime(model, probe, candidate['input_size'], candidate['thresholds'])
            forward, total, actions = [], [], Counter()
            for photo_path in files:
                image = read_rgb(photo_path)
                _, _, plan, forward_ms, total_ms = _runtime(model, image, candidate['input_size'], candidate['thresholds'])
                forward.append(forward_ms); total.append(total_ms); actions[plan['action']] += 1
            result['threads'][str(threads)] = {'pure_inference_ms': _stats(forward),
                                              'full_pipeline_ms': _stats(total), 'action_counts': dict(actions)}
        result['sources_unchanged'] = all(sha256(photo_path) == before for photo_path, before in originals)
        if not result['sources_unchanged']:
            raise RuntimeError('Read-only real photo changed')
        four = result['threads']['4']['pure_inference_ms']
        result['cpu_gate'] = four['median'] <= 250 and four['p95'] <= 500
        write_json(report_path, result)
        return result
    finally:
        torch.set_num_threads(previous_threads); cv2.setNumThreads(previous_cv)


def freeze_selection(args, calibration=None):
    output = _directory(args)
    lock = output / 'selection-lock.json'
    if lock.exists():
        selection = _load(lock)
        _validate_lock(args, selection)
        return selection
    calibration = calibration or calibrate_development(args)
    candidates = calibration['groups']
    ranked_224 = sorted((candidates[group] for group in ('S224', 'F224', 'J224')),
                        key=lambda row: (not row['gates']['eligible'], *row['selection_score'], row['step'], row['group']))
    chosen = ranked_224[0]
    resolution = {'J384_D_eligible': candidates['J384']['gates']['eligible'],
                  'J224_mild_p90': _mild_tail(candidates['J224']), 'J384_mild_p90': _mild_tail(candidates['J384'])}
    resolution['at_least_10pct_tail_improvement'] = resolution['J384_mild_p90'] <= .9 * resolution['J224_mild_p90']
    if resolution['J384_D_eligible'] and resolution['at_least_10pct_tail_improvement']:
        resolution['J384_CPU'] = _benchmark(args, candidates['J384'])
        if resolution['J384_CPU']['cpu_gate']:
            chosen = candidates['J384']
    cpu = _benchmark(args, chosen)
    _real_files(args)
    selection = {key: chosen[key] for key in ('group', 'step', 'input_size', 'checkpoint', 'checkpoint_sha256', 'thresholds')}
    selection.update(D_eligible=chosen['gates']['eligible'], D_gates=chosen['gates'],
                     CPU_eligible=cpu['cpu_gate'], resolution_rule=resolution,
                     research_only=not (chosen['gates']['eligible'] and cpu['cpu_gate']),
                     manifest_sha256=sha256(args.manifest),
                     test_schedule_sha256=sha256(Path(args.run) / 'sealed/test.json'),
                     development_calibration_sha256=sha256(output / 'D-calibration.json'),
                     source_tree_sha256=_load(Path(args.run) / 'source-inventory.json')['tree_sha256'],
                     frozen_before_T=True, frozen_time_unix=time.time(), application_changed=False)
    return _immutable(lock, selection)


class _SealedDataset:
    def __init__(self, manifest, schedule, size):
        self.manifest, self.schedule, self.size = str(manifest), schedule, size
        self.positions = sorted(range(len(schedule)), key=lambda index: (schedule[index]['source_index'], index))
        self.sources = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        from .unified_runner import sample_record
        if self.sources is None:
            self.sources = Sources(self.manifest, cache_size=2)
        position = self.positions[index]
        pixels, _, _, _, info, image = sample_record(self.sources, self.schedule[position], self.size)
        baseline = letterbox(image, 224)[0] if self.size != 224 else pixels
        return position, pixels, baseline, json.dumps(info)


def evaluate_sealed(args, selection=None):
    from .unified_runner import read_schedule, loader, measure_predictions, actual_view_audit
    from .unified_model import build_unified, load_unified, predict
    from .model import torch
    selection = selection or freeze_selection(args)
    _validate_lock(args, selection)
    output = _directory(args)
    report_path = output / 'T-report.json'
    if report_path.exists():
        report = _load(report_path)
        if report['selection_sha256'] != sha256(output / 'selection-lock.json'):
            raise RuntimeError('T belongs to a different frozen candidate')
        return report
    if not (output / 'selection-lock.json').exists() or not selection.get('frozen_before_T'):
        raise RuntimeError('Candidate and thresholds must be locked before opening T')
    schedule_path = Path(args.run) / 'sealed/test.json'
    if sha256(schedule_path) != selection['test_schedule_sha256']:
        raise RuntimeError('Sealed scenario schedule changed')
    schedule = read_schedule(args.run, 'test', allow_test=True)
    if len(schedule) != 4780:
        raise RuntimeError('T requires all 4780 sealed views')
    device = getattr(args, 'device', 'cuda')
    model, _, _ = load_unified(args.weights, args.g0_checkpoint, Path(args.run) / selection['checkpoint'])
    g0, _, _ = build_unified(args.weights, args.g0_checkpoint, 224)
    model.to(device); g0.to(device)
    predictions, baseline_predictions = np.empty((4780, 9)), np.empty((4780, 9))
    metadata = [None] * 4780
    data = loader(_SealedDataset(args.manifest, schedule, selection['input_size']), 4 if selection['input_size'] == 384 else 16, args.workers)
    completed = 0
    for indices, pixels, baseline_pixels, encoded in data:
        candidate_batch = predict(model, pixels.numpy(), device)
        baseline_batch = predict(g0, baseline_pixels.numpy(), device)
        for index, candidate, baseline, info in zip(indices.numpy(), candidate_batch, baseline_batch, encoded, strict=True):
            predictions[index], baseline_predictions[index], metadata[index] = candidate, baseline, json.loads(info)
        completed += len(indices)
        if completed % 400 < len(indices) or completed == 4780:
            _status(args, 'sealed_T', completed=completed, total=4780)
    del model, g0, data
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    audit = actual_view_audit(metadata)
    for mode, bins in audit['actual_photo_occupancy_by_geometry'].items():
        total = sum(bins.values())
        if total and (bins.get(1, 0) + bins.get(2, 0)) / total < .80:
            raise RuntimeError('T per-geometry large-film quota violated')
    references = _references(metadata)
    summary, rows = measure_predictions(predictions, metadata, selection['input_size'], selection['thresholds'], references)
    baseline, _ = measure_predictions(baseline_predictions, _metadata_size(metadata, 224), 224, selection['thresholds'], references)
    np.save(output / 'T-predictions.local.npy', predictions)
    np.save(output / 'T-G0-predictions.local.npy', baseline_predictions)
    write_json(output / 'T-metadata.local.json', metadata)
    report = {'selection_sha256': sha256(output / 'selection-lock.json'),
              'scope': 'same_exposed_478_sources_new_sealed_scenes_NOT_independent_patient_test',
              'views': 4780, 'summary': summary, 'G0': baseline, 'gates': admission_gates(summary, baseline),
              'actual_data_audit': audit, 'baselines': _baselines(args, metadata),
              'source_cluster_intervals': _cluster_intervals(rows), 'photo_weighted': _photo_weighted(rows),
              'application_changed': False, 'T_used_for_selection': False}
    write_json(report_path, report)
    return report


def evaluate_regression(args, selection=None):
    """Frozen ratio-v1 images/seed recipe: 1434 clean + 1434 noisy."""
    from .ratio_runner import EvaluationSet, loader
    from .unified_model import build_unified, load_unified, predict
    from .model import torch
    selection = selection or freeze_selection(args)
    _validate_lock(args, selection)
    output = _directory(args)
    report_path = output / 'R-report.json'
    if report_path.exists():
        return _load(report_path)
    indices = _load(Path(args.run) / 'split.local.json')['dev']
    device, size = getattr(args, 'device', 'cuda'), selection['input_size']
    model, _, _ = load_unified(args.weights, args.g0_checkpoint, Path(args.run) / selection['checkpoint'])
    g0, _, _ = build_unified(args.weights, args.g0_checkpoint, 224)
    model.to(device); g0.to(device)
    candidate_data = loader(EvaluationSet(args.manifest, indices, size), 4 if size == 384 else 16, args.workers)
    baseline_data = loader(EvaluationSet(args.manifest, indices, 224), 4 if size == 384 else 16, args.workers)
    records, reference_records = [], []
    for candidate_batch, baseline_batch in zip(candidate_data, baseline_data, strict=True):
        clean, noisy, truth, log_ratio, modes, photo_sizes = candidate_batch
        old_clean, old_noisy, old_truth, old_log_ratio, old_modes, old_sizes = baseline_batch
        if not np.array_equal(modes.numpy(), old_modes.numpy()) or not np.array_equal(photo_sizes.numpy(), old_sizes.numpy()):
            raise RuntimeError('Historical regression pairing differs')
        for condition, pixels, old_pixels in [('clean', clean, old_clean), ('noisy', noisy, old_noisy)]:
            predictions, old_predictions = predict(model, pixels.numpy(), device), predict(g0, old_pixels.numpy(), device)
            for index, (prediction, old_prediction) in enumerate(zip(predictions, old_predictions, strict=True)):
                photo_size = tuple(map(int, photo_sizes[index].numpy()))
                row = measure_geometry(decode_prediction(prediction, size, photo_size), truth[index].numpy(), float(log_ratio[index]), size)
                baseline_row = measure_geometry(decode_prediction(old_prediction, 224, photo_size), old_truth[index].numpy(), float(old_log_ratio[index]), 224)
                stratum = {'condition': condition, 'geometry': ('identity', 'mild', 'hard')[int(modes[index])]}
                records.append({**row, **stratum}); reference_records.append({**baseline_row, **stratum})
        if len(records) % 400 < 32 or len(records) == 2868:
            _status(args, 'historical_R', completed=len(records), total=2868)
    del model, g0, candidate_data, baseline_data
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    if len(records) != 2868:
        raise RuntimeError('Historical regression view count changed')
    summary, baseline = summarize_rows(records), summarize_rows(reference_records)
    report = {'selection_sha256': sha256(output / 'selection-lock.json'), 'views': 2868,
              'scope': 'frozen_ratio_v1_scene_regression_not_new_recipe', 'summary': summary, 'G0': baseline,
              'gates': admission_gates({'R': summary}, {'R': baseline})}
    write_json(report_path, report)
    return report


def review_and_benchmark(args, selection=None):
    from .model import build, torch, keras
    from .ratio_model import predict as old_predict
    from .ratio_evaluation import predicted_quad
    from .ratio_review import rectify_canvas, panel
    from .unified_export import load_export
    selection = selection or freeze_selection(args)
    _validate_lock(args, selection)
    output = _directory(args)
    report_path = output / 'real47-report.json'
    if report_path.exists():
        return _load(report_path)
    export = _export(args, selection)
    benchmark = _benchmark(args, selection)
    scripted, _ = load_export(output / 'exports' / selection['group'] / export['artifact'])
    with keras.device('cpu'):
        baseline, _ = build(args.weights)
    baseline.to('cpu')
    files = _real_files(args)
    hashes = [sha256(path) for path in files]
    gallery = output / 'real47'
    gallery.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads(); torch.set_num_threads(4)
    rows = []
    try:
        for ordinal, path in enumerate(files, 1):
            image = read_rgb(path)
            result, decoded, plan, _, _ = _runtime(scripted, image, selection['input_size'], selection['thresholds'])
            overlay = image.copy()
            if decoded['quad'] is not None and np.isfinite(decoded['quad']).all():
                cv2.polylines(overlay, [np.rint(decoded['quad']).astype(np.int32)], True, (0, 255, 0), max(1, round(image.shape[1] / 500)))
            old_input = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            old_parameters = old_predict(baseline, old_input[None], 'cpu')[0]
            old_quad = (predicted_quad(old_parameters, 224, True) + .5) * [image.shape[1] / 224, image.shape[0] / 224] - .5
            try:
                old_result, _ = rectify_canvas(image, old_quad)
                old_valid = True
            except ValueError:
                old_result, old_valid = None, False
            tiles = [panel(image, f'#{ordinal:03d} original'), panel(overlay, 'Unified predicted corners'),
                     panel(result, f'Unified: {plan["action"]}'), panel(old_result, 'B1.1 baseline')]
            sheet = np.concatenate([np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)], axis=0)
            contact_name = f'photo-{ordinal:03d}.local.jpg'
            if not cv2.imwrite(str(gallery / contact_name), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError('Failed to write local review contact sheet')
            output_name = None
            if result is not None:
                output_name = f'output-{ordinal:03d}.local.png'
                if not cv2.imwrite(str(gallery / output_name), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError('Failed to write local result image')
            rows.append({'ordinal': ordinal, 'contact_sheet': contact_name, 'output': output_name,
                         'action': plan['action'], 'reason': plan['reason'], 'valid': decoded['valid'],
                         'quad': decoded['quad'].tolist() if decoded['quad'] is not None else None,
                         'ratio': decoded['ratio'], 'B11_valid': old_valid,
                         'output_size': list(plan['output_size']),
                         'matrix': np.asarray(plan['matrix']).tolist() if plan['matrix'] is not None else None})
            _status(args, 'real47_gallery', completed=ordinal, total=47)
    finally:
        torch.set_num_threads(previous_threads)
    if any(sha256(path) != before for path, before in zip(files, hashes, strict=True)):
        raise RuntimeError('Read-only real photo changed')
    report = {'selection_sha256': sha256(output / 'selection-lock.json'), 'count': 47,
              'scope': 'unpaired_exposed_visual_observation_not_ratio_accuracy', 'sources_unchanged': True,
              'visual_review_status': 'pending_explicit_per_image_review', 'records': rows,
              'CPU': benchmark, 'export': export, 'application_changed': False}
    write_json(report_path, report)
    # Main reviewer may complete this separate record; report refresh never opens T again.
    template = {'selection_sha256': sha256(output / 'selection-lock.json'), 'reviewer': None,
                'scope': 'unpaired_visual_only', 'records': [{'ordinal': index, 'reviewed': False,
                    'content_complete': None, 'orientation_correct': None, 'no_obvious_stretch': None,
                    'background_removed': None, 'normal_image_preserved': None,
                    'notes': ''} for index in range(1, 48)]}
    review_path = output / 'real47-observations.local.json'
    if not review_path.exists():
        write_json(review_path, template)
    return report


def final_report(args, selection=None):
    """Refresh final evidence after explicit visual review, without re-running T."""
    selection = selection or _load(_directory(args) / 'selection-lock.json')
    _validate_lock(args, selection)
    output = _directory(args)
    development = _load(output / 'D-calibration.json')['groups'][selection['group']]
    test, regression, real = (_load(output / name) for name in ('T-report.json', 'R-report.json', 'real47-report.json'))
    lock_sha = sha256(output / 'selection-lock.json')
    if any(record['selection_sha256'] != lock_sha for record in (test, regression, real)):
        raise RuntimeError('Final evidence is not bound to the same selection')
    observations = _load(output / 'real47-observations.local.json')
    rows = observations.get('records', [])
    fields = ('content_complete', 'orientation_correct', 'no_obvious_stretch', 'background_removed', 'normal_image_preserved')
    reviewed = (observations.get('selection_sha256') == lock_sha and len(rows) == 47
                and {row.get('ordinal') for row in rows} == set(range(1, 48))
                and all(row.get('reviewed') is True for row in rows))
    visual_pass = bool(reviewed and all(all(row.get(key) is True for key in fields) for row in rows))
    train_audit_path = Path(args.run) / 'cache/train/complete.json'
    training = _load(train_audit_path) if train_audit_path.exists() else {}
    audit = training.get('audit', {})
    data_pass = bool(training.get('complete') and training.get('views') == 80000
                     and audit.get('source_count') == 4114
                     and audit.get('source_slots_min') == 19 and audit.get('source_slots_max') == 20
                     and audit.get('counts', {}).get('photo_above_half') == 51200
                     and audit.get('full_film_visibility_verified')
                     and test['actual_data_audit']['full_film_visibility_verified'])
    integrity_path = Path(args.run) / 'original-integrity-after.json'
    integrity = _load(integrity_path) if integrity_path.exists() else {}
    before = _load(Path(args.run) / 'original-integrity-before.json')
    manifest_unchanged = (integrity.get('manifest_sha256') == before['manifest_sha256'] == sha256(args.manifest))
    original_integrity = (integrity.get('originals_checked') == 4620
                          and integrity.get('all_hashes_unchanged') is True and manifest_unchanged)
    reliable_passthrough = (integrity.get('reliable_source_passthrough_count') == 4620
                            and integrity.get('reliable_source_values_unchanged') is True)
    checks = {'data_audit': bool(data_pass), 'original_source_integrity': bool(original_integrity),
              'manifest_unchanged': bool(manifest_unchanged), 'reliable_source_passthrough': bool(reliable_passthrough),
              'D': development['gates']['eligible'],
              'T': test['gates']['eligible'], 'R': regression['gates']['eligible'],
              'CPU': real['CPU']['cpu_gate'], 'export': real['export']['max_absolute_output_difference'] <= 1e-5,
              'real47_review_complete': bool(reviewed), 'real47_visual': visual_pass}
    report = {'status': ('quality_gates_passed_local_integration_pending' if all(checks.values()) else
                        'research_complete_no_admitted_candidate' if reviewed else 'research_complete_visual_review_pending'),
              'selection': selection, 'checks': checks, 'failed': [key for key, value in checks.items() if not value],
              'quality_eligible': all(checks.values()), 'deployment_performed': False, 'application_changed': False,
              'active_application_model': 'B1.1', 'total_training_exposures': 560000,
              'independent_real_ratio_validated': False,
              'test_scope': test['scope'], 'visual_scope': real['scope']}
    write_json(output / 'final-report.json', report)
    write_json(Path(args.run) / 'completion.json', report)
    _status(args, report['status'], quality_eligible=report['quality_eligible'], failed=report['failed'])
    return report


def finalize(args):
    """Run D calibration -> immutable selection -> T -> R -> CPU/local real gallery."""
    _verify_training(args)
    selection = freeze_selection(args, calibrate_development(args))
    evaluate_sealed(args, selection)
    evaluate_regression(args, selection)
    review_and_benchmark(args, selection)
    return final_report(args, selection)
