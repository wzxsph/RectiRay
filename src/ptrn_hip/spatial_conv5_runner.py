"""Reproducible small-set spatial localization experiment; originals stay read-only.

This separately versioned runner caches the provably frozen encoder prefix.
Stage B uses a source-stratified 32 x 4 fixed view set and GT ratios only for
localization diagnostics. It is never an end-to-end or independent test score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .data import write_json
from .ratio_data import Sources, sha256
from .unified_runner import sample_record
from .unified_geometry import quad_reasons
from .ratio_synthesis import project

ARMS = ('gap_h8', 'gap_quad', 'spatial_quad', 'spatial_heatmap')
CHECKPOINTS = (500, 2000, 5000)
SEED = 20260923


def bind_sources(run, extra=()):
    """Freeze imported training sources, independent of unrelated parallel work."""
    root = Path(__file__).parent
    paths = [p for p in root.glob('*.py') if not p.name.startswith('spatial_')]
    paths += [root / n for n in ('spatial_model.py', 'spatial_runner.py', 'spatial_conv5_runner.py', 'spatial_feature_cache.py', *extra)]
    mapping = {p.name: sha256(p) for p in sorted(set(paths))}
    binding = {'files': mapping,
               'sha256': hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()}
    target = Path(run) / 'source-binding.json'
    if target.exists():
        if json.loads(target.read_text()) != binding:
            raise RuntimeError('Stage source changed after freeze; use a new stage/run revision')
    else:
        write_json(target, binding)
        for p in paths:
            q = Path(run) / 'source-snapshot' / p.name
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(p.read_bytes())
    return binding


def fixed_small_schedule(records, indices):
    """32 evenly spaced source aspect ranks, including both aspect extremes."""
    if len(indices) < 32:
        raise ValueError('Small-set diagnostic requires at least 32 training sources')
    ranked = sorted(map(int, indices), key=lambda i: (
        records[i]['original_size'][0] / records[i]['original_size'][1], i))
    selected = [ranked[i] for i in np.linspace(0, len(ranked) - 1, 32, dtype=int)]
    rows = []
    for ordinal, source in enumerate(selected):
        for scene, (kind, mode, condition) in enumerate((
                ('digital', 'front', 'clean'), ('photo', 'front', 'clear'),
                ('photo', 'mild', 'noisy'), ('photo', 'medium', 'noisy'))):
            rows.append({'source_index': source, 'view_index': 200000 + ordinal * 4 + scene,
                         'kind': kind, 'mode': mode, 'condition': condition,
                         'level': 'clear' if condition in ('clean', 'clear') else 'medium',
                         'occupancy_bin': (ordinal + scene) % 3,
                         'pure_roll': kind == 'photo' and mode == 'front', 'domain': 'train'})
    return rows


def prepare_small(args):
    run = args.run / 'B'
    run.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.prior_run / 'split.local.json').read_text())
    if {k: len(v) for k, v in split.items()} != {'train': 4114, 'dev': 478, 'quarantine': 28}:
        raise ValueError('Unexpected source partition')
    audit = json.loads((args.prior_run / 'original-integrity-before.json').read_text())
    if sha256(args.manifest) != audit['manifest_sha256']:
        raise ValueError('Manifest changed')
    sources = Sources(args.manifest, cache_size=1)
    schedule = fixed_small_schedule(sources.records, split['train'])
    dataset_path = run / 'views.npz'
    if dataset_path.exists():
        meta = json.loads((run / 'dataset.json').read_text())
        if meta['schedule'] != schedule or meta['manifest_sha256'] != sha256(args.manifest):
            raise RuntimeError('Fixed dataset binding differs')
        if meta['pixels_sha256'] != sha256(dataset_path):
            raise RuntimeError('Fixed dataset corrupted')
        return meta
    pixels, truth, inverse, ratios, metadata = [], [], [], [], []
    for row in schedule:
        p, q, r, inv, info, _ = sample_record(sources, row, 224)
        pixels.append(p); truth.append(q); inverse.append(inv); ratios.append(r); metadata.append(info)
    np.savez_compressed(dataset_path, pixels=np.stack(pixels), quad=np.stack(truth),
                        inverse=np.stack(inverse), log_ratio=np.array(ratios))
    record = {'scope': 'training_fit_only_32_sources_128_fixed_views_GT_ratio_diagnostic',
              'schedule': schedule, 'metadata': metadata, 'manifest_sha256': sha256(args.manifest),
              'pixels_sha256': sha256(dataset_path), 'source_count': 32, 'views': 128,
              'generator': 'frozen_unified_v1', 'seed': SEED, 'input_size': 224}
    write_json(run / 'dataset.json', record)
    return record


def statistics(prediction, truth, inverse, metadata, size=224):
    q = np.asarray(prediction, np.float64)
    if q.shape != truth.shape or not np.isfinite(q).all():
        raise ValueError('Nonfinite or incorrectly shaped predicted coordinates')
    errors = np.linalg.norm(q - truth, axis=2) / math.hypot(size, size)
    valid, photo_errors, identity_signed = [], [], []
    for p, t, inv, info in zip(q, truth, inverse, metadata):
        pq, tq = project(p, inv), project(t, inv)
        valid.append(not quad_reasons(pq, info['photo_size']))
        photo_errors.append(float(np.linalg.norm(pq - tq, axis=1).mean() /
                                  np.hypot(*info['photo_size'])))
        if info['kind'] == 'digital':
            # Positive is inward on all four sides; preserve signed boundary bias.
            identity_signed.append([float((pq[[0, 3], 0] - tq[[0, 3], 0]).mean()),
                                    float((tq[[1, 2], 0] - pq[[1, 2], 0]).mean()),
                                    float((pq[[0, 1], 1] - tq[[0, 1], 1]).mean()),
                                    float((tq[[2, 3], 1] - pq[[2, 3], 1]).mean())])
    result = {'count': len(q), 'corner_nme_mean': float(errors.mean()),
              'corner_nme_p90': float(np.percentile(errors.mean(axis=1), 90)),
              'worst_corner_nme': float(errors.max()), 'valid_quads': sum(valid),
              'photo_nme_mean': float(np.mean(photo_errors)),
              'digital_inward_side_bias_photo_px': np.mean(identity_signed, axis=0).tolist(),
              'by_kind': {k: {'count': sum(i['kind'] == k for i in metadata),
                             'mean_nme': float(errors[[i['kind'] == k for i in metadata]].mean()),
                             'valid_quads': sum(v for v, i in zip(valid, metadata) if i['kind'] == k)}
                          for k in sorted({i['kind'] for i in metadata})}}
    result['fit_pass'] = bool(result['corner_nme_mean'] <= .005 and
                              result['corner_nme_p90'] <= .01 and all(valid))
    return result


def index_features(features, indices):
    return tuple(f[indices] for f in features) if isinstance(features, (tuple, list)) else features[indices]


@contextlib.contextmanager
def gpu_lock():
    with open('/tmp/hip-xray-ai-active10-gpu.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def train_small(args):
    from .model import torch, keras
    from .ratio_model import tensor_input
    from .spatial_model import build_spatial, coordinate_loss, parameter_groups, contract
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    keras.utils.set_random_seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for approved training')
    if args.target not in CHECKPOINTS:
        raise ValueError('Unexpected small-fit checkpoint')
    meta = prepare_small(args)
    stage_root = args.run / ('B' if args.stage == 'head' else 'B-conv5-cached')
    stage_root.mkdir(parents=True, exist_ok=True)
    binding = bind_sources(stage_root)
    output = stage_root / args.arm
    output.mkdir(exist_ok=True)
    data = np.load(args.run / 'B/views.npz')
    truth_np, inverse = data['quad'], data['inverse']
    truth = torch.tensor(truth_np, device='cuda')
    model = build_spatial(args.weights, args.g0_checkpoint, args.arm, size=224,
                          stage=args.stage, initial_quad=truth_np.mean(axis=0)).to('cuda')
    optimizer = torch.optim.Adam(parameter_groups(model, args.stage))
    config = {'arm': args.arm, 'size': 224, 'stage': args.stage, 'seed': SEED,
              'effective_batch': 16, 'microbatch': args.microbatch, 'head_lr': 1e-4,
              'backbone_lr': 1e-5, 'clip_norm': 1., 'contract': contract(224),
              'dataset_sha256': meta['pixels_sha256'], 'source_sha256': binding['sha256'],
              'g0_sha256': sha256(args.g0_checkpoint), 'b11_sha256': sha256(args.weights),
              'GT_ratio': 'diagnostic_only_no_ratio_training', 'checkpoints': list(CHECKPOINTS), 'execution': 'frozen_prefix_cache_equivalent_to_online'}
    cfg_path = output / 'config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise RuntimeError('Configuration changed on resume')
    write_json(cfg_path, config)
    checkpoints = sorted(output.glob('step-*.pt'))
    start = 0
    if checkpoints:
        latest = checkpoints[-1]
        side = json.loads(latest.with_suffix('.json').read_text())
        if side['sha256'] != sha256(latest): raise RuntimeError('Checkpoint digest mismatch')
        payload = torch.load(latest, map_location='cuda', weights_only=True)
        if payload['config'] != config: raise RuntimeError('Checkpoint config mismatch')
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        start = int(payload['step'])
    if start > args.target: raise ValueError('Cannot rewind an existing arm')
    x = tensor_input(data['pixels'], 'cuda')
    features = None
    prefix_cache = None
    if args.stage == 'last_block':
        from .spatial_feature_cache import FrozenPrefixCache
        prefix_cache = FrozenPrefixCache(model, x, batch_size=8)
    if args.stage == 'head':
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(x), 8):
                chunks.append(model.extract_features(x[offset:offset + 8]))
        if isinstance(chunks[0], (tuple, list)):
            features = tuple(torch.cat([v[i] for v in chunks]) for i in range(len(chunks[0])))
        else: features = torch.cat(chunks)
        del chunks
    def evaluate(step):
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(x), args.microbatch):
                ids = torch.arange(offset, min(offset + args.microbatch, len(x)), device='cuda')
                p = model.forward_features(index_features(features, ids)) if features is not None else prefix_cache.forward(ids)
                predictions.append(p.detach().float().cpu().numpy())
        prediction = np.concatenate(predictions)
        result = statistics(prediction, truth_np, inverse, meta['metadata'])
        result.update(step=step, arm=args.arm, stage=args.stage,
                      max_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        write_json(output / f'eval-{step:04d}.json', result)
        np.save(output / f'quad-{step:04d}.npy', prediction)
        return result
    initial = evaluate(start)
    started = time.monotonic()
    rng = np.random.default_rng(SEED)
    # Each step is indexed independently for exact resume order.
    order = np.stack([rng.permutation(128) for _ in range((max(CHECKPOINTS) * 16 + 127) // 128)])
    batches = order.reshape(-1, 16)
    if 16 % args.microbatch: raise ValueError('Microbatch must divide 16')
    torch.cuda.reset_peak_memory_stats()
    trainable = [p for group in optimizer.param_groups for p in group['params']]
    bn_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                 if any(word in k for word in ('moving_mean', 'moving_variance', 'running_mean', 'running_var'))}
    if not bn_before:
        raise RuntimeError('Expected frozen BatchNorm buffers were not found')
    losses = []
    for step in range(start + 1, args.target + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = batches[step - 1]
        loss_value = 0.
        for offset in range(0, 16, args.microbatch):
            ids = torch.as_tensor(batch[offset:offset + args.microbatch], device='cuda')
            p = model.forward_features(index_features(features, ids)) if features is not None else prefix_cache.forward(ids)
            loss = coordinate_loss(p, truth[ids], 224)
            (loss * (args.microbatch / 16)).backward()
            loss_value += float(loss.detach()) * args.microbatch / 16
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in trainable):
            raise FloatingPointError('Nonfinite model parameter')
        losses.append(loss_value)
        if step % 16 == 0: gc.collect(0)
        if step % 50 == 0 or step == args.target:
            progress = {'phase': 'small_fit', 'arm': args.arm, 'stage': args.stage,
                        'step': step, 'target': args.target, 'loss_last50': float(np.mean(losses[-50:])),
                        'gradient_norm': float(norm), 'elapsed_seconds': time.monotonic() - started,
                        'reserved_mib': torch.cuda.memory_reserved() / 2**20}
            write_json(args.run / 'status.json', progress)
            print(json.dumps(progress), flush=True)
        if step in CHECKPOINTS:
            for k, old in bn_before.items():
                if not torch.equal(old, model.state_dict()[k].detach().cpu()):
                    raise RuntimeError('Frozen BN state changed')
            bind_sources(stage_root)
            target = output / f'step-{step:04d}.pt'
            temporary = target.with_suffix('.tmp')
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'step': step, 'config': config}, temporary)
            temporary.replace(target)
            result = evaluate(step)
            write_json(target.with_suffix('.json'), {'sha256': sha256(target), 'step': step,
                       'config': config, 'evaluation': result, 'elapsed_seconds': time.monotonic() - started})
            print(json.dumps({'checkpoint': step, 'result': result}), flush=True)
    return initial if args.target == start else result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-small', 'train-small'])
    for name in ('run', 'prior-run', 'manifest', 'weights', 'g0-checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default=ARMS[0])
    parser.add_argument('--target', type=int, choices=CHECKPOINTS, default=500)
    parser.add_argument('--stage', choices=['head', 'last_block'], default='head')
    parser.add_argument('--microbatch', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'prepare-small':
        record = prepare_small(args)
        print(json.dumps({'sources': record['source_count'], 'views': record['views']}))
    else:
        if args.stage != 'last_block':
            raise ValueError('Cached-prefix revision is only for the conv5 retry')
        with gpu_lock(): train_small(args)


if __name__ == '__main__': main()
