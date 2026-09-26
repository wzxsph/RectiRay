"""Opt-in V3 CPU verification / GPU research training; no application replacement.

All sources are traversed in each round before reshuffling. Resume binds data,
source code, initial weights and objective settings; prefetched but unused rows
are regenerated deterministically. New model effectiveness is not implied.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time

# Device visibility must be selected by the CLI before Keras is imported.

def save_checkpoint(path, model, optimizer, *, step, cursor, binding):
    import torch
    from .partial_geometry import CONTRACT
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'contract': CONTRACT, 'model_contract': model.contract(), 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'step': step, 'cursor': cursor,
               'binding': binding, 'torch_rng': torch.get_rng_state(),
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def restore_checkpoint(path, model, optimizer, binding):
    import torch
    from .partial_geometry import CONTRACT
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('contract') != CONTRACT or data.get('binding') != binding or data.get('model_contract') != model.contract():
        raise ValueError('Checkpoint contract/data/code/config binding mismatch')
    if data['cursor'] != data['step']*binding['effective_batch']:
        raise ValueError('Checkpoint sample cursor mismatch')
    model.load_state_dict(data['model'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    torch.set_rng_state(data['torch_rng'])
    if torch.cuda.is_available() and data['cuda_rng']: torch.cuda.set_rng_state_all(data['cuda_rng'])
    return data['step'], data['cursor']


class ShuffledRounds:
    """Map an absolute sample cursor into a reproducible full-source round."""
    def __init__(self, dataset):
        self.dataset, self.round_cache = dataset, {}

    def __getitem__(self, cursor):
        import numpy as np
        round_index, offset = divmod(cursor, len(self.dataset))
        if round_index not in self.round_cache:
            rng = np.random.default_rng(np.random.SeedSequence([self.dataset.seed, round_index, 188]))
            self.round_cache = {round_index: rng.permutation(len(self.dataset))}
        self.dataset.round_index = round_index
        return self.dataset[int(self.round_cache[round_index][offset])]


def train_step(model, optimizer, batches, *, device, effective_batch, loss_options=None):
    import torch
    from .partial_objectives import partial_loss
    optimizer.zero_grad(set_to_none=True)
    count, totals = 0, {}
    for batch in batches:
        tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
        n = len(tensors['pixels'])
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            pixels = tensors['pixels'].float()/255
            normalized = (pixels-pixels.new_tensor([.485, .456, .406]))/pixels.new_tensor([.229, .224, .225])
            output = model(normalized)
        terms = partial_loss(output, tensors, model.size, **(loss_options or {}))
        (terms['total']*(n/effective_batch)).backward()
        for key, value in terms.items(): totals[key] = totals.get(key, 0.)+float(value.detach())*n/effective_batch
        count += n
        del output, tensors, terms
    # Bound Keras/Torch graph cycles to one effective batch, instead of making
    # each microbatch pay a full Python heap traversal. CUDA memory is checked
    # in the sustained optimizer smoke before increasing physical batch size.
    gc.collect()
    if count != effective_batch: raise ValueError('Incomplete effective batch; no optimizer update')
    params = [p for group in optimizer.param_groups for p in group['params']]
    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in params): raise FloatingPointError('Nonfinite updated parameter')
    return totals


def worker_init(_):
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--backgrounds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--microbatch', type=int, choices=[1, 2, 4, 8, 16, 32, 64], default=2)
    p.add_argument('--effective-batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--size', type=int, choices=[224, 384], default=224)
    p.add_argument('--stage', choices=['head', 'last_block'], default='head')
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--coverage-weight', type=float, default=1.)
    p.add_argument('--mask-weight', type=float, default=1.)
    p.add_argument('--ratio-weight', type=float, default=.25)
    p.add_argument('--resume', type=Path)
    p.add_argument('--eval-limit', type=int, help='CPU smoke subset only; omit for full fixed development')
    args = p.parse_args()
    if args.device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['KERAS_BACKEND'] = 'torch'
    import numpy as np
    import torch
    import cv2
    from torch.utils.data import DataLoader
    from .partial_data import PartialDataset, sha256
    from .partial_model import build_partial
    if (min(args.steps, args.effective_batch) < 1 or args.effective_batch % args.microbatch or args.workers < 0
        or min(args.coverage_weight, args.mask_weight, args.ratio_weight) < 0
        or (args.eval_limit is not None and args.eval_limit < 1)):
        raise ValueError('Invalid steps/batch/objective configuration')
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('No allocated GPU')
    if args.device == 'cuda' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 GPU required')
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        raise ValueError('Refusing to overwrite an existing run')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    dataset = PartialDataset(args.bundle, args.backgrounds, size=args.size, seed=args.seed)
    # Freeze development identities/scene recipe before any optimizer update.
    development = PartialDataset(args.bundle, args.backgrounds, split='dev', size=args.size, seed=args.seed)
    code = Path(__file__).resolve().parent
    binding = {**dataset.fingerprint, 'effective_batch': args.effective_batch, 'stage': args.stage,
               'loss': {'coverage_weight': args.coverage_weight, 'mask_weight': args.mask_weight,
                        'ratio_weight': args.ratio_weight},
               'development': development.fingerprint, 'eval_limit': args.eval_limit,
               'code': {f.name: sha256(f) for f in sorted(code.glob('*.py'))},
               'weights': {name: sha256(args.bundle/'weights'/name) for name in ('b11.weights.h5', 'g0.pt')}}
    protocol_path = args.output/'frozen-protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != binding:
        raise ValueError('Existing frozen protocol differs; refusing overwrite')
    if not protocol_path.exists(): protocol_path.write_text(json.dumps(binding, indent=2)+'\n')
    model = build_partial(args.bundle/'weights/b11.weights.h5', args.bundle/'weights/g0.pt',
                          size=args.size, stage=args.stage).to(args.device)
    optimizer = torch.optim.Adam(model.optimizer_groups(args.stage))
    start, cursor = (restore_checkpoint(args.resume, model, optimizer, binding) if args.resume else (0, 0))
    if start >= args.steps: raise ValueError('Target steps already reached')
    bn_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                 if 'moving_mean' in k or 'moving_variance' in k}
    if not bn_before: raise RuntimeError('No BN state found for freeze verification')
    indices = range(cursor, args.steps*args.effective_batch)
    loader = DataLoader(ShuffledRounds(dataset), sampler=indices, batch_size=args.microbatch,
                        num_workers=args.workers, pin_memory=args.device == 'cuda', worker_init_fn=worker_init,
                        **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if args.workers else {}))
    iterator = iter(loader)
    if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
    evaluated_full = False
    for step in range(start+1, args.steps+1):
        begun = time.perf_counter()
        terms = train_step(model, optimizer, (next(iterator) for _ in range(args.effective_batch//args.microbatch)),
                           device=args.device, effective_batch=args.effective_batch, loss_options=binding['loss'])
        cursor += args.effective_batch
        record = {'step': step, 'cursor': cursor, 'round': cursor/len(dataset),
                  'seconds_including_data': time.perf_counter()-begun, **terms}
        if args.device == 'cuda': record['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        with (args.output/'training.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if step in (500, 1250, 5000) or step == args.steps or step % 250 == 0:
            if not all(torch.equal(v, model.state_dict()[k].cpu()) for k, v in bn_before.items()):
                raise RuntimeError('Frozen BN mutated')
            save_checkpoint(args.output/f'step-{step}.pt', model, optimizer, step=step, cursor=cursor, binding=binding)
            if step in (500, 1250, 5000) or step == args.steps:
                from .partial_evaluation import evaluate
                evaluation = evaluate(model, development, device=args.device, limit=args.eval_limit)
                (args.output/f'evaluation-{step}.json').write_text(json.dumps(evaluation, indent=2)+'\n')
                evaluated_full = evaluation['full_development']
    (args.output/'completion.json').write_text(json.dumps({'steps': args.steps, 'BN_unchanged': True,
        'code_smoke_only': args.device == 'cpu', 'effectiveness_verified': False,
        'GPU_performance_verified': False, 'full_development_evaluated': evaluated_full}, indent=2)+'\n')


if __name__ == '__main__': main()
