"""CPU-only numerical gates using synthetic tensors; no real cache or GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

from ptrn_hip import unified_benchmark as benchmark
from ptrn_hip.model import keras, torch


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(), 'Run these tests with CUDA_VISIBLE_DEVICES=""'
    torch.set_num_threads(2)


def test_same_shape_stays_strict_cross_shape_accepts_small_cancellation_error():
    reference = [torch.tensor([10., 1e-4, -3.])]
    changed = [reference[0] + torch.tensor([0., 5e-6, 0.])]
    with pytest.raises(AssertionError, match='Head gradient'):
        benchmark._check_gradient_equivalence(changed, reference, strict=True)
    report = benchmark._check_gradient_equivalence(changed, reference)
    assert report['passed']
    assert report['global']['relative_l2'] < 1e-6
    assert report['global']['reference_max_abs'] == 10
    assert benchmark._check_gradient_equivalence(reference, reference, strict=True)['global']['bitwise_equal']


@pytest.mark.parametrize('kind', ['scale', 'direction', 'localized'])
def test_cross_shape_gates_reject_material_gradient_errors(kind):
    reference = [torch.ones(65536)]
    changed = [reference[0].clone()]
    if kind == 'scale':
        changed[0] *= 1.001
    elif kind == 'direction':
        changed[0] *= -1
    else:
        # Its relative L2 and cosine pass: the peak gate must still catch it.
        changed[0][0] += .001
    with pytest.raises(AssertionError, match='Head gradient') as failure:
        benchmark._check_gradient_equivalence(changed, reference)
    checks = failure.value.equivalence_details['checks'][0]
    assert not all(checks.values())
    if kind == 'localized':
        assert checks['relative_l2'] and checks['cosine'] and not checks['scaled_max_abs']


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_comparisons_fail(bad):
    with pytest.raises(FloatingPointError, match='Non-finite'):
        benchmark._check_gradient_equivalence([torch.tensor([bad])], [torch.ones(1)])


def test_zero_reference_and_shape_mismatch_fail_closed():
    zeros = [torch.zeros(3)]
    assert benchmark._check_gradient_equivalence(zeros, zeros)['passed']
    with pytest.raises(AssertionError):
        benchmark._check_gradient_equivalence([torch.tensor([1e-12, 0., 0.])], zeros)
    with pytest.raises(ValueError, match='shapes'):
        benchmark._check_gradient_equivalence([torch.ones(4)], zeros)
    with pytest.raises(ValueError, match='length'):
        benchmark._check_gradient_equivalence([], zeros)


def test_adam_probe_is_copy_only_and_matches_first_step_formula():
    initial = [torch.tensor([[.2, -.3], [0., .4]]), torch.zeros(2)]
    gradients = [torch.tensor([[.5, -.4], [0., .1]]), torch.tensor([.2, -.1])]
    before = [p.clone() for p in initial]
    original_gradients = [g.clone() for g in gradients]
    updated, updates, info = benchmark._adam_probe(initial, gradients, 'cpu')
    norm = torch.linalg.vector_norm(torch.cat([g.ravel() for g in gradients]))
    clip = min(1., 1. / (float(norm) + 1e-6))
    for value, delta, old, gradient in zip(updated, updates, initial, gradients, strict=True):
        clipped = gradient * clip
        expected = old - 1e-4 * clipped / (clipped.abs() + 1e-8)
        torch.testing.assert_close(value, expected, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(delta, value-old)
    for old, current, gradient, original in zip(before, initial, gradients, original_gradients, strict=True):
        assert torch.equal(old, current) and torch.equal(gradient, original)
        assert current.grad is None
    assert info['clip_norm'] == 1 and info['learning_rate'] == 1e-4


def test_adam_gate_catches_near_zero_gradient_flip_hidden_by_gradient_norm():
    initial = [torch.zeros(2)]
    reference = [torch.tensor([1., 1e-8])]
    changed = [torch.tensor([1., -1e-8])]
    assert benchmark._check_gradient_equivalence(changed, reference)['passed']
    _, expected_update, _ = benchmark._adam_probe(initial, reference, 'cpu')
    _, actual_update, _ = benchmark._adam_probe(initial, changed, 'cpu')
    with pytest.raises(AssertionError, match='First Adam update'):
        benchmark._check_update_equivalence(actual_update, expected_update)


def test_update_direction_gate_catches_even_tiny_flip():
    reference = [torch.tensor([1e-4, 1e-12])]
    changed = [torch.tensor([1e-4, -1e-12])]
    with pytest.raises(AssertionError) as failure:
        benchmark._check_update_equivalence(changed, reference)
    checks = failure.value.equivalence_details['checks'][0]
    assert checks['max_abs'] and checks['relative_l2'] and not checks['direction']


def test_update_gate_rejects_wrong_optimizer_step_scale():
    initial, gradients = [torch.zeros(3)], [torch.tensor([1., -2., .3])]
    _, reference, _ = benchmark._adam_probe(initial, gradients, 'cpu')
    _, changed, _ = benchmark._adam_probe(initial, gradients, 'cpu', learning_rate=5e-5)
    with pytest.raises(AssertionError, match='First Adam update'):
        benchmark._check_update_equivalence(changed, reference)


def test_real_geometry_loss_keeps_effective_batch_mean_and_head_unchanged():
    from ptrn_hip.unified_model import decode_prediction, forward_features
    rng = np.random.default_rng(123)
    with keras.device('cpu'):
        head = keras.layers.Dense(9, dtype='float32')
        head(torch.zeros(1, 1920))
    kernel = rng.normal(0, 1e-5, (1920, 9)).astype(np.float32)
    bias = np.array([.7, 0, 0, 0, .6, 0, 0, 0, 0], np.float32)
    head.set_weights([kernel, bias])
    parameters = [v.value for v in head.trainable_weights]
    original = [p.detach().clone() for p in parameters]
    features = rng.normal(0, .1, (64, 1920)).astype(np.float32)
    with torch.no_grad():
        truth = decode_prediction(forward_features(head, features), 224)[0] + 2.
    ratio = torch.full((64,), .2)
    inverse = torch.eye(3).repeat(64, 1, 1)
    small = benchmark._head_pass(head, parameters, features, truth, ratio, inverse, 16)
    full = benchmark._head_pass(head, parameters, features, truth, ratio, inverse, 64)
    benchmark._check_outputs(full[0], small[0])
    benchmark._check_outputs(full[1], small[1])
    benchmark._check_gradient_equivalence(full[2], small[2])
    for value, expected in zip(parameters, original, strict=True):
        assert torch.equal(value, expected)


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    from ptrn_hip import unified_runner as runner
    run = tmp_path / 'run'
    (run / 'schedules').mkdir(parents=True)
    (run / 'schedules/train.json').write_text('[]')
    assets = {}
    for name in ('manifest', 'weights', 'g0_checkpoint'):
        path = tmp_path / name
        path.write_bytes(b'synthetic preflight binding')
        assets[name] = path
    args = SimpleNamespace(run=run, workers=0, **assets)
    monkeypatch.setattr(runner, 'assert_source', lambda _: {'tree_sha256':'synthetic', 'git_base':'synthetic'})
    monkeypatch.setattr(runner, 'read_schedule', lambda *args: [None] * 80000)
    monkeypatch.setattr(runner, 'configure', lambda: {'device':'cpu_test_double'})
    monkeypatch.setattr(benchmark, '_clear_cuda', lambda: None)
    monkeypatch.setattr(benchmark, '_cache_consistency', lambda *args: {'passed':True, 'optimizer_updates':2})
    calls = []
    def joint(args, schedule, size, micro):
        calls.append((size, micro))
        return {'updates':100, 'estimated_1250_updates_seconds_excluding_evaluation':1250.}
    monkeypatch.setattr(benchmark, '_joint_benchmark', joint)
    return args, calls


def test_disposable_steps_accounted_without_changing_two_timing_probes(orchestration):
    args, calls = orchestration
    report = benchmark.benchmark(args)
    assert calls == [(224, 16), (384, 4)]
    assert report['diagnostic_optimizer_updates_planned'] == 202
    assert report['diagnostic_optimizer_updates_completed'] == 202
    assert report['diagnostic_training_exposures_completed'] == 12800
    assert report['joint_optimizer_updates_completed'] == 200
    assert report['cache_adam_probe_updates_completed'] == 2
    assert report['formal_training_budget_consumed'] == 0
    assert not list(args.run.rglob('*.pt'))


def test_partial_adam_diagnostic_failure_is_not_counted_as_formal_training(orchestration, monkeypatch):
    args, calls = orchestration
    def fail(*args):
        error = AssertionError('synthetic bad first update')
        error.cache_equivalence = {'passed':False, 'optimizer_updates':1}
        raise error
    monkeypatch.setattr(benchmark, '_cache_consistency', fail)
    with pytest.raises(AssertionError, match='synthetic'):
        benchmark.benchmark(args)
    import json
    report = json.loads((args.run/'preflight-benchmark.json').read_text())
    assert report['diagnostic_optimizer_updates_completed'] == 1
    assert report['diagnostic_training_exposures_completed'] == 0
    assert report['joint_optimizer_updates_completed'] == 0
    assert report['cache_adam_probe_updates_completed'] == 1
    assert report['formal_training_budget_consumed'] == 0
    assert not calls
