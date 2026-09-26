import cv2
import numpy as np
import pytest


def test_benchmark_report_never_overwrites_read_only_real_inputs(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from ptrn_hip import unified_finalize as final
    inputs = tmp_path / 'read_only_images'
    inputs.mkdir()
    for ordinal in range(47):
        assert cv2.imwrite(str(inputs / f'{ordinal:03d}.png'), np.full((12, 9, 3), ordinal, np.uint8))
    before = {path: path.read_bytes() for path in inputs.iterdir()}
    run = tmp_path / '.local' / 'run'
    run.mkdir(parents=True)
    args = SimpleNamespace(run=run, real_dir=inputs)
    threads = [1]
    fake_torch = SimpleNamespace(get_num_threads=lambda: threads[0], set_num_threads=lambda value: threads.__setitem__(0, value))
    monkeypatch.setitem(sys.modules, 'ptrn_hip.model', SimpleNamespace(torch=fake_torch))
    monkeypatch.setitem(sys.modules, 'ptrn_hip.unified_export', SimpleNamespace(load_export=lambda *a, **kw: (object(), {})))
    monkeypatch.setattr(final, '_export', lambda *a: {'artifact': 'fake.pt'})
    monkeypatch.setattr(final, '_status', lambda *a, **kw: None)
    monkeypatch.setattr(final, '_runtime', lambda *a: (None, {}, {'action': 'keep'}, 10., 12.))
    candidate = {'group': 'S224', 'input_size': 224, 'checkpoint_sha256': 'fixture', 'thresholds': {'shape': .005}}
    report = final._benchmark(args, candidate)
    assert report['same_real_photo_count'] == 47 and report['sources_unchanged']
    assert (run / 'final/exports/S224/cpu47.json').is_file()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert threads == [1]
    assert report['threads']['2']['pure_inference_ms']['p95'] == 10.



def test_json_artifact_guard_rejects_image_destination_and_nonlocal_directory(tmp_path):
    from ptrn_hip.unified_finalize import write_json
    image = tmp_path / '.local' / 'original.jpg'
    image.parent.mkdir()
    image.write_bytes(b'original image fixture')
    with pytest.raises(ValueError, match='JSON destination'):
        write_json(image, {'bad': True})
    assert image.read_bytes() == b'original image fixture'
    with pytest.raises(ValueError, match='JSON destination'):
        write_json(tmp_path / 'report.json', {'bad': True})



def test_interrupted_export_preserves_partial_attempt_then_retries(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from ptrn_hip import unified_finalize as final
    run = tmp_path / '.local' / 'run'
    partial = run / 'final/exports/S224'
    partial.mkdir(parents=True)
    (partial / 'partial.pt').write_bytes(b'local partial fixture')
    (run / 'source-inventory.json').write_text('{"tree_sha256":"fixture-tree"}')
    def export_candidate(weights, g0, checkpoint, output, source_sha, real_dir):
        output.mkdir(parents=True, exist_ok=False)
        assert source_sha == 'fixture-tree'
        return {'fixture_export': True}
    monkeypatch.setitem(sys.modules, 'ptrn_hip.unified_export', SimpleNamespace(
        export_candidate=export_candidate, load_export=lambda *a, **kw: None))
    monkeypatch.setattr(final, '_status', lambda *a, **kw: None)
    args = SimpleNamespace(run=run, weights='fixture', g0_checkpoint='fixture', real_dir=None)
    candidate = {'group': 'S224', 'input_size': 224, 'checkpoint': 'S224/step.pt', 'checkpoint_sha256': 'fixture'}
    assert final._export(args, candidate)['fixture_export']
    preserved = list(partial.parent.glob('S224.incomplete-*'))
    assert len(preserved) == 1 and (preserved[0] / 'partial.pt').read_bytes() == b'local partial fixture'
    assert partial.is_dir()



def test_selection_lock_rejects_argument_and_calibration_mutation(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from ptrn_hip import unified_finalize as final
    from ptrn_hip.ratio_data import sha256
    run = tmp_path / '.local/run'
    output = run / 'final'
    output.mkdir(parents=True)
    (run / 'sealed').mkdir()
    (run / 'S224').mkdir()
    checkpoint = run / 'S224/step.pt'; checkpoint.write_bytes(b'checkpoint fixture')
    calibration = output / 'D-calibration.json'; calibration.write_text('{}')
    schedule = run / 'sealed/test.json'; schedule.write_text('[]')
    manifest = run / 'manifest.local.json'; manifest.write_text('{}')
    selection = {'checkpoint': 'S224/step.pt', 'checkpoint_sha256': sha256(checkpoint),
                 'development_calibration_sha256': sha256(calibration), 'test_schedule_sha256': sha256(schedule),
                 'manifest_sha256': sha256(manifest), 'source_tree_sha256': 'fixture-tree',
                 'thresholds': {'shape': .005}}
    final.write_json(output / 'selection-lock.json', selection)
    monkeypatch.setitem(sys.modules, 'ptrn_hip.unified_runner', SimpleNamespace(assert_source=lambda run: {'tree_sha256': 'fixture-tree'}))
    args = SimpleNamespace(run=run, manifest=manifest)
    final._validate_lock(args, selection)
    with pytest.raises(RuntimeError, match='arguments differ'):
        final._validate_lock(args, {**selection, 'thresholds': {'shape': .01}})
    calibration.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='calibration changed'):
        final._validate_lock(args, selection)


def test_real_input_directory_cannot_overlap_run_outputs(tmp_path):
    from types import SimpleNamespace
    from ptrn_hip.unified_finalize import _real_files
    root = tmp_path / '.local' / 'inputs'
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match='disjoint'):
        _real_files(SimpleNamespace(run=root / 'run', real_dir=root))
