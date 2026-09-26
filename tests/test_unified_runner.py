"""Runner orchestration checks using abstract CPU fixtures, never patient data."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ptrn_hip import unified_runner as runner
from ptrn_hip.data import write_json
from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.unified_data import build_schedule


class FixtureSources:
    def __init__(self, manifest=None, cache_size=1):
        self.calls = []

    def read(self, index):
        self.calls.append(index)
        # Distinct constant-valued, nonmedical originals make reordering visible.
        return np.full((97, 173, 3), 30 + int(index), np.uint8)


def test_sample_record_preserves_geometry_pairing_and_full_keep_targets():
    rows = build_schedule([11], split='dev')
    sources = FixtureSources()
    metadata, photos = [], {}
    for record in rows:
        pixels, q, log_ratio, inverse, info, image = runner.sample_record(sources, record, 224)
        assert pixels.shape == (224, 224, 3)
        np.testing.assert_allclose(project(q, inverse), info['quad_photo'], atol=1e-4)
        np.testing.assert_allclose(
            project(pixel_corners(*info['source_size']), np.asarray(info['source_to_photo'])),
            info['quad_photo'], atol=1e-8)
        assert log_ratio == pytest.approx(np.log(173 / 97))
        assert info['photo_size'] == [image.shape[1], image.shape[0]]
        assert info['source_index'] == 11
        assert info['condition'] == record['condition']
        if info['kind'] == 'photo':
            photos.setdefault(info['mode'], []).append((q, info['source_to_photo']))
        else:
            np.testing.assert_array_equal(info['source_to_photo'], np.eye(3))
            np.testing.assert_array_equal(info['quad_photo'], pixel_corners(173, 97))
        metadata.append(info)
    for pair in photos.values():
        np.testing.assert_array_equal(pair[0][0], pair[1][0])
        np.testing.assert_array_equal(pair[0][1], pair[1][1])
    assert runner.actual_view_audit(metadata)['full_film_visibility_verified']


def test_view_dataset_returns_original_schedule_indices_after_source_sorting(monkeypatch):
    monkeypatch.setattr(runner, 'Sources', FixtureSources)
    rows = [dict(source_index=source, view_index=i, kind='digital', mode='front',
                 occupancy_bin=1, level='clear', pure_roll=False, domain='train', condition='train')
            for i, source in enumerate((8, 2, 5))]
    dataset = runner.ViewDataset('unused-fixture-manifest', rows, 224, positions=[1, 2, 0], legacy=True)
    items = [dataset[i] for i in range(len(dataset))]
    assert [item[0] for item in items] == [1, 2, 0]
    assert [json.loads(item[5])['source_index'] for item in items] == [2, 5, 8]
    assert dataset.sources.calls == [2, 5, 8]
    for item in items:
        assert item[1].shape == (224, 224, 3)
        assert item[6].shape == (224, 224, 3)


def test_runner_supplies_required_preserve_and_front_evaluation_strata():
    records = build_schedule([1, 2], split='dev')
    source = FixtureSources()
    metadata, predictions = [], []
    for record in records:
        if record['kind'] == 'photo' and record['mode'] != 'front':
            continue
        _, q, log_ratio, _, info, _ = runner.sample_record(source, record, 224)
        unit = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], np.float32)
        matrix = cv2.getPerspectiveTransform(unit, (q * (2. / 223) - 1).astype(np.float32))
        photo_q = np.asarray(info['quad_photo'])
        edges = np.linalg.norm(np.roll(photo_q, -1, axis=0) - photo_q, axis=1)
        delta = float(log_ratio) - np.log(max(edges[0], edges[2]) / max(edges[1], edges[3]))
        predictions.append(np.r_[matrix.reshape(-1)[:8], delta])
        metadata.append(info)
    summary, _ = runner.measure_predictions(np.asarray(predictions), metadata, 224)
    assert set(summary['front_subgroup']) == {'upright', 'rotated'}
    assert set(summary['preserve_kind']) == {'digital', 'appearance', 'rectified'}
    assert set(summary['occupancy']) <= {'30-50', '50-75', '75-90', 'full_canvas'}
    assert summary['condition_geometry']['clear|front']['count'] == 2
    assert summary['condition_geometry']['noisy|front']['count'] == 2
    assert summary['condition_geometry']['keep|preserve']['count'] == 6


def _geometric_metadata(record):
    """Known analytic squares provide actual areas without rendering 80k images."""
    photo = record['kind'] == 'photo'
    occupancy = (.4, .6, .8)[record['occupancy_bin']] if photo else .9801
    side = np.sqrt(occupancy * 10000) if photo else 99.
    offset = (99 - side) / 2
    q = np.array([[offset, offset], [offset + side, offset],
                  [offset + side, offset + side], [offset, offset + side]])
    matrix = [[side / 99, 0., offset], [0., side / 99, offset], [0., 0., 1.]]
    mode = record['mode'] if photo else 'identity'
    return {**record, 'mode': mode, 'occupancy': occupancy,
            'tilt': {'front': 0., 'mild': 4., 'medium': 11., 'hard': 20., 'identity': 0.}[mode],
            'roll': 5. if record['pure_roll'] else 0.,
            'appearance': record['level'], 'requested_appearance': record['level'],
            'photo_size': [100, 100], 'common_size': [100, 100], 'source_size': [100, 100],
            'source_to_photo': matrix, 'quad_photo': q.tolist(),
            'quad_input': (q * 223 / 99).tolist(),
            'inverse_letterbox': np.diag([99 / 223, 99 / 223, 1.]).tolist(),
            'log_ratio': 0., 'camera_attempts': 1, 'moire': False, 'camera': 'none'}


def test_actual_audit_checks_complete_80000_slot_distribution_without_rendering():
    metadata = [_geometric_metadata(row) for row in build_schedule(range(4114))]
    result = runner.actual_view_audit(metadata)
    assert result['views'] == 80000
    assert result['source_count'] == 4114
    assert result['source_slots_min'] == 19 and result['source_slots_max'] == 20
    assert result['counts']['photo_above_half'] == 51200
    assert result['actual_photo_occupancy_by_geometry']['hard'] == {0: 640, 1: 1920, 2: 640}


@pytest.mark.parametrize('damage,message', [
    ('area', 'occupancy'), ('stratum', 'occupancy'), ('corner', 'corner|outside|visibility'),
    ('tilt', 'tilt|angle'),
])
def test_actual_audit_rejects_bad_rendered_geometry(damage, message):
    row = next(row for row in build_schedule([1], split='dev')
               if row['kind'] == 'photo' and row['mode'] == 'mild')
    info = _geometric_metadata(row)
    if damage == 'area':
        info['occupancy'] = .29
    elif damage == 'stratum':
        info['occupancy_bin'] = (info['occupancy_bin'] + 1) % 3
    elif damage == 'corner':
        info['quad_photo'][0][0] = -1.
    else:
        info['tilt'] = 26.
    with pytest.raises(RuntimeError, match=message):
        runner.actual_view_audit([info])


def test_sealed_schedule_cannot_be_read_without_explicit_final_evaluation(tmp_path):
    write_json(tmp_path / 'sealed/test.json', [{'fixture_only': True}])
    with pytest.raises(RuntimeError, match='Sealed test'):
        runner.read_schedule(tmp_path, 'test')
    assert runner.read_schedule(tmp_path, 'test', allow_test=True) == [{'fixture_only': True}]


def test_source_freeze_snapshots_code_and_rejects_drift_without_running_git(tmp_path, monkeypatch):
    project_root = tmp_path / 'project'
    source = project_root / 'src/example.py'
    source.parent.mkdir(parents=True)
    source.write_text('VALUE = 1\n')
    (project_root / 'tests').mkdir()
    (project_root / 'tests/test_example.py').write_text('def test_example(): pass\n')
    (project_root / 'pyproject.toml').write_text('[project]\nname = "fixture"\n')
    run = project_root / '.local/run'
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: 'fixture-revision\n')
    initial = runner.freeze_source(run)
    assert runner.freeze_source(run) == initial == runner.assert_source(run)
    assert set(initial['files']) == {'src/example.py', 'tests/test_example.py', 'pyproject.toml'}
    assert (run / 'source-snapshot/src/example.py').read_text() == 'VALUE = 1\n'
    source.write_text('VALUE = 2\n')
    with pytest.raises(RuntimeError, match='source drift'):
        runner.assert_source(run)
    with pytest.raises(RuntimeError, match='source changed'):
        runner.freeze_source(run)
    assert (run / 'source-snapshot/src/example.py').read_text() == 'VALUE = 1\n'


class ArrayBatch:
    """Small numpy stand-in for the DataLoader CPU tensors used by the cache."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def __len__(self):
        return len(self.value)


@pytest.fixture
def cache_fixture(tmp_path, monkeypatch):
    args = SimpleNamespace(run=tmp_path / 'run', manifest=tmp_path / 'fixture.json',
                           g0_checkpoint=tmp_path / 'g0.fixture', workers=0)
    args.g0_checkpoint.write_bytes(b'fixture checkpoint, no weights')
    write_json(args.manifest, {'fixture_only': True})
    rows = [dict(source_index=source, view_index=i, kind='digital', mode='front',
                 occupancy_bin=1, level='clear', pure_roll=False, domain='train', condition='train')
            for i, source in enumerate((8, 2, 5))]
    write_json(args.run / 'schedules/train.json', rows)
    write_json(args.run / 'data-audit.json', {})
    monkeypatch.setattr(runner, 'Sources', FixtureSources)
    monkeypatch.setattr(runner, 'assert_source', lambda run: {'tree_sha256': 'fixture-tree'})
    monkeypatch.setattr(runner, 'disk_check', lambda *args, **kwargs: 10.)
    controls = {'stop_after': None, 'seen_positions': [], 'feature_calls': 0}

    def fixture_loader(dataset, batch, workers):
        controls['seen_positions'].append(list(dataset.positions))
        for i in range(len(dataset)):
            if controls['stop_after'] is not None and i == controls['stop_after']:
                raise RuntimeError('Fixture interruption after durable cache batch')
            index, pixels, q, logr, inverse, info, old = dataset[i]
            yield (ArrayBatch([index]), ArrayBatch([pixels]), ArrayBatch([q]), ArrayBatch([logr]),
                   ArrayBatch([inverse]), [info], ArrayBatch([old]))

    def feature_predict(backbone, pixels, device):
        controls['feature_calls'] += 1
        # Deliberately encode source-specific center intensity to test index mapping.
        return np.repeat(pixels[:, 112, 112, :1].astype(np.float32), 1920, axis=1)

    monkeypatch.setattr(runner, 'loader', fixture_loader)
    monkeypatch.setitem(sys.modules, 'ptrn_hip.unified_model', SimpleNamespace(
        feature_predict=feature_predict, forward_features=None, predict=None))
    monkeypatch.setitem(sys.modules, 'ptrn_hip.ratio_model', SimpleNamespace(predict=None))
    monkeypatch.setitem(sys.modules, 'ptrn_hip.model', SimpleNamespace(torch=None))
    return args, rows, controls


def test_feature_cache_resumes_only_unfinished_slots_and_restores_schedule_order(cache_fixture):
    args, rows, controls = cache_fixture
    controls['stop_after'] = 1
    with pytest.raises(RuntimeError, match='Fixture interruption'):
        runner.build_feature_cache(args, 'train', None, None)
    done = np.load(args.run / 'cache/train/done.npy')
    np.testing.assert_array_equal(done, [False, True, False])
    controls['stop_after'] = None
    features, metadata = runner.build_feature_cache(args, 'train', None, None)
    assert controls['seen_positions'] == [[1, 2, 0], [2, 0]]
    assert controls['feature_calls'] == 3
    np.testing.assert_array_equal(features[:, 0], [38, 32, 35])
    assert [info['source_index'] for info in metadata] == [8, 2, 5]
    assert [info['cache_index'] for info in metadata] == [0, 1, 2]
    runner.build_feature_cache(args, 'train', None, None)
    assert controls['seen_positions'][-1] == []
    assert controls['feature_calls'] == 3


def test_cache_rejects_missing_durable_metadata(cache_fixture):
    args, _, controls = cache_fixture
    controls['stop_after'] = 1
    with pytest.raises(RuntimeError, match='Fixture interruption'):
        runner.build_feature_cache(args, 'train', None, None)
    (args.run / 'cache/train/metadata.jsonl').write_text('')
    with pytest.raises(RuntimeError, match='journal|metadata'):
        runner.build_feature_cache(args, 'train', None, None)


def test_cache_rejects_changed_binding_before_reusing_features(cache_fixture):
    args, _, _ = cache_fixture
    runner.build_feature_cache(args, 'train', None, None)
    args.g0_checkpoint.write_bytes(b'changed fixture checkpoint')
    with pytest.raises(RuntimeError, match='binding|signature|checkpoint'):
        runner.build_feature_cache(args, 'train', None, None)


def test_cache_cannot_reuse_done_flags_when_features_are_missing(cache_fixture):
    args, _, _ = cache_fixture
    runner.build_feature_cache(args, 'train', None, None)
    (args.run / 'cache/train/features.npy').unlink()
    with pytest.raises(RuntimeError, match='cache|Cache|feature|Feature'):
        runner.build_feature_cache(args, 'train', None, None)


def test_training_cache_load_rejects_changed_schedule(cache_fixture):
    args, rows, _ = cache_fixture
    runner.build_feature_cache(args, 'train', None, None)
    changed = deepcopy(rows)
    changed[0], changed[1] = changed[1], changed[0]
    write_json(args.run / 'schedules/train.json', changed)
    with pytest.raises(RuntimeError, match='binding|signature|schedule|cache|Cache'):
        runner.cached_split(args, 'train')


def _fixture_checkpoint(folder, step):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'step-{step:04d}.pt'
    path.write_bytes(f'CPU fixture optimizer state {step}'.encode())
    write_json(path.with_suffix('.json'), {'step': step, 'sha256': runner.sha256(path)})
    return path


def _fixture_evaluation(folder, step, score):
    _fixture_checkpoint(folder, step)
    return runner.commit_evaluation(folder, step, np.full((3, 9), step, np.float32),
                                    {'step': step, 'selection_score': [score]})


def test_evaluation_commit_survives_interruption_after_predictions_before_json(tmp_path, monkeypatch):
    _fixture_checkpoint(tmp_path, 250)
    real_write = runner.write_json

    def interrupt_commit(path, value):
        if path.name == 'evaluation-0250.json':
            predictions = tmp_path/'predictions-0250.npy'
            assert predictions.exists() and not predictions.with_suffix('.npy.tmp').exists()
            assert runner.sha256(predictions) == value['predictions_sha256']
            raise RuntimeError('fixture interruption before evaluation commit')
        real_write(path, value)

    monkeypatch.setattr(runner, 'write_json', interrupt_commit)
    with pytest.raises(RuntimeError, match='fixture interruption'):
        runner.commit_evaluation(tmp_path, 250, np.ones((3, 9), np.float32),
                                 {'step': 250, 'selection_score': [1.]})
    assert runner.completed_evaluations(tmp_path, 250, 3) == {}
    assert not (tmp_path/'evaluation-0250.json').exists()
    monkeypatch.setattr(runner, 'write_json', real_write)
    committed = runner.commit_evaluation(tmp_path, 250, np.ones((3, 9), np.float32),
                                         {'step': 250, 'selection_score': [1.]})
    assert runner.completed_evaluations(tmp_path, 250, 3) == {250: committed}


def test_partial_prediction_write_never_publishes_evaluation(tmp_path, monkeypatch):
    _fixture_checkpoint(tmp_path, 250)

    def interrupted_save(output, *args, **kwargs):
        output.write(b'incomplete numpy payload')
        raise RuntimeError('fixture interrupted prediction write')

    monkeypatch.setattr(runner.np, 'save', interrupted_save)
    with pytest.raises(RuntimeError, match='fixture interrupted'):
        runner.commit_evaluation(tmp_path, 250, np.ones((3, 9)),
                                 {'step': 250, 'selection_score': [1.]})
    assert not (tmp_path/'predictions-0250.npy').exists()
    assert not (tmp_path/'evaluation-0250.json').exists()
    assert runner.completed_evaluations(tmp_path, 250, 3) == {}


def test_resume_rebuilds_best_after_committed_evaluation_before_best_update(tmp_path):
    earlier = _fixture_evaluation(tmp_path, 250, 2.)
    runner.rebuild_best(tmp_path, {250: earlier})
    latest = _fixture_evaluation(tmp_path, 500, 1.)
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 250
    complete = runner.completed_evaluations(tmp_path, 500, 3)
    assert complete == {250: earlier, 500: latest}
    runner.rebuild_best(tmp_path, complete)
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 500


@pytest.mark.parametrize('damage', ['missing_predictions', 'changed_predictions', 'wrong_shape', 'wrong_checkpoint'])
def test_latest_incomplete_evaluation_is_recomputed_and_cannot_supply_best(tmp_path, damage):
    earlier = _fixture_evaluation(tmp_path, 250, 2.)
    _fixture_evaluation(tmp_path, 500, 1.)
    path = tmp_path/'predictions-0500.npy'
    if damage == 'missing_predictions':
        path.unlink()
    elif damage == 'changed_predictions':
        np.save(path, np.zeros((3, 9), np.float32))
    elif damage == 'wrong_shape':
        np.save(path, np.zeros((2, 9), np.float32))
    else:
        record_path = tmp_path/'evaluation-0500.json'
        record = json.loads(record_path.read_text())
        record['checkpoint_sha256'] = 'different checkpoint'
        write_json(record_path, record)
    complete = runner.completed_evaluations(tmp_path, 500, 3)
    assert complete == {250: earlier}
    runner.rebuild_best(tmp_path, complete)
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 250
    _fixture_evaluation(tmp_path, 500, 1.)
    runner.rebuild_best(tmp_path, runner.completed_evaluations(tmp_path, 500, 3))
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 500


def test_missing_earlier_evaluation_fails_instead_of_silently_removing_candidate(tmp_path):
    _fixture_checkpoint(tmp_path, 250)
    _fixture_evaluation(tmp_path, 500, 1.)
    with pytest.raises(RuntimeError, match='Incomplete earlier'):
        runner.completed_evaluations(tmp_path, 500, 3)


def test_best_is_removed_when_no_evaluation_committed(tmp_path):
    write_json(tmp_path/'best.json', {'step': 250})
    runner.rebuild_best(tmp_path, {})
    assert not (tmp_path/'best.json').exists()


def test_evaluation_cannot_bind_wrong_checkpoint_sidecar(tmp_path):
    checkpoint = _fixture_checkpoint(tmp_path, 250)
    checkpoint.write_bytes(b'changed fixture optimizer state')
    with pytest.raises(RuntimeError, match='match.*checkpoint'):
        runner.commit_evaluation(tmp_path, 250, np.ones((3, 9)),
                                 {'step': 250, 'selection_score': [1.]})
    assert not (tmp_path/'predictions-0250.npy').exists()


def test_resume_preserves_unpaired_latest_checkpoint_and_temporary_write(tmp_path):
    previous = _fixture_checkpoint(tmp_path, 250)
    orphan = tmp_path/'step-0500.pt'; orphan.write_bytes(b'uncommitted optimizer fixture')
    temporary = tmp_path/'step-0500.json.tmp'; temporary.write_bytes(b'partial sidecar fixture')
    assert runner.resume_checkpoint(tmp_path, 1250) == previous
    assert not orphan.exists() and not temporary.exists()
    archived = list((tmp_path/'interrupted').glob('checkpoint-write-*'))
    assert len(archived) == 1
    assert (archived[0]/orphan.name).read_bytes() == b'uncommitted optimizer fixture'
    assert (archived[0]/temporary.name).read_bytes() == b'partial sidecar fixture'
    recovery = json.loads((archived[0]/'recovery.json').read_text())
    assert recovery['resume_checkpoint'] == previous.name
    assert recovery['discarded_updates_may_require_repeated_computation'] is True
    assert runner.resume_checkpoint(tmp_path, 1250) == previous


def test_resume_preserves_first_interrupted_checkpoint_without_inventing_updates(tmp_path):
    incomplete = tmp_path/'step-0250.tmp'; incomplete.write_bytes(b'partial optimizer fixture')
    assert runner.resume_checkpoint(tmp_path, 1250) is None
    assert len(list((tmp_path/'interrupted').glob('*/step-0250.tmp'))) == 1


def test_resume_rejects_existing_sha_mismatch_without_falling_back_or_archiving(tmp_path):
    _fixture_checkpoint(tmp_path, 250)
    damaged = _fixture_checkpoint(tmp_path, 500)
    damaged.write_bytes(b'corrupted committed optimizer fixture')
    with pytest.raises(RuntimeError, match='SHA'):
        runner.resume_checkpoint(tmp_path, 1250)
    assert damaged.exists() and not (tmp_path/'interrupted').exists()


def test_resume_rejects_checkpoint_outside_group_budget(tmp_path):
    _fixture_checkpoint(tmp_path, 1500)
    with pytest.raises(RuntimeError, match='budget'):
        runner.resume_checkpoint(tmp_path, 1250)


def test_training_manifest_must_match_the_audited_original_sources(tmp_path):
    manifest = tmp_path/'fixture-manifest.json'
    manifest.write_text('{"abstract_fixture": 1}')
    args = SimpleNamespace(run=tmp_path/'run', manifest=manifest)
    write_json(args.run/'original-integrity-before.json', {'manifest_sha256': runner.sha256(manifest)})
    runner.check_training_manifest(args)
    manifest.write_text('{"abstract_fixture": 2}')
    with pytest.raises(RuntimeError, match='manifest.*audited'):
        runner.check_training_manifest(args)


def test_approved_training_and_checkpoint_budgets_are_unchanged():
    assert {group: value[2] * 64 for group, value in runner.GROUPS.items()} == {
        'S224': 320000, 'F224': 80000, 'J224': 80000, 'J384': 80000}
    assert sum(value[2] * 64 for value in runner.GROUPS.values()) == 560000
    assert sum(value[2] // 250 for value in runner.GROUPS.values()) == 35
    assert {group: 64 // value[3] for group, value in runner.GROUPS.items()} == {
        'S224': 1, 'F224': 1, 'J224': 4, 'J384': 16}
