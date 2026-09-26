"""V3 contract tests use generated fixtures, not private real images."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from ptrn_hip.partial_geometry import rectify_visible, visible_polygon, visibility, CONTRACT
from ptrn_hip.partial_synthesis import sample, pre_crop
from ptrn_hip.ratio_synthesis import pixel_corners, project
from ptrn_hip.partial_objectives import visible_coverage, partial_loss
from ptrn_hip.partial_runner import train_step, save_checkpoint, restore_checkpoint, ShuffledRounds
from ptrn_hip.partial_data import recipe, PartialDataset, sha256


@pytest.fixture
def source():
    a = np.zeros((160, 200, 3), np.uint8)
    yy, xx = np.mgrid[:160, :200]
    a[..., 0] = xx
    a[..., 1] = yy
    a[..., 2] = (xx+yy) % 256
    return a


@pytest.mark.parametrize('integrity', ['complete', 'light', 'moderate'])
@pytest.mark.parametrize('family', ['viewer_screen', 'coco_context', 'near_color', 'mixed_color'])
def test_scene_labels_and_determinism(source, integrity, family):
    kwargs = dict(integrity=integrity, family=family, source_crop=True, background=np.flip(source, axis=0).copy())
    view = sample(source, 11, 7, **kwargs)
    again = sample(source, 11, 7, **kwargs)
    assert np.array_equal(view.photo, again.photo)
    np.testing.assert_allclose(view.quad, again.quad)
    x, y, r, b = view.info['source_pre_crop']
    assert (r-x)*(b-y)/(source.shape[0]*source.shape[1]) >= .7
    assert view.ratio == (r-x)/(b-y)
    np.testing.assert_allclose(project(pixel_corners(r-x, b-y), view.transform), view.quad, atol=1e-4)
    assert view.info['occupancy'] >= .5
    missing = view.info['missing_fraction']
    if integrity == 'complete': assert missing < 1e-5
    elif integrity == 'light': assert 0 < missing <= .1
    else: assert .1 < missing <= .3
    if integrity != 'complete': assert not all(view.info['corner_visible'])
    batch = view.input(224)
    assert batch['mask'].shape == (224, 224)
    assert batch['quad'].shape == (4, 2)
    assert np.max(batch['mask']-batch['photo_valid']) < 1e-5


def test_keep_pixel_exact_and_never_mutate_source(source):
    before = source.copy()
    v = sample(source, 0, 0, kind='keep')
    result = rectify_visible(source, None, None, trusted_original=True)
    assert np.array_equal(result.pixels, source) and np.array_equal(v.photo, source)
    assert result.valid_mask.min() == 255
    assert np.array_equal(source, before)
    with pytest.raises(ValueError): sample(source, 0, 0, kind='keep', source_crop=True)


def test_minification_suppresses_alias_stripes_without_changing_smooth_detail():
    height, width = 240, 320
    yy, xx = np.mgrid[:height, :width]
    photo = np.stack([(xx % 2)*255, xx*255/(width-1), yy*255/(height-1)], axis=2).astype(np.uint8)
    quad = pixel_corners(width, height)
    result = rectify_visible(photo, quad, (width-1)/(height-1), max_side=73)
    assert result.info['resampling'] == 'area_3x'
    old = cv2.warpPerspective(photo, result.photo_to_output, tuple(result.info['output_size']),
                              flags=cv2.INTER_LINEAR)
    improved = result.pixels[10:-10, 10:-10]
    reference = old[10:-10, 10:-10]
    stripe_error = np.abs(improved[..., 0].astype(np.float32)-127.5).mean()
    old_stripe_error = np.abs(reference[..., 0].astype(np.float32)-127.5).mean()
    assert stripe_error < 25 and stripe_error < old_stripe_error/2
    assert np.max(np.abs(improved[..., 1:].astype(np.int16)-reference[..., 1:].astype(np.int16))) <= 2


def test_magnified_and_mixed_scale_output_uses_original_bilinear(source):
    magnified = rectify_visible(source, [[80, 40], [119, 40], [119, 79], [80, 79]],
                                1, max_side=320)
    assert magnified.info['resampling'] == 'bilinear'
    assert magnified.pixels.shape[:2] == (200, 200)
    old = cv2.warpPerspective(source, magnified.photo_to_output,
                              tuple(magnified.info['output_size']), flags=cv2.INTER_LINEAR)
    old[magnified.valid_mask == 0] = 0
    np.testing.assert_array_equal(magnified.pixels, old)

    larger_source = cv2.resize(source, (320, 240), interpolation=cv2.INTER_LINEAR)
    trapezoid = [[10, 10], [310, 10], [200, 200], [120, 200]]
    mixed = rectify_visible(larger_source, trapezoid, 1.6, max_side=160)
    assert mixed.info['resampling'] == 'bilinear'
    old = cv2.warpPerspective(larger_source, mixed.photo_to_output,
                              tuple(mixed.info['output_size']), flags=cv2.INTER_LINEAR)
    old[mixed.valid_mask == 0] = 0
    np.testing.assert_array_equal(mixed.pixels, old)


def test_partial_minification_keeps_old_mask_and_has_no_black_edge_halo():
    photo = np.full((256, 320, 3), 200, np.uint8)
    quad = [[-100, -80], [319, -80], [319, 255], [-100, 255]]
    result = rectify_visible(photo, quad, 419/335, max_side=80)
    assert result.info['resampling'] == 'area_3x'
    assert result.info['outside_corners'] > 0
    width, height = result.info['output_size']
    expected_mask = np.zeros((height, width), np.uint8)
    poly = visible_polygon(quad, photo.shape[1], photo.shape[0])
    cv2.fillConvexPoly(expected_mask,
                       np.rint(project(poly, result.photo_to_output)).astype(np.int32), 255)
    support = cv2.warpPerspective(np.full(photo.shape[:2], 255, np.uint8),
                                  result.photo_to_output, (width, height), flags=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(result.valid_mask, np.minimum(expected_mask, support))
    assert np.any(result.valid_mask == 0)
    assert np.all(result.pixels[result.valid_mask > 0] == 200)
    assert np.all(result.pixels[result.valid_mask == 0] == 0)


@pytest.mark.parametrize('quad', [
    [[-25, 0], [199, 0], [199, 159], [-25, 159]],
    [[-30, -20], [180, 10], [190, 150], [5, 140]],
    [[0, -40], [199, -40], [199, 159], [0, 159]],
])
def test_oracle_outside_corners_render_visible_content(source, quad):
    result = rectify_visible(source, quad, 1.25, max_side=256)
    assert result.pixels.size and result.valid_mask.sum() > 0
    assert result.info['outside_corners'] > 0
    q = project(np.asarray(quad), result.photo_to_output)
    assert abs((q[1, 0]-q[0, 0])/(q[3, 1]-q[0, 1])-1.25) < 1e-5
    assert np.max(np.abs(q[[0, 1], 1]-q[0, 1])) < 1e-5
    assert np.all(result.pixels[result.valid_mask == 0] == 0)
    # Output validity must correspond to actual captured pixels, including cut edges.
    yy, xx = np.where(result.valid_mask > 0)
    photo = project(np.stack([xx, yy], axis=1), np.linalg.inv(result.photo_to_output))
    assert photo[:, 0].min() >= -.6 and photo[:, 1].min() >= -.6
    assert photo[:, 0].max() <= 199.6 and photo[:, 1].max() <= 159.6


@pytest.mark.parametrize('q', [np.zeros((4, 2)), [[0, 0], [0, 99], [99, 99], [99, 0]],
                             [[0, 0], [99, 99], [99, 0], [0, 99]], [[np.nan, 0]]*4])
def test_reject_mirror_fold_degenerate(source, q):
    with pytest.raises(ValueError): rectify_visible(source, q, 1)


def test_outside_truth_zero_coverage_loss_and_inward_crop_penalty():
    truth = torch.tensor([[[-20., -20.], [243., -20.], [243., 243.], [-20., 243.]]])
    support = torch.ones((1, 224, 224))
    pred = truth.clone().requires_grad_()
    missed, bg = visible_coverage(pred, truth, support, 224)
    assert float(missed.detach()) == 0 and float(bg.detach()) == 0
    inward = torch.tensor([[[15., 15.], [208., 15.], [208., 208.], [15., 208.]]], requires_grad=True)
    missed, bg = visible_coverage(inward, truth, support, 224)
    assert missed > .1 and bg < 1e-5
    missed.backward()
    assert torch.isfinite(inward.grad).all()


def test_balanced_schedule_all_sources_and_scene_freeze():
    rows = [recipe(i, v, round_index=0, seed=123, domain='train') for i in range(100) for v in range(16)]
    photo = [r for r in rows if r['kind'] == 'photo']
    assert len(photo) == 1280
    assert sum(r['high_occupancy'] for r in photo)/len(photo) >= .8
    assert {r['family'] for r in photo} == {'viewer_screen', 'coco_context', 'near_color', 'mixed_color'}
    assert rows == [recipe(i, v, round_index=0, seed=123, domain='train') for i in range(100) for v in range(16)]
    assert rows != [recipe(i, v, round_index=1, seed=123, domain='train') for i in range(100) for v in range(16)]


class FakeDataset:
    seed = 123
    def __len__(self): return 37
    def __getitem__(self, i): return self.round_index, i


def test_round_shuffle_resume_does_not_omit_or_duplicate():
    stream = ShuffledRounds(FakeDataset())
    first = [stream[i] for i in range(37)]
    assert sorted(i for _, i in first) == list(range(37))
    assert set(r for r, _ in first) == {0}
    second = [stream[i] for i in range(37, 74)]
    assert first != second and sorted(i for _, i in second) == list(range(37))
    restored = ShuffledRounds(FakeDataset())
    assert [stream[i] for i in range(31, 45)] == [restored[i] for i in range(31, 45)]


@pytest.fixture
def model(monkeypatch):
    from test_spatial_model import _tiny_g0
    from ptrn_hip import spatial_model
    from ptrn_hip.model import keras
    from ptrn_hip.partial_model import PartialModel
    torch.set_num_threads(2)
    monkeypatch.setattr(spatial_model, 'build_unified', _tiny_g0)
    keras.backend.clear_session()
    keras.utils.set_random_seed(125)
    # A Torch .to('cpu') does not change Keras's default tensor device. Keep
    # its encoder on CPU for the entire fixture, even when CUDA is visible.
    with keras.device('cpu'):
        spatial = spatial_model.build_spatial('fixture', 'fixture', 'spatial_quad', stage='last_block')
        yield PartialModel(spatial, feature_dim=20).to('cpu')


def test_unbounded_readout_and_finite_collapsed_loss(model):
    with torch.no_grad():
        model.spatial.direct.weight.zero_()
        model.spatial.direct.bias.copy_(torch.tensor([-.1, -.1, 1.1, -.1, 1.1, 1.1, -.1, 1.1]))
    out = model(torch.zeros((1, 224, 224, 3)))
    assert out['quad'].min() < 0 and out['quad'].max() > 223
    collapsed = out['quad']*0+111.5
    out['quad'] = collapsed
    batch = {'quad': torch.tensor([[[0., 0.], [223., 0.], [223., 223.], [0., 223.]]]),
             'mask': torch.ones((1, 224, 224)), 'photo_valid': torch.ones((1, 224, 224)), 'log_ratio': torch.zeros(1)}
    loss = partial_loss(out, batch, 224)['total']
    loss.backward()
    assert torch.isfinite(loss)


def make_batch(n=2):
    return {'pixels': np.full((n, 224, 224, 3), 80, np.uint8),
            'quad': np.tile(np.array([[-5., 10.], [215., 4.], [225., 219.], [-5., 218.]], np.float32), (n, 1, 1)),
            'mask': np.ones((n, 224, 224), np.float32), 'photo_valid': np.ones((n, 224, 224), np.float32),
            'log_ratio': np.zeros(n, np.float32)}


def test_optimizer_bn_checkpoint_resume_and_binding(model, tmp_path):
    optimizer = torch.optim.Adam(model.optimizer_groups('last_block'))
    before = {k: v.clone() for k, v in model.state_dict().items() if 'moving_' in k}
    assert before
    bn_params = {id(v.value) for l in model.spatial.encoder.layers if 'bn' in l.name for v in l.weights}
    assert not bn_params.intersection(id(p) for g in optimizer.param_groups for p in g['params'])
    terms = train_step(model, optimizer, [make_batch(1), make_batch(1)], device='cpu', effective_batch=2)
    assert np.isfinite(list(terms.values())).all()
    for k, v in before.items(): torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    binding = {'effective_batch': 2, 'data': 'fixture'}
    path = tmp_path/'step.pt'
    save_checkpoint(path, model, optimizer, step=1, cursor=2, binding=binding)
    expected = train_step(model, optimizer, [make_batch()], device='cpu', effective_batch=2)
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    assert restore_checkpoint(path, model, optimizer, binding) == (1, 2)
    actual = train_step(model, optimizer, [make_batch()], device='cpu', effective_batch=2)
    assert expected == actual
    for k, v in parameters.items(): torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)
    with pytest.raises(ValueError): restore_checkpoint(path, model, optimizer, {'effective_batch': 2, 'data': 'changed'})


def test_wrong_contract_rejected(model, tmp_path):
    p = tmp_path/'legacy.pt'
    torch.save({'contract': 'ptrn-spatial-corners-v2'}, p)
    with pytest.raises(ValueError): restore_checkpoint(p, model, torch.optim.Adam(model.optimizer_groups('head')), {})


def test_asset_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError): PartialDataset.safe_path(tmp_path, '../outside.jpg')


def test_oracle_visible_grid_distinguishes_missing_input_from_new_crop(source):
    from ptrn_hip.partial_evaluation import measure
    view = sample(source, 3, 9, integrity='moderate', family='near_color')
    result = measure(view, view.quad, view.ratio)
    assert result['valid_geometry']
    assert result['input_missing_fraction'] > .1
    assert result['additional_crop_fraction'] < .005
    assert result['visible_grid_max'] < 1e-5
    smaller = (view.quad-view.quad.mean(0))*.9+view.quad.mean(0)
    worse = measure(view, smaller, view.ratio)
    assert worse['additional_crop_fraction'] > result['additional_crop_fraction']+.05


def test_accumulation_matches_full_batch_update(model):
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.optimizer_groups('head'))
    full = train_step(model, optimizer, [make_batch(2)], device='cpu', effective_batch=2)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.optimizer_groups('head'))
    accumulated = train_step(model, optimizer, [make_batch(1), make_batch(1)], device='cpu', effective_batch=2)
    assert abs(full['total']-accumulated['total']) < 1e-5
    for k, v in expected.items(): torch.testing.assert_close(model.state_dict()[k], v, atol=2e-6, rtol=1e-5)


@pytest.fixture
def portable(tmp_path, source):
    bundle = tmp_path/'bundle'
    bg = tmp_path/'backgrounds'
    (bundle/'data/films').mkdir(parents=True)
    bg.mkdir()
    rows = []
    splits = {'train': [0, 1, 2], 'dev': [3, 4], 'quarantine': [5]}
    for i in range(6):
        path = bundle/f'data/films/{i}.png'
        cv2.imwrite(str(path), source+i)
        split = next(k for k, ids in splits.items() if i in ids)
        rows.append({'source_path': str(path.relative_to(bundle)), 'source_sha256': sha256(path), 'split': split})
    backgrounds = []
    for i, split in enumerate(('train', 'dev')):
        path = bg/f'{split}.png'
        cv2.imwrite(str(path), source+10+i)
        backgrounds.append({'id': split, 'path': path.name, 'sha256': sha256(path), 'split': split, 'source_kind': 'real_photo'})
    (bundle/'data/manifest.json').write_text(json.dumps({'records': rows}))
    (bundle/'data/split.json').write_text(json.dumps(splits))
    (bundle/'data/backgrounds-manifest.json').write_text(json.dumps({'rows': backgrounds}))
    return bundle, bg


def test_dataset_split_quarantine_hash_and_background_binding(portable):
    bundle, bg = portable
    train = PartialDataset(bundle, bg)
    dev = PartialDataset(bundle, bg, split='dev')
    assert len(train) == 48 and len(dev) == 24
    assert not set(train.ids).intersection(dev.ids)
    assert 5 not in train.ids+dev.ids
    assert all(r['split'] == 'train' for r in train.bg)
    assert all(r['split'] == 'dev' for r in dev.bg)
    with pytest.raises(ValueError): PartialDataset(bundle, bg, split='dev', round_index=1)
    (bundle/'data/films/0.png').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA'): PartialDataset(bundle, bg)


def test_cross_split_background_rejected(portable):
    bundle, bg = portable
    p = bundle/'data/backgrounds-manifest.json'
    data = json.loads(p.read_text())
    data['rows'][1]['sha256'] = data['rows'][0]['sha256']
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='overlap'): PartialDataset(bundle, bg)


def test_worker_data_loading_matches_single_process(portable):
    from torch.utils.data import DataLoader
    bundle, bg = portable
    dataset = PartialDataset(bundle, bg)
    indices = [0, 1, 2, 47, 48, 49]
    direct = DataLoader(ShuffledRounds(dataset), sampler=indices, batch_size=2, num_workers=0)
    parallel = DataLoader(ShuffledRounds(dataset), sampler=indices, batch_size=2, num_workers=2,
                          multiprocessing_context='spawn')
    for a, b in zip(direct, parallel, strict=True):
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_384_uses_same_unbounded_contract(monkeypatch):
    from test_spatial_model import _tiny_g0
    from ptrn_hip import spatial_model
    from ptrn_hip.model import keras
    from ptrn_hip.partial_model import PartialModel, RawPartial
    monkeypatch.setattr(spatial_model, 'build_unified', _tiny_g0)
    with keras.device('cpu'):
        net = PartialModel(spatial_model.build_spatial('fixture', 'fixture', 'spatial_quad', size=384), feature_dim=20).to('cpu')
        out = RawPartial(net)(torch.zeros((1, 384, 384, 3)))
    assert out[0].shape == (1, 4, 2)
    assert out[2].shape == (1, 1, 96, 96)
    assert net.contract()['contract'] == CONTRACT
    assert all(torch.isfinite(v).all() for v in out)
