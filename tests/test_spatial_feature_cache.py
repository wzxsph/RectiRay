"""Frozen-prefix equivalence, with synthetic weights and no GPU or case data."""
import numpy as np
import pytest

from ptrn_hip.model import keras, torch
from ptrn_hip.spatial_feature_cache import FrozenPrefixCache
from ptrn_hip.spatial_model import SpatialModel, coordinate_loss, parameter_groups


def tiny_model(arm):
    keras.backend.clear_session()
    keras.utils.set_random_seed(2041)
    image = keras.Input((224, 224, 3))
    x = keras.layers.AveragePooling2D(4)(image)
    x = keras.layers.Conv2D(8, 1, name='conv2_block6_concat')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(12, 1, name='conv3_block12_concat')(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(16, 1, name='conv4_block48_concat')(x)
    x = keras.layers.AveragePooling2D(2, name='pool4_pool')(x)
    x = keras.layers.BatchNormalization(name='conv5_bn')(x)
    x = keras.layers.Conv2D(20, 1, name='conv5_block32_concat')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1920, name='features1920')(x)
    backbone = keras.Model(image, x)
    kernel = np.zeros((1920, 9), np.float32)
    bias = np.zeros(9, np.float32)
    model = SpatialModel(backbone, kernel, bias, arm, 224, 'last_block', None)
    return model


def grads(model, fn, trainable, target):
    for p in trainable:
        p.grad = None
    output = fn()
    loss = coordinate_loss(output, target, 224)
    loss.backward()
    values = [None if p.grad is None else p.grad.detach().clone() for p in trainable]
    return output.detach(), values


@pytest.mark.parametrize('arm', ('gap_quad', 'spatial_quad'))
def test_cached_suffix_matches_online_output_and_trainable_gradients(arm):
    torch.set_num_threads(2)
    model = tiny_model(arm)
    groups = parameter_groups(model, 'last_block')
    trainable = [p for group in groups for p in group['params']]
    x = torch.randn((3, 224, 224, 3), dtype=torch.float32)
    target = torch.tensor([[[15., 16.], [205., 13.], [209., 202.], [12., 207.]]])
    indices = torch.tensor([2, 0])
    cached = FrozenPrefixCache(model, x, batch_size=1)
    assert not cached.pool4.requires_grad
    assert not any(t.requires_grad for t in cached.taps)
    bn_before = {k: v.clone() for k, v in model.state_dict().items()
                 if 'moving_mean' in k or 'moving_variance' in k}
    online_q, online_g = grads(model, lambda: model(x[indices]), trainable, target.repeat(2, 1, 1))
    cached_q, cached_g = grads(model, lambda: cached.forward(indices), trainable, target.repeat(2, 1, 1))
    torch.testing.assert_close(cached_q, online_q, atol=1e-5, rtol=1e-5)
    assert any(g is not None for g in online_g[len(groups[0]['params']):])
    for before, after in zip(online_g, cached_g):
        assert (before is None) == (after is None)
        if before is not None:
            torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)
    for key, before in bn_before.items():
        torch.testing.assert_close(model.state_dict()[key], before, atol=0, rtol=0)


def test_cache_rejects_head_stage_and_invalid_indices():
    model = tiny_model('gap_quad')
    x = torch.zeros((2, 224, 224, 3))
    model.set_stage('head')
    with pytest.raises(ValueError, match='last_block'):
        FrozenPrefixCache(model, x)
    model.set_stage('last_block')
    cache = FrozenPrefixCache(model, x)
    with pytest.raises(IndexError):
        cache.forward([2])
    with pytest.raises(IndexError):
        cache.forward([])
