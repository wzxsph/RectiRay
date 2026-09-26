"""CPU checks for the four-arm research model, using synthetic encoder weights."""
import numpy as np
import pytest

from ptrn_hip.model import keras, torch
from ptrn_hip import spatial_model as spatial


def _tiny_g0(_weights, _g0_checkpoint, size):
    inp = keras.Input((None, None, 3))
    x = keras.layers.AveragePooling2D(4)(inp)
    x = keras.layers.Conv2D(8, 1, name="conv2_block6_concat")(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(12, 1, name="conv3_block12_concat")(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.Conv2D(16, 1, name="conv4_block48_concat")(x)
    x = keras.layers.AveragePooling2D(2)(x)
    x = keras.layers.BatchNormalization(name="conv5_bn")(x)
    x = keras.layers.Conv2D(20, 1, name="conv5_block32_concat")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1920, name="features1920")(x)
    backbone = keras.Model(inp, x)
    photo = keras.Input((size, size, 3))
    head = keras.layers.Dense(9, dtype="float32", name="unified_geometry")
    model = keras.Model(photo, head(backbone(photo, training=False)))
    kernel, bias = head.get_weights()
    kernel[:] = 0
    bias[:] = [.7, 0, 0, 0, .7, 0, 0, 0, 0]
    head.set_weights([kernel, bias])
    return model, backbone, head


@pytest.fixture
def assets(monkeypatch):
    keras.backend.clear_session()
    keras.utils.set_random_seed(125)
    torch.set_num_threads(2)
    monkeypatch.setattr(spatial, "build_unified", _tiny_g0)
    return "fixture-b11", "fixture-g0"


@pytest.mark.parametrize("arm", sorted(spatial.ARMS))
def test_arm_forward_cache_loss_and_optimizer(assets, arm):
    model = spatial.build_spatial(*assets, arm, size=224, stage="head")
    x = torch.zeros((2, 224, 224, 3))
    features = model.extract_features(x)
    online = model(x)
    cached = model.forward_features(features)
    assert online.shape == (2, 4, 2)
    torch.testing.assert_close(online, cached)
    assert torch.isfinite(online).all()
    assert not model.encoder.trainable_weights
    if arm == "spatial_heatmap":
        # The zero-initialized readout must give the same uniform heatmap to
        # every corner. Its FP32 reduction can vary slightly across CPUs.
        logits = model._spatial_logits(features)
        torch.testing.assert_close(logits, torch.zeros_like(logits), atol=0, rtol=0)
        centre = (model.size - 1) / 2
        corner_agreement_budget = 4 * torch.finfo(online.dtype).eps * centre
        torch.testing.assert_close(
            online, online[:, :1].expand_as(online), atol=corner_agreement_budget, rtol=0,
        )
        fp32_reduction_budget = 8 * torch.finfo(online.dtype).eps * (model.size - 1)
        torch.testing.assert_close(
            online, torch.full_like(online, centre), atol=fp32_reduction_budget, rtol=0,
        )
    target = torch.tensor([[[12., 15.], [210., 14.], [207., 201.], [10., 203.]]]).repeat(2, 1, 1)
    loss = spatial.coordinate_loss(cached, target, 224)
    assert torch.isfinite(loss)
    groups = spatial.parameter_groups(model, "head")
    assert len(groups) == 1 and groups[0]["lr"] == 1e-4
    optimizer = torch.optim.Adam(groups)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in groups[0]["params"] if p.grad is not None)
    optimizer.step()


def test_last_block_only_opens_conv5_and_keeps_bn_frozen(assets):
    model = spatial.build_spatial(*assets, "spatial_quad", stage="last_block")
    groups = spatial.parameter_groups(model, "last_block")
    assert [group["lr"] for group in groups] == [1e-4, 1e-5]
    assert model.encoder.get_layer("conv4_block48_concat").trainable is False
    assert model.encoder.get_layer("conv5_block32_concat").trainable is True
    assert model.encoder.get_layer("conv5_bn").trainable is False
    bn_keys = [key for key in model.state_dict() if "moving_mean" in key or "moving_variance" in key]
    assert bn_keys, "BN state assertion must never be vacuous"
    x = torch.zeros((1, 224, 224, 3))
    q = model(x)
    spatial.coordinate_loss(q, q.detach() + 1, 224).backward()
    assert any(p.grad is not None for p in groups[1]["params"])


def test_coordinate_loss_accepts_temporarily_invalid_quad_and_rejects_nonfinite():
    collapsed = torch.full((1, 4, 2), 111.5, requires_grad=True)
    target = torch.tensor([[[0., 0.], [223., 0.], [223., 223.], [0., 223.]]])
    loss = spatial.coordinate_loss(collapsed, target, 224)
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    with pytest.raises(FloatingPointError):
        spatial.coordinate_loss(torch.full((1, 4, 2), float("nan")), target, 224)
    with pytest.raises(ValueError):
        spatial.coordinate_loss(collapsed, target[:, :3], 224)


def test_heatmap_boundary_grid_and_raw_pixel_wrapper(assets):
    model = spatial.build_spatial(*assets, "spatial_heatmap", size=224)
    # Force four corner logits to their respective extreme output cells.
    features = model.extract_features(torch.zeros((1, 224, 224, 3)))
    logits = torch.full((1, 4, 56, 56), -100.)
    logits[0, 0, 0, 0] = 100.
    logits[0, 1, 0, -1] = 100.
    logits[0, 2, -1, -1] = 100.
    logits[0, 3, -1, 0] = 100.
    model._spatial_logits = lambda unused: logits
    q = model.forward_features(features)
    torch.testing.assert_close(q[0], torch.tensor([[0., 0.], [223., 0.], [223., 223.], [0., 223.]]), atol=1e-5, rtol=0)
    # Build a separate model because the test-only method override is not exportable.
    raw = spatial.build_spatial(*assets, "gap_quad").raw_pixels()
    pixels = torch.full((1, 224, 224, 3), 100.)
    expected = raw.model((pixels / 255 - raw.mean) / raw.std)
    torch.testing.assert_close(raw(pixels), expected)
    assert spatial.contract(224)["output_shape"] == [None, 4, 2]


def test_384_spatial_readout_keeps_pixel_centre_contract(assets):
    model = spatial.build_spatial(*assets, "spatial_quad", size=384)
    output = model(torch.zeros((1, 384, 384, 3)))
    assert output.shape == (1, 4, 2)
    assert torch.isfinite(output).all()
    assert spatial.contract(384, "spatial_quad")["spatial_readout_grid"] == 56


@pytest.mark.parametrize("arm", ["gap_quad", "spatial_heatmap"])
def test_raw_rgb_cpu_trace_has_same_coordinates(assets, arm):
    raw = spatial.build_spatial(*assets, arm).raw_pixels().eval()
    pixels = torch.full((1, 224, 224, 3), 87.)
    with torch.no_grad():
        expected = raw(pixels)
        traced = torch.jit.trace(raw, pixels, check_trace=False)
        torch.testing.assert_close(traced(pixels), expected, atol=1e-4, rtol=1e-5)
