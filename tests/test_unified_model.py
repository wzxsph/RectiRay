"""Geometry, checkpoint and deployment contract tests using synthetic inputs."""
import json
import math

import numpy as np
import pytest

from ptrn_hip import unified_model as unified
from ptrn_hip.model import keras, torch
from ptrn_hip.ratio_data import sha256


def _write_checkpoint(path, model, config, step=1):
    payload = {"state_dict": model.state_dict(), "step": step, "config": config}
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps({"sha256": sha256(path), "step": step, "config": config}))
    return path


def _tiny_ratio_model(weights, size, with_ratio, mean_log_ratio):
    """Small image encoder with the same f1920 and named conv5/BN boundary."""
    inputs = keras.Input((None, None, 3), name="tiny_pixels")
    pooled = keras.layers.GlobalAveragePooling2D(name="tiny_pool")(inputs)
    early = keras.layers.Dense(6, name="conv4_projection")(pooled)
    late = keras.layers.Dense(1920, name="conv5_projection")(early)
    features = keras.layers.BatchNormalization(name="conv5_bn")(late)
    backbone = keras.Model(inputs, features, name="tiny_encoder")
    pixels = keras.Input((size, size, 3), name="tiny_photo")
    h = keras.layers.Dense(8, dtype="float32", name="homography")(backbone(pixels, training=False))
    model = keras.Model(pixels, h, name="tiny_g0")
    kernel, bias = model.get_layer("homography").get_weights()
    kernel[:] = np.random.default_rng(41).normal(0, .0001, kernel.shape)
    bias[:] = [.7, 0, .01, 0, .6, -.02, 0, 0]
    model.get_layer("homography").set_weights([kernel, bias])
    return model, backbone


@pytest.fixture
def assets(tmp_path, monkeypatch):
    keras.backend.clear_session()
    keras.utils.set_random_seed(71)
    torch.set_num_threads(2)
    weights = tmp_path / "fixture.weights.h5"
    weights.write_bytes(b"synthetic architecture initialization fixture")
    monkeypatch.setattr(unified, "B11_SHA", sha256(weights))
    monkeypatch.setattr(unified, "build_ratio", _tiny_ratio_model)
    model, _ = _tiny_ratio_model(weights, 224, False, 0)
    config = {"group": "G0", "input_size": 224, "ratio_head": False,
              "initial_weights_sha256": sha256(weights)}
    checkpoint = _write_checkpoint(tmp_path / "g0.pt", model, config, step=1250)
    monkeypatch.setattr(unified, "G0_SHA", sha256(checkpoint))
    return weights, checkpoint, model


def test_initialization_copies_g0_and_adds_one_zero_residual(assets):
    weights, checkpoint, old = assets
    model, backbone, head = unified.build_unified(weights, checkpoint)
    original_kernel, original_bias = old.get_layer("homography").get_weights()
    kernel, bias = head.get_weights()
    np.testing.assert_array_equal(kernel[:, :8], original_kernel)
    np.testing.assert_array_equal(bias[:8], original_bias)
    assert np.count_nonzero(kernel[:, 8]) == np.count_nonzero(bias[8:]) == 0
    assert head.count_params() == 17289
    assert len(model.layers) == 3  # Input, encoder, the only Dense9 prediction layer.
    pixels = np.full((2, 224, 224, 3), 117, dtype=np.uint8)
    with torch.no_grad():
        expected = old(unified.tensor_input(pixels, "cpu"), training=False).numpy()
    actual = unified.predict(model, pixels, "cpu")
    np.testing.assert_allclose(actual[:, :8], expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(actual[:, 8], 0)
    assert not backbone.trainable_weights


def test_initialization_rejects_unapproved_weight_or_g0_sha(assets, tmp_path):
    weights, checkpoint, _ = assets
    other = tmp_path / "other.weights.h5"
    other.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="B1.1.*SHA"):
        unified.build_unified(other, checkpoint)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA"):
        unified.build_unified(weights, checkpoint)


def test_residual_decode_uses_photo_coordinates_and_stops_edge_gradient():
    prediction = torch.tensor([[.8, 0, .1, 0, .6, -.1, 0, 0, .2]], requires_grad=True)
    # Deliberately non-isotropic test mapping distinguishes input-space and photo-space ratios.
    inverse = torch.tensor([[[2., 0, 10], [0, 4., 20], [0, 0, 1.]]])
    network, photo, log_ratio = unified.decode_prediction(prediction, 224, inverse)
    torch.testing.assert_close(photo, network * torch.tensor([2., 4.]) + torch.tensor([10., 20.]))
    assert log_ratio.item() == pytest.approx(math.log((.8 / .6) / 2) + .2, abs=1e-6)
    log_ratio.sum().backward()
    torch.testing.assert_close(prediction.grad[:, :8], torch.zeros((1, 8)))
    torch.testing.assert_close(prediction.grad[:, 8], torch.ones(1))


def test_losses_supervise_current_h_and_delta_separately():
    prediction = torch.tensor([[.7, 0, 0, 0, .6, 0, 0, 0, .1]], requires_grad=True)
    quad = unified.decode_prediction(prediction, 224)[0].detach() + 2
    truth_ratio = torch.tensor([math.log(.7 / .6)])
    total, q_loss, r_loss = unified.losses(prediction, quad, truth_ratio, 224)
    assert total.item() == pytest.approx(q_loss.item() + .25 * r_loss.item())
    r_loss.backward(retain_graph=True)
    torch.testing.assert_close(prediction.grad[:, :8], torch.zeros((1, 8)))
    assert prediction.grad[0, 8] != 0
    prediction.grad.zero_()
    q_loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :8]) > 0
    assert prediction.grad[0, 8] == 0


@pytest.mark.parametrize("kind", ["nan", "horizon", "reflected", "collapsed", "overflow"])
def test_invalid_geometry_is_not_clamped_or_skipped(kind):
    prediction = torch.tensor([[.7, 0, 0, 0, .6, 0, 0, 0, 0]])
    if kind == "nan":
        prediction[0, 8] = float("nan")
    elif kind == "horizon":
        prediction[0, 6] = 1
    elif kind == "reflected":
        prediction[0, 0] = -.7
    elif kind == "collapsed":
        prediction[0, 4] = 0
    else:
        prediction[0, 0] = prediction[0, 4] = 1e20
    with pytest.raises(FloatingPointError):
        unified.decode_prediction(prediction, 224)


def test_strict_shapes_prevent_target_broadcast_and_wrong_contract():
    prediction = torch.tensor([[.7, 0, 0, 0, .6, 0, 0, 0, 0]])
    quad = unified.decode_prediction(prediction, 224)[0]
    with pytest.raises(ValueError, match="Targets"):
        unified.losses(prediction, quad, torch.zeros((1, 1)), 224)
    with pytest.raises(ValueError, match="inverse_letterbox"):
        unified.decode_prediction(prediction, 224, torch.eye(3))
    with pytest.raises(ValueError, match="shape"):
        unified.decode_prediction(prediction[:, :8], 224)
    with pytest.raises(ValueError, match="224 or 384"):
        unified.contract(512)


@pytest.mark.parametrize("stage", ["head", "last_block"])
def test_all_head_outputs_update_while_bn_stays_frozen(assets, stage):
    weights, checkpoint, _ = assets
    model, backbone, head = unified.build_unified(weights, checkpoint)
    groups = unified.parameter_groups(model, backbone, head, stage)
    assert [g["lr"] for g in groups] == ([1e-4] if stage == "head" else [1e-4, 1e-5])
    frozen_bn = {v.path: v.value.detach().clone() for layer in backbone.layers
                 if isinstance(layer, keras.layers.BatchNormalization) for v in layer.weights}
    early_before = backbone.get_layer("conv4_projection").get_weights()
    kernel_before, bias_before = head.get_weights()
    optimizer = torch.optim.Adam(groups)
    pixels = np.full((2, 224, 224, 3), 100, dtype=np.uint8)
    # A training=True caller must still never update nested BN statistics.
    prediction = model(unified.tensor_input(pixels, "cpu"), training=True)
    target = unified.decode_prediction(prediction, 224)[0].detach() + 1
    total, _, _ = unified.losses(prediction, target, torch.zeros(2), 224)
    total.backward()
    optimizer.step()
    kernel_after, bias_after = head.get_weights()
    assert np.any(kernel_after[:, :8] != kernel_before[:, :8])
    assert np.any(kernel_after[:, 8] != kernel_before[:, 8])
    assert np.any(bias_after[:8] != bias_before[:8]) and bias_after[8] != bias_before[8]
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            for v in layer.weights:
                torch.testing.assert_close(v.value, frozen_bn[v.path], rtol=0, atol=0)
    for actual, original in zip(backbone.get_layer("conv4_projection").get_weights(), early_before, strict=True):
        np.testing.assert_array_equal(actual, original)
    if stage == "last_block":
        with pytest.raises(ValueError, match="frozen encoder"):
            unified.feature_predict(backbone, pixels, "cpu")


def test_cached_and_online_outputs_losses_and_gradients_match(assets):
    weights, checkpoint, _ = assets
    model, backbone, head = unified.build_unified(weights, checkpoint)
    rng = np.random.default_rng(70)
    pixels = rng.integers(0, 256, (3, 224, 224, 3), dtype=np.uint8)
    online = model(unified.tensor_input(pixels, "cpu"), training=False)
    targets = unified.decode_prediction(online, 224)[0].detach() + 1
    ratios = torch.zeros(3)
    online_loss = unified.losses(online, targets, ratios, 224)
    online_loss[0].backward()
    gradients = [v.value.grad.clone() for v in head.trainable_weights]
    for v in head.trainable_weights:
        v.value.grad = None
    cached = unified.feature_predict(backbone, pixels, "cpu")
    predicted = unified.forward_features(head, cached)
    cached_loss = unified.losses(predicted, targets, ratios, 224)
    cached_loss[0].backward()
    torch.testing.assert_close(predicted, online, rtol=0, atol=0)
    for actual, expected in zip(cached_loss, online_loss, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, expected in zip(head.trainable_weights, gradients, strict=True):
        torch.testing.assert_close(value.value.grad, expected, rtol=0, atol=0)
    # An H update must change cached-path corners; cached f never contains old q.
    with torch.no_grad():
        head.bias.value[2] += .05
    new_quad = unified.decode_prediction(unified.forward_features(head, cached), 224)[0]
    assert not torch.equal(new_quad, unified.decode_prediction(predicted, 224)[0])


def test_accumulated_gradient_matches_full_batch():
    torch.manual_seed(72)
    head = torch.nn.Linear(1920, 9)
    with torch.no_grad():
        head.weight.mul_(.002)
        head.bias.copy_(torch.tensor([.7, 0, 0, 0, .6, 0, 0, 0, .01]))
    features = torch.randn(64, 1920)
    quad = torch.tensor([[[30., 40.], [190., 40.], [190., 180.], [30., 180.]]]).repeat(64, 1, 1)
    ratios = torch.full((64,), .1)
    unified.losses(head(features), quad, ratios, 224)[0].backward()
    expected = [parameter.grad.clone() for parameter in head.parameters()]
    head.zero_grad(set_to_none=True)
    for start in range(0, 64, 16):
        (unified.losses(head(features[start:start + 16]), quad[start:start + 16], ratios[start:start + 16], 224)[0] / 4).backward()
    for parameter, want in zip(head.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, want, rtol=1e-4, atol=2e-6)


def test_checkpoint_roundtrip_supports_384_and_rejects_historical_log_ratio(assets, tmp_path):
    weights, checkpoint, _ = assets
    model, _, head = unified.build_unified(weights, checkpoint)
    with torch.no_grad():
        head.bias.value[8] = .12
    config = {"input_size": 224, "contract": unified.contract(224)}
    candidate = _write_checkpoint(tmp_path / "candidate.pt", model, config)
    restored, _, _ = unified.load_unified(weights, checkpoint, candidate, size=384)
    assert restored.input_shape == (None, 384, 384, 3)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    historical = {"input_size": 224, "contract": {"contract": "ptrn-ratio-v1-rgb224-h8-logratio9"}}
    _write_checkpoint(candidate, model, historical)
    with pytest.raises(ValueError, match="explicit H8\\+delta"):
        unified.load_unified(weights, checkpoint, candidate)


def test_checkpoint_sidecar_config_and_hash_are_both_enforced(assets, tmp_path):
    weights, checkpoint, _ = assets
    model, _, _ = unified.build_unified(weights, checkpoint)
    config = {"input_size": 224, "contract": unified.contract(224)}
    candidate = _write_checkpoint(tmp_path / "candidate.pt", model, config)
    sidecar = json.loads(candidate.with_suffix(".json").read_text())
    sidecar["config"]["input_size"] = 384
    candidate.with_suffix(".json").write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="configuration"):
        unified.read_unified_checkpoint(candidate)


def test_scripted_export_keeps_delta_semantics_dynamic_batch_and_runtime_guards(assets, tmp_path):
    from ptrn_hip.unified_export import (script_model, load_export, _check_runtime_guards)
    weights, checkpoint, _ = assets
    model, _, head = unified.build_unified(weights, checkpoint)
    with torch.no_grad():
        head.bias.value[8] = .12
    saved = script_model(model, 224)
    path = tmp_path / "fixture.pt"
    torch.jit.save(saved, str(path), _extra_files={"contract.json": json.dumps(unified.contract(224))})
    restored, metadata = load_export(path, expected_sha256=sha256(path))
    assert metadata["output_fields"][-1] == "delta_log_ratio"
    assert "NOT complete" in metadata["ninth_value"]
    batch = np.full((2, 224, 224, 3), 100, dtype=np.uint8)
    expected = unified.predict(model, batch, "cpu")
    with torch.no_grad():
        actual = restored(torch.tensor(batch, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual[:, 8], .12, rtol=0, atol=1e-7)
    assert all(_check_runtime_guards(restored, 224).values())
    with pytest.raises(ValueError, match="input size"):
        load_export(path, expected_sha256=sha256(path), size=384)
    with pytest.raises(ValueError, match="SHA"):
        load_export(path, expected_sha256="wrong")


def test_export_rejects_same_shape_historical_metadata(assets, tmp_path):
    from ptrn_hip.unified_export import script_model, load_export
    weights, checkpoint, _ = assets
    model, _, _ = unified.build_unified(weights, checkpoint)
    saved = script_model(model, 224)
    metadata = unified.contract(224)
    metadata["ninth_value"] = "complete log(W/H)"
    path = tmp_path / "historical.pt"
    torch.jit.save(saved, str(path), _extra_files={"contract.json": json.dumps(metadata)})
    with pytest.raises(ValueError, match="contract mismatch"):
        load_export(path, expected_sha256=sha256(path))


def test_export_candidate_roundtrip_reports_sha_and_removes_duplicate_trace(assets, tmp_path):
    from ptrn_hip.unified_export import export_candidate, load_export
    weights, g0, _ = assets
    model, _, _ = unified.build_unified(weights, g0)
    config = {"input_size": 224, "contract": unified.contract(224),
              "source_sha": "fixture-revision", "source_tree_sha": "fixture-source-tree"}
    checkpoint = _write_checkpoint(tmp_path / "candidate.pt", model, config)
    output = tmp_path / ".local" / "export"
    report = export_candidate(weights, g0, checkpoint, output, "fixture-export-revision")
    artifact = output / report["artifact"]
    restored, metadata = load_export(artifact)
    assert sha256(artifact) == report["artifact_sha256"]
    assert metadata["training_source_tree_sha"] == "fixture-source-tree"
    assert metadata["output_fields"][-1] == "delta_log_ratio"
    assert report["comparison_count"] == 2
    assert report["batch_two_max_absolute_output_difference"] <= 1e-5
    assert report["missing_resource_fails"]
    assert not (output / "trace-raw.local.pt").exists()
    assert report["removed_debug_records"] > 0
