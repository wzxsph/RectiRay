"""CPU-only geometry objective tests with synthetic pixel-centre quadrilaterals."""
import math

import pytest

from ptrn_hip.model import torch
from ptrn_hip import spatial_objectives as objectives


def _quad(left=16., top=16., right=48., bottom=48.):
    return torch.tensor([[[left, top], [right, top], [right, bottom], [left, bottom]]],
                        dtype=torch.float32)


def _coverage(pred, truth, size=64, *, inverse=None, photo_size=None):
    if inverse is None:
        inverse = torch.eye(3).unsqueeze(0)
    if photo_size is None:
        photo_size = (size, size)
    return objectives.coverage_loss(pred, truth, (size, size),
                                    inverse_letterbox=inverse,
                                    photo_sizes=torch.tensor([photo_size]))


def test_identical_cover_and_grid_have_zero_loss():
    truth = _quad()
    cover = _coverage(truth, truth)
    assert cover.total.item() == pytest.approx(0., abs=1e-8)
    assert cover.missed.item() == cover.background.item() == cover.invalid.item() == 0.
    assert cover.photo_boundary.item() == 0.
    grid = objectives.grid_geometry_loss(truth, truth, torch.ones(1), torch.ones(1), (64, 64))
    assert grid.total.item() == pytest.approx(0., abs=1e-6)
    assert grid.angle.item() == pytest.approx(0., abs=1e-6)


def test_inward_missing_film_costs_more_but_whole_photo_pays_background():
    truth = _quad()
    inward = _coverage(_quad(20, 20, 44, 44), truth)
    outward = _coverage(_quad(12, 12, 52, 52), truth)
    whole_photo = _coverage(_quad(0, 0, 63, 63), truth)
    assert inward.missed > 0 and inward.background < inward.missed
    assert outward.background > 0 and outward.missed < outward.background
    assert inward.total > outward.total
    assert whole_photo.background > outward.background
    assert whole_photo.total > 0


def test_soft_cover_has_corner_gradient_and_finite_illegal_quad_penalty():
    truth = _quad()
    pred = _quad(19, 18, 45, 44).requires_grad_()
    terms = _coverage(pred, truth)
    terms.total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
    collapsed = torch.full((1, 4, 2), 32., requires_grad=True)
    invalid = _coverage(collapsed, truth)
    assert torch.isfinite(invalid.total) and invalid.invalid > 1
    invalid.total.backward()
    assert torch.isfinite(collapsed.grad).all()
    folded = truth[:, [0, 2, 1, 3]].clone().requires_grad_()
    folded_loss = _coverage(folded, truth)
    assert torch.isfinite(folded_loss.total) and folded_loss.invalid > 0


def test_photo_boundary_penalizes_letterbox_padding_and_has_gradient():
    # Original 32x64 photograph occupies x=16..47 of the 64x64 input canvas.
    inverse = torch.tensor([[[1., 0., -16.], [0., 1., 0.], [0., 0., 1.]]])
    truth = _quad(16, 0, 47, 63)
    legal = _coverage(truth, truth, inverse=inverse, photo_size=(32, 64))
    assert legal.photo_boundary.item() == pytest.approx(0., abs=1e-8)
    assert legal.total.item() == pytest.approx(0., abs=1e-8)
    # Still on the 64 canvas, but 2 px beyond the actual photographed extent.
    padding = _quad(14, 0, 49, 63).requires_grad_()
    outside = _coverage(padding, truth, inverse=inverse, photo_size=(32, 64))
    assert outside.photo_boundary.item() > 0
    outside.photo_boundary.backward()
    assert padding.grad is not None and torch.isfinite(padding.grad).all()
    assert padding.grad.abs().sum().item() > 0


def test_photo_boundary_penalizes_outside_letterbox_canvas():
    inverse = torch.eye(3).unsqueeze(0)
    truth = _quad(0, 0, 63, 63)
    prediction = _quad(-3, -2, 66, 65).requires_grad_()
    terms = _coverage(prediction, truth, inverse=inverse, photo_size=(64, 64))
    assert torch.isfinite(terms.total)
    assert terms.photo_boundary.item() > 0
    terms.photo_boundary.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_full_photo_boundary_counteracts_coverage_outward_preference():
    truth = _quad(0, 0, 63, 63)
    inward = _coverage(_quad(1, 0, 62, 63), truth)
    outward = _coverage(_quad(-1, 0, 64, 63), truth)
    assert inward.missed > 0 and inward.photo_boundary == 0
    assert outward.background > 0 and outward.photo_boundary > 0
    assert outward.total > inward.total


def test_photo_boundary_rejects_missing_or_malformed_acquisition_geometry():
    truth = _quad()
    with pytest.raises(TypeError):
        objectives.coverage_loss(truth, truth, (64, 64))
    with pytest.raises(ValueError, match="inverse"):
        objectives.coverage_loss(truth, truth, (64, 64),
                                 inverse_letterbox=torch.eye(3),
                                 photo_sizes=torch.tensor([[64, 64]]))


def test_grid_detects_ratio_stretch_and_projective_error_with_gradients():
    truth = _quad()
    ratio = torch.tensor([1.])
    wrong_ratio = torch.tensor([1.4], requires_grad=True)
    stretched = objectives.grid_geometry_loss(truth, truth, wrong_ratio, ratio, (64, 64))
    assert stretched.grid.item() > .001
    stretched.total.backward()
    assert wrong_ratio.grad is not None and wrong_ratio.grad.abs().item() > 0
    pred = truth.clone().detach()
    pred[0, 1, 0] -= 4
    pred.requires_grad_()
    skewed = objectives.grid_geometry_loss(pred, truth, ratio, ratio, (64, 64))
    assert skewed.grid.item() > 0
    skewed.total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_grid_rejects_singular_homography_instead_of_dropping_sample():
    truth = _quad()
    with pytest.raises(FloatingPointError, match="Singular"):
        objectives.grid_geometry_loss(torch.full_like(truth, 32.), truth,
                                      torch.ones(1), torch.ones(1), (64, 64))


def test_ratio_residual_starts_at_zero_and_cannot_update_locator():
    module = objectives.RatioResidual(feature_dim=12, hidden=8, size=64)
    corners = _quad().requires_grad_()
    features = torch.ones((1, 12), requires_grad=True)
    delta = module(corners, features)
    assert delta.shape == (1,) and delta.item() == 0
    expected = math.log(1.)
    assert module.decode_log_ratio(corners, features).item() == pytest.approx(expected)
    loss = objectives.ratio_residual_loss(module.decode_log_ratio(corners, features),
                                          torch.tensor([math.log(1.25)]))
    loss.backward()
    assert corners.grad is None and features.grad is None
    assert module.network[-1].bias.grad is not None
    assert module.network[-1].bias.grad.abs().item() > 0
