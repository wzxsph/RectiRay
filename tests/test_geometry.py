import numpy as np
import pytest
import cv2

from ptrn_hip.geometry import UNIT_CORNERS, as_matrix, corners, project, pixel_matrix, rectify, geometry_metrics


def test_normalized_and_pixel_mappings_agree_and_invert():
    h = np.array([[.6, -.15, .1], [.1, .7, -.2], [.05, -.03, 1.]])
    p = (UNIT_CORNERS + 1) * [640, 480] / 2
    actual = project(p, pixel_matrix(h, (640, 480), (1200, 900)))
    np.testing.assert_allclose(actual, corners(h, (1200, 900)), atol=1e-8)
    np.testing.assert_allclose(project(project(UNIT_CORNERS, h), np.linalg.inv(h)), UNIT_CORNERS, atol=1e-9)


def test_rectification_preserves_left_right_and_corner_order():
    source = np.zeros((120, 160, 3), np.uint8)
    source[:, :80, 0] = 255
    source[:, 80:, 2] = 255
    output, h = rectify(source, np.eye(3))
    assert output[60, 25, 0] == 255 and output[60, 135, 2] == 255
    assert np.linalg.det(h) > 0


def test_bad_projections_are_not_reported_as_valid():
    source = np.zeros((200, 200, 3), np.uint8)
    with pytest.raises(ValueError):
        rectify(source, np.diag([-1., 1., 1.]))
    bad = np.array([[1., 0, 0], [0, 1, 0], [1, 0, 1]])
    with pytest.raises(ValueError):
        corners(bad)


def test_corner_metric_detects_180_rotation_even_with_same_region():
    identity = np.eye(3).reshape(-1)[:8][None]
    turned = np.diag([-1., -1., 1.]).reshape(-1)[:8][None]
    metrics = geometry_metrics(identity, turned)
    assert metrics['quad_iou_mean'] == pytest.approx(1)
    assert metrics['corner_nme_mean'] > .9
