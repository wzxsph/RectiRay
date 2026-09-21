import cv2
import numpy as np
from PIL import Image

from ptrn_hip.geometry import as_matrix, corners, project
from ptrn_hip.synthesis import Synthesizer
from ptrn_hip.engine import batches


def synthesizer(tmp_path):
    Image.new("RGB", (256, 256), "#203040").save(tmp_path / "background.jpg")
    return Synthesizer(tmp_path)


def test_synthetic_landmarks_follow_label_and_screen_does_not_move_target(tmp_path):
    synth = synthesizer(tmp_path)
    source = np.full((224, 224, 3), 90, np.uint8)
    points = np.array([[50, 70], [170, 80], [100, 170]], np.float64)
    for color, point in zip([(255, 0, 0), (0, 255, 0), (0, 0, 255)], points):
        cv2.circle(source, tuple(point.astype(int)), 7, color, -1)
    matrix = np.array([[.65, .1, .05], [-.1, .7, -.05], [.04, -.03, 1]])
    expected = (project(points * 2 / 224 - 1, matrix) + 1) * 224 / 2
    for margins in [None, [.3, .2, .1, .4]]:
        result = synth.render(source, matrix, np.zeros_like(source), margins)
        for channel, point in enumerate(expected):
            x, y = np.rint(point).astype(int)
            pixel = result[y, x]
            assert pixel[channel] > 220
            assert sum(int(v) for i, v in enumerate(pixel) if i != channel) < 40


def test_random_labels_are_deterministic_visible_and_not_mirrored(tmp_path):
    synth = synthesizer(tmp_path)
    source = np.full((224, 224, 3), 150, np.uint8)
    a, label_a = synth.sample(source, 123)
    b, label_b = synth.sample(source, 123)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(label_a, label_b)
    for seed in range(100):
        _, label = synth.sample(source, seed)
        q = corners(as_matrix(label)).astype(np.float32)
        assert np.max(np.abs(q)) < .97
        assert cv2.isContourConvex(q)
        assert cv2.contourArea(q, oriented=True) > .22


def test_epoch_covers_every_source_before_next_epoch():
    class Capture:
        def batch(self, images, indices, seed):
            return indices
    stream = batches(None, np.arange(10), Capture(), 4, 123)
    selected = np.concatenate([next(stream) for _ in range(3)])
    assert set(selected) == set(range(10))
    assert len(selected) == 12
