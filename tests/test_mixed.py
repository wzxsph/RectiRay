import numpy as np
from PIL import Image

from ptrn_hip.mixed_synthesis import MixedSynthesizer
from ptrn_hip.geometry import as_matrix, corners
from ptrn_hip.experiment import pixel_metrics, selection_gates, decision_gate


def make_synth(tmp_path):
    Image.new("RGB", (256, 256), (50, 80, 100)).save(tmp_path/"background.jpg")
    return MixedSynthesizer(tmp_path)


def test_identity_is_exact_and_includes_edges(tmp_path):
    source = np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    image, label, info = make_synth(tmp_path).sample_record(source, 17, mode="identity")
    np.testing.assert_array_equal(image, source)
    np.testing.assert_array_equal(label, [1, 0, 0, 0, 1, 0, 0, 0])
    assert abs(corners(as_matrix(label))).max() == 1
    assert info["area_fraction"] == 1
    image[:] = 0
    assert source.sum() > 0


def test_mild_visible_and_window_never_changes_inner_source(tmp_path):
    synth = make_synth(tmp_path)
    source = np.full((224, 224, 3), 180, np.uint8)
    for seed in range(100):
        image, label, info = synth.sample_record(source, seed, augment=False, mode="mild")
        q = corners(as_matrix(label))
        assert abs(q).max() <= .995
        assert -12 <= info["angle_degrees"] <= 12
        center = ((as_matrix(label)[:2, 2]+1)*112).astype(int)
        np.testing.assert_array_equal(image[center[1], center[0]], [180]*3)
    matrix = np.eye(3)
    matrix[:2, :2] *= .75
    bg = synth.window_background(np.zeros_like(source), matrix, np.random.default_rng(0))
    with_window = synth.render(source, matrix, bg)
    without_window = synth.render(source, matrix, np.zeros_like(source))
    np.testing.assert_array_equal(with_window[30:194, 30:194], without_window[30:194, 30:194])


def test_modes_and_seeds_are_reproducible(tmp_path):
    synth = make_synth(tmp_path)
    source = np.full((224, 224, 3), 100, np.uint8)
    modes = []
    for seed in range(100):
        a, y, info = synth.sample_record(source, seed)
        b, z, again = synth.sample_record(source, seed)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(y, z)
        assert info == again
        modes.append(info["mode"])
    assert set(modes) == {"identity", "mild", "hard"}


def test_metric_keeps_invalid_cases_and_detects_orientation():
    identity = np.eye(3).reshape(-1)[:8]
    turned = np.diag([-1., -1., 1.]).reshape(-1)[:8]
    truth = np.stack([identity]*3)
    prediction = np.stack([identity, turned, np.full(8, np.nan)])
    scores, rows = pixel_metrics(truth, prediction, [(640, 480)]*3)
    assert scores["count"] == 3 and scores["invalid"] == 1
    assert rows[0]["corner_nme"] == 0
    assert rows[1]["corner_nme"] > .99
    assert rows[2]["corner_nme"] == 1


def test_identity_alone_cannot_qualify_a_checkpoint():
    def group(mean, p90=.01, invalid=0):
        return {"corner_nme_mean": mean, "corner_nme_p90": p90, "invalid": invalid}
    baseline = {"identity": group(.16), "mild": group(.03), "hard": group(.025)}
    collapsed = {"identity": group(0), "mild": group(.1), "hard": group(.3)}
    assert not all(selection_gates(collapsed, baseline).values())
    assert not decision_gate(collapsed, baseline)
    good = {"identity": group(.005), "mild": group(.02), "hard": group(.026)}
    assert all(selection_gates(good, baseline).values())
    assert decision_gate(good, baseline)
