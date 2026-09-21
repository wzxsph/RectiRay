import hashlib
import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from ptrn_hip.pipeline import process_image


@pytest.mark.parametrize("kind", ["digital", "unknown"])
def test_passthrough_keeps_high_bit_depth_and_metadata_without_decoding(tmp_path, monkeypatch, kind):
    path = tmp_path/"source.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("test_tag", "preserve me")
    pixels = np.arange(64*64, dtype=np.uint16).reshape(64, 64)*13
    Image.fromarray(pixels).save(path, pnginfo=metadata)
    original = path.read_bytes()
    def forbidden(*args):
        raise AssertionError("Digital/unknown input must never be decoded or predicted")
    monkeypatch.setattr("ptrn_hip.pipeline.read_rgb", forbidden)
    output = tmp_path/"copy.png"
    result = process_image(path, output, kind, forbidden)
    assert not result["model_used"] and not result["decoded"]
    assert output.read_bytes() == original == path.read_bytes()
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    with Image.open(output) as im:
        np.testing.assert_array_equal(np.asarray(im), pixels)
        assert im.info["test_tag"] == "preserve me"


def test_refuses_overwriting_original_or_existing_output(tmp_path):
    path = tmp_path/"source.png"
    Image.new("RGB", (80, 80)).save(path)
    with pytest.raises(FileExistsError):
        process_image(path, path, "digital")
    with pytest.raises(FileExistsError):
        process_image(path, path, "phone", lambda _: np.eye(3).reshape(-1)[:8])


def test_phone_uses_original_pixels_and_unknown_defaults_to_preserve(tmp_path):
    path = tmp_path/"source.png"
    Image.new("RGB", (160, 120), "red").save(path)
    original = path.read_bytes()
    output = tmp_path/"candidate.png"
    def predict(image):
        assert image.shape == (224, 224, 3)
        return np.eye(3).reshape(-1)[:8]
    result = process_image(path, output, "phone", predict)
    assert result["model_used"]
    assert path.read_bytes() == original
    assert Image.open(output).size == (161, 121)
    assert process_image(path)["action"] == "unknown_preserved"
