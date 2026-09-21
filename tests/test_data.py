import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from ptrn_hip.data import prepare


def test_source_images_unchanged_duplicate_copies_retained_and_grouped(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    data = np.random.default_rng(4).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    Image.fromarray(data).save(source / "a.png")
    (source / "b.png").write_bytes((source / "a.png").read_bytes())
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    output = tmp_path / "cache"
    summary = prepare({"fixture": source}, output)
    records = json.loads((output / "manifest.local.json").read_text())["records"]
    assert summary["readable"] == 2
    assert summary["exact_duplicate_copies_retained"] == 1
    assert records[0]["split"] == records[1]["split"]
    assert all("group" not in key and "patient" not in key for row in records for key in row)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert before == after
    with pytest.raises(FileExistsError):
        prepare({"fixture": source}, output)
    with pytest.raises(ValueError):
        prepare({"fixture": source}, source / "nested-cache")
