import json

import pytest
import torch

from ptrn_hip.spatial_export import (export_checkpoint_cpu, load_export,
                                     script_corners, sha256)


class TinyRaw(torch.nn.Module):
    def forward(self, pixels):
        scalar = pixels.mean(dim=(1, 2, 3))[:, None, None]
        points = torch.tensor([[10, 10], [210, 10], [210, 210], [10, 210]], dtype=torch.float32)
        return points[None] + scalar * .001


class TinyModel:
    def raw_pixels(self):
        return TinyRaw()


def test_corners_only_trace_reload_and_runtime_contract(tmp_path):
    scripted = script_corners(TinyModel(), 224)
    path = tmp_path / 'corners.pt'
    metadata = {'contract': 'ptrn-spatial-corners-v2', 'arm': 'gap_quad',
                'input_shape': [None, 224, 224, 3], 'output_shape': [None, 4, 2],
                'input_dtype': 'float32', 'output_dtype': 'float32',
                'output_fields': 'TL,TR,BR,BL input pixel-centre x,y',
                'artifact_role': 'corners_only_localization_submodel',
                'complete_geometry_candidate': False, 'ratio_method': 'none'}
    torch.jit.save(scripted, str(path), _extra_files={'contract.json': json.dumps(metadata)})
    saved, result = load_export(path, expected_sha256=sha256(path), size=224, arm='gap_quad')
    assert result['complete_geometry_candidate'] is False
    pixels = torch.full((2, 224, 224, 3), 127., dtype=torch.float32)
    torch.testing.assert_close(saved(pixels), TinyRaw()(pixels))
    with pytest.raises((RuntimeError, torch.jit.Error)):
        saved(torch.full((1, 224, 224, 3), float('nan')))
    with pytest.raises((RuntimeError, torch.jit.Error)):
        saved(torch.zeros((1, 224, 224, 3), dtype=torch.float64))
    with pytest.raises(ValueError, match='SHA'):
        load_export(path, expected_sha256='0' * 64)


def test_export_launcher_hides_cuda_and_requires_local_output(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='local'):
        export_checkpoint_cpu('w', 'g', 'c', tmp_path / 'out')
    captured = {}

    def fake_run(command, env, check):
        captured.update(command=command, cuda=env['CUDA_VISIBLE_DEVICES'], check=check)
        output = tmp_path / '.local' / 'out'
        output.mkdir(parents=True)
        (output / 'export-report.json').write_text('{}')

    monkeypatch.setattr('ptrn_hip.spatial_export.subprocess.run', fake_run)
    assert export_checkpoint_cpu('w', 'g', 'c', tmp_path / '.local' / 'out') == {}
    assert captured['cuda'] == '' and captured['check'] is True
