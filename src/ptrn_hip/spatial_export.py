"""CPU-only TorchScript export of a spatial *corners-only* checkpoint.

The artifact is deliberately not a complete geometry candidate: a separately
validated ratio module or an explicitly labelled edge baseline is still needed.
Exports and reports must stay under a local ``.local`` directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import torch

ARMS = frozenset({'gap_h8', 'gap_quad', 'spatial_quad', 'spatial_heatmap'})
SIZES = frozenset({224, 384})
CORNER_CONTRACT = 'ptrn-spatial-corners-v2'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class _ContractGuard(torch.nn.Module):
    __constants__ = ['size']

    def __init__(self, traced, size: int):
        super().__init__()
        self.traced = traced
        self.size = size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 4 or pixels.size(0) < 1 or pixels.size(1) != self.size or pixels.size(2) != self.size or pixels.size(3) != 3:
            raise RuntimeError('Expected nonempty NHWC RGB at the selected size')
        if pixels.dtype != torch.float32:
            raise RuntimeError('Expected float32 raw pixels')
        if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 255).any()):
            raise RuntimeError('Expected finite RGB 0..255')
        result = self.traced(pixels)
        if result.dim() != 3 or result.size(0) != pixels.size(0) or result.size(1) != 4 or result.size(2) != 2:
            raise RuntimeError('Expected TL/TR/BR/BL [N,4,2]')
        if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
            raise RuntimeError('Expected finite float32 corners')
        return result


def script_corners(model, size):
    """Trace a raw-pixel entrypoint and script its persistent runtime checks."""
    if size not in SIZES:
        raise ValueError('unsupported_input_size')
    wrapped = model.raw_pixels().to('cpu').eval()
    example = torch.zeros((1, size, size, 3), dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.jit.TracerWarning)
        traced = torch.jit.trace(wrapped, example, check_trace=False, strict=False)
    return torch.jit.script(_ContractGuard(traced, size).eval())


def _metadata(size, arm, checkpoint):
    from .spatial_model import contract
    return {**contract(size, arm), 'artifact_role': 'corners_only_localization_submodel',
            'ratio_method': 'none', 'ratio_output': None, 'action_output': None,
            'complete_geometry_candidate': False, 'checkpoint_sha256': sha256(checkpoint),
            'cpu_only': True}


def load_export(path, *, expected_sha256, size=None, arm=None):
    """Hash and embedded contract must match before inference is exposed."""
    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError('artifact_SHA_mismatch')
    extra = {'contract.json': ''}
    model = torch.jit.load(str(path), map_location='cpu', _extra_files=extra).eval()
    try:
        metadata = json.loads(extra['contract.json'])
    except (TypeError, ValueError) as error:
        raise ValueError('missing_or_invalid_embedded_contract') from error
    actual_size = metadata.get('input_shape', [None, None])[1]
    if (actual_size not in SIZES or metadata.get('arm') not in ARMS
            or metadata.get('contract') != CORNER_CONTRACT
            or metadata.get('input_shape') != [None, actual_size, actual_size, 3]
            or metadata.get('output_shape') != [None, 4, 2]
            or metadata.get('output_fields') != 'TL,TR,BR,BL input pixel-centre x,y'
            or metadata.get('input_dtype') != 'float32'
            or metadata.get('output_dtype') != 'float32'):
        raise ValueError('spatial_contract_mismatch')
    if (metadata.get('artifact_role') != 'corners_only_localization_submodel'
            or metadata.get('complete_geometry_candidate') is not False
            or metadata.get('ratio_method') != 'none'
            or metadata.get('ratio_output') is not None
            or metadata.get('action_output') is not None):
        raise ValueError('incomplete_or_mislabelled_export')
    if (size is not None and metadata['input_shape'][1] != size) or (arm is not None and metadata['arm'] != arm):
        raise ValueError('requested_contract_mismatch')
    return model, metadata


def export_checkpoint_cpu(weights, g0_checkpoint, checkpoint, output):
    """Execute export in a child with CUDA devices hidden, returning JSON report."""
    output = Path(output).resolve()
    if '.local' not in output.parts:
        raise ValueError('exports_must_be_under_local_directory')
    command = [sys.executable, '-m', 'ptrn_hip.spatial_export', '_worker',
               '--weights', str(weights), '--g0-checkpoint', str(g0_checkpoint),
               '--checkpoint', str(checkpoint), '--output', str(output)]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run(command, env=env, check=True)
    return json.loads((output / 'export-report.json').read_text())


def _worker(weights, g0_checkpoint, checkpoint, output):
    from .spatial_model import build_spatial
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU_export_requires_hidden_CUDA_devices')
    output = Path(output).resolve()
    if '.local' not in output.parts or output.exists():
        raise ValueError('output_must_be_new_local_directory')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    arm, size, stage = config['arm'], int(config['size']), config['stage']
    if arm not in ARMS or size not in SIZES:
        raise ValueError('unexpected_checkpoint_contract')
    torch.set_num_threads(4)
    model = build_spatial(weights, g0_checkpoint, arm, size=size, stage=stage)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.to('cpu').eval()
    scripted = script_corners(model, size)
    metadata = _metadata(size, arm, checkpoint)
    metadata.update(stage=stage, b11_weights_sha256=sha256(weights),
                    g0_checkpoint_sha256=sha256(g0_checkpoint))
    output.mkdir(parents=True)
    artifact = output / 'spatial-corners-only.local.torchscript.pt'
    torch.jit.save(scripted, str(artifact), _extra_files={'contract.json': json.dumps(metadata)})
    saved, restored = load_export(artifact, expected_sha256=sha256(artifact), size=size, arm=arm)
    if restored != metadata:
        raise AssertionError('export_metadata_changed_on_reload')
    rng = np.random.default_rng(20260923)
    differences = []
    with torch.no_grad():
        for batch in (1, 2):
            pixels = torch.from_numpy(rng.integers(0, 256, (batch, size, size, 3), dtype=np.uint8).astype(np.float32))
            expected = model.raw_pixels().eval()(pixels).detach().numpy()
            actual = saved(pixels).detach().numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
            differences.append(float(np.max(np.abs(actual - expected))))
    for invalid in (torch.zeros((size, size, 3)), torch.zeros((1, size - 1, size, 3)),
                    torch.zeros((1, size, size, 3), dtype=torch.float64),
                    torch.full((1, size, size, 3), float('nan'))):
        try:
            saved(invalid)
        except (RuntimeError, torch.jit.Error):
            pass
        else:
            raise AssertionError('export_accepted_invalid_input')
    report = {'artifact': artifact.name, 'artifact_sha256': sha256(artifact),
              'checkpoint_sha256': metadata['checkpoint_sha256'], 'contract': metadata,
              'reload_random_max_abs_difference': differences,
              'scope': 'CPU_corners_only_no_ratio_or_action_no_real_photo_validation'}
    (output / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def predict_local_photos(artifact, artifact_sha256, photos, size, *, ordinal_start=0):
    """Read local RGB photos and return ordinal/corners only; never save pixels."""
    from .data import read_rgb
    from .ratio_synthesis import letterbox
    saved, metadata = load_export(artifact, expected_sha256=artifact_sha256, size=size)
    rows = []
    with torch.no_grad():
        for ordinal, path in enumerate(photos, start=ordinal_start):
            pixels = letterbox(read_rgb(path), size)[0]
            prediction = saved(torch.tensor(pixels[None], dtype=torch.float32))[0].numpy()
            rows.append({'ordinal': ordinal, 'quad_input': prediction.tolist(),
                         'ratio_method': 'none_corners_only'})
    return {'contract': metadata, 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('export', '_worker'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--g0-checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = (_worker(args.weights, args.g0_checkpoint, args.checkpoint, args.output)
              if args.command == '_worker' else
              export_checkpoint_cpu(args.weights, args.g0_checkpoint, args.checkpoint, args.output))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
