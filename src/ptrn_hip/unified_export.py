"""CPU TorchScript export for the explicit H8 + delta unified contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import cv2
import numpy as np

from .data import read_rgb, write_json
from .model import keras, torch
from .ratio_data import sha256
from .ratio_export import clean_archive
from .ratio_synthesis import letterbox
from .unified_model import (B11_SHA, G0_SHA, contract, load_unified, predict,
                            read_unified_checkpoint)


class _PixelInput(torch.nn.Module):
    """Trace the Keras Torch forward; runtime checks are scripted separately."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor([.485, .456, .406]))
        self.register_buffer("std", torch.tensor([.229, .224, .225]))

    def forward(self, pixels):
        return self.base((pixels / 255.0 - self.mean) / self.std, training=False).float()


class _ContractGuard(torch.nn.Module):
    """Keep shape/dtype/finiteness validation alive after tracing the base."""
    __constants__ = ["input_size"]

    def __init__(self, traced, input_size: int):
        super().__init__()
        self.traced = traced
        self.input_size = input_size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 4:
            raise RuntimeError("Expected NHWC [N,size,size,3]")
        if pixels.size(0) < 1 or pixels.size(1) != self.input_size or pixels.size(2) != self.input_size or pixels.size(3) != 3:
            raise RuntimeError("Input shape violates unified contract")
        if pixels.dtype != torch.float32:
            raise RuntimeError("Unified export requires float32 RGB pixels")
        if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 255).any()):
            raise RuntimeError("Expected finite RGB values in 0..255")
        output = self.traced(pixels)
        if output.dim() != 2 or output.size(0) != pixels.size(0) or output.size(1) != 9:
            raise RuntimeError("Output must be [N,9] H8 plus delta_log_ratio")
        if output.dtype != torch.float32 or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Unified output must be finite float32")
        return output


def script_model(model, size):
    """CPU artifact, with scripted runtime guards around the traced encoder."""
    contract(size)  # Validate supported resolution before doing any tracing.
    model.to("cpu")
    wrapped = _PixelInput(model).eval()
    example = torch.zeros((1, size, size, 3), dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(wrapped, example, check_trace=True, strict=True)
    return torch.jit.script(_ContractGuard(traced, size).eval())


def _validate_metadata(metadata, size=None):
    actual_size = metadata.get("input_size")
    expected = contract(actual_size)
    if size is not None and actual_size != size:
        raise ValueError("Export input size mismatch")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Export contract mismatch; historical log-ratio outputs are incompatible")
    return actual_size


def load_export(path, *, expected_sha256=None, size=None):
    """Check file SHA and embedded delta contract before returning a CPU model."""
    path = Path(path)
    if expected_sha256 is None:
        report = json.loads((path.parent / "export-report.json").read_text())
        expected_sha256 = report["artifact_sha256"]
    if sha256(path) != expected_sha256:
        raise ValueError("TorchScript artifact SHA mismatch")
    extra = {"contract.json": ""}
    model = torch.jit.load(str(path), map_location="cpu", _extra_files=extra).eval()
    try:
        metadata = json.loads(extra["contract.json"])
    except (ValueError, TypeError) as error:
        raise ValueError("Missing/invalid embedded unified contract") from error
    _validate_metadata(metadata, size)
    return model, metadata


def _check_runtime_guards(saved, size):
    failures = {}
    cases = {
        "rank": torch.zeros((size, size, 3)),
        "empty_batch": torch.zeros((0, size, size, 3)),
        "height": torch.zeros((1, size - 1, size, 3)),
        "width": torch.zeros((1, size, size - 1, 3)),
        "channels": torch.zeros((1, size, size, 1)),
        "dtype": torch.zeros((1, size, size, 3), dtype=torch.float64),
        "nonfinite": torch.full((1, size, size, 3), float("nan")),
        "range": torch.full((1, size, size, 3), 256.0),
    }
    with torch.no_grad():
        for name, value in cases.items():
            try:
                saved(value)
            except (RuntimeError, torch.jit.Error):
                failures[name] = True
            else:
                raise AssertionError(f"Export accepted invalid {name}")
    return failures


def export_candidate(weights, g0_checkpoint, checkpoint, output, export_source_sha, real_dir=None):
    """Write a local export and CPU differential report; return its report.

    real_dir, if provided, is an explicitly authorized read-only local photo
    directory. Reports contain only ordinals and aggregate evidence, no paths.
    """
    output = Path(output)
    if ".local" not in output.parts:
        raise ValueError("Research exports must remain in a .local output directory")
    if not export_source_sha or any(part in str(export_source_sha) for part in ("/", "\\")):
        raise ValueError("An export source revision identifier is required")
    payload = read_unified_checkpoint(checkpoint)
    config = payload["config"]
    del payload
    size = config["input_size"]
    with keras.device("cpu"):
        model, _, _ = load_unified(weights, g0_checkpoint, checkpoint)
    model.to("cpu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        scripted = script_model(model, size)
        metadata = {
            **contract(size),
            "training_source_sha": config.get("source_sha"),
            "training_source_tree_sha": config.get("source_tree_sha"),
            "checkpoint_sha256": sha256(checkpoint),
            "g0_initialization_sha256": G0_SHA,
            "b11_architecture_weights_sha256": B11_SHA,
            "export_source_sha": str(export_source_sha),
            "weights_redistribution_approved": False,
            "windows_verified": False,
            "versions": {"torch": torch.__version__, "keras": keras.__version__, "opencv": cv2.__version__},
        }
        output.mkdir(parents=True, exist_ok=False)
        raw = output / "trace-raw.local.pt"
        target = output / "ptrn-unified-geometry-v1.torchscript.pt"
        torch.jit.save(scripted, str(raw), _extra_files={"contract.json": json.dumps(metadata)})
        removed = clean_archive(raw, target)
        saved, restored_metadata = load_export(target, expected_sha256=sha256(target), size=size)
        if restored_metadata != metadata:
            raise ValueError("Export metadata changed during archive cleanup")
        if set(scripted.state_dict()) != set(saved.state_dict()) or not all(
                torch.equal(value, saved.state_dict()[key]) for key, value in scripted.state_dict().items()):
            raise ValueError("Export tensor state changed during archive cleanup")

        rng = np.random.default_rng(20260922)
        arrays = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                  np.zeros((size, size, 3), dtype=np.uint8)]
        originals = []
        if real_dir is not None:
            paths = sorted(p for p in Path(real_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not paths:
                raise ValueError("Explicit real-photo directory contains no supported images")
            for path in paths:
                before = sha256(path)
                try:
                    pixels = read_rgb(path)
                except Exception as error:
                    raise RuntimeError(type(error).__name__) from None
                arrays.append(letterbox(pixels, size)[0])
                originals.append((path, before))
        records = []
        with torch.no_grad():
            for index, pixels in enumerate(arrays):
                expected = predict(model, pixels[None], "cpu")
                actual = saved(torch.tensor(pixels[None], dtype=torch.float32)).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                records.append({"ordinal": index, "max_abs_error": float(np.max(np.abs(actual - expected)))})
            batch = np.stack(arrays[:2])
            expected = predict(model, batch, "cpu")
            actual = saved(torch.tensor(batch, dtype=torch.float32)).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            batch_error = float(np.max(np.abs(actual - expected)))
        guards = _check_runtime_guards(saved, size)
        if any(sha256(path) != before for path, before in originals):
            raise ValueError("Read-only source photo changed")
        timings = {}
        probe = torch.tensor(arrays[0][None], dtype=torch.float32)
        for threads in (2, 4):
            torch.set_num_threads(threads)
            samples = []
            with torch.no_grad():
                for index in range(13):
                    started = time.perf_counter()
                    saved(probe)
                    if index >= 3:
                        samples.append((time.perf_counter() - started) * 1000)
            timings[str(threads)] = {"warm_p50_ms": float(np.median(samples)),
                                     "warm_p95_ms": float(np.percentile(samples, 95)),
                                     "measured": 10, "warmup": 3}
        try:
            torch.jit.load(str(output / "intentionally-missing.pt"), map_location="cpu")
        except (ValueError, RuntimeError, OSError):
            missing_failed = True
        else:
            raise AssertionError("Missing model resource unexpectedly loaded")
        report = {
            "metadata": metadata,
            "artifact": target.name,
            "artifact_sha256": sha256(target),
            "removed_debug_records": removed,
            "state_equal_after_sanitizing": True,
            "comparison_count": len(records),
            "max_absolute_output_difference": max(row["max_abs_error"] for row in records),
            "batch_two_max_absolute_output_difference": batch_error,
            "runtime_rejection_checks": guards,
            "records": records,
            "cpu_forward": timings,
            "missing_resource_fails": missing_failed,
            "real_photo_count": len(originals),
            "real_photo_scope": "unpaired_development_visual_only",
            "real_ratio_accuracy_verified": False,
        }
        write_json(output / "export-report.json", report)
        # The sanitized artifact is retained; do not keep a duplicate 70 MiB trace.
        raw.unlink()
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--g0-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-source-sha", required=True)
    parser.add_argument("--real-dir", type=Path)
    args = parser.parse_args()
    report = export_candidate(args.weights, args.g0_checkpoint, args.checkpoint,
                              args.output, args.export_source_sha, args.real_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
