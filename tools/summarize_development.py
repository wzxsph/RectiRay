"""Compare complete, checkpoint-bound development evaluations without exporting rows.

Example:
  python tools/summarize_development.py --run plus2250=.local/eval/plus2250 \
      --run plus2000=.local/eval/plus2000 --output .local/eval/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = (
    "additional_crop_fraction",
    "extra_background_fraction",
    "corner_error_diagonal",
    "absolute_ratio_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory or not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("Use a short label=directory for --run")
    return name, Path(directory)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_run(name: str, directory: Path, count: int) -> dict:
    if any((directory / part).exists() for part in ("inference-failed.json", "metrics-failed.json")):
        raise ValueError(f"{name}: failed evaluation")
    binding = read_json(directory / "binding.json")
    prediction = read_json(directory / "inference-complete.json")
    metric = read_json(directory / "metrics-complete.json")
    evaluation = read_json(directory / "evaluation.json")
    if any(item.get("count") != count for item in (binding, prediction, metric, evaluation)):
        raise ValueError(f"{name}: incomplete sample count")
    if not evaluation.get("full_development"):
        raise ValueError(f"{name}: result is not a full development evaluation")
    checkpoint = Path(binding["checkpoint"])
    if sha256(checkpoint) != binding["checkpoint_sha256"]:
        raise ValueError(f"{name}: checkpoint SHA-256 differs")
    if prediction.get("checkpoint_sha256", binding["checkpoint_sha256"]) != binding["checkpoint_sha256"]:
        raise ValueError(f"{name}: prediction/checkpoint binding differs")
    for key in ("checkpoint_sha256", "step", "development", "runtime_code", "batch", "prediction_backend",
                "cudnn_tf32", "matmul_tf32"):
        if evaluation["execution"].get(key) != binding.get(key):
            raise ValueError(f"{name}: evaluation/{key} binding differs")
    rows = evaluation["rows"]
    if len(rows) != count or any(row.get("index") != i for i, row in enumerate(rows)):
        raise ValueError(f"{name}: rows missing, duplicated or out of order")
    if sum(bool(row["valid_geometry"]) for row in rows) != evaluation["valid_geometry"]:
        raise ValueError(f"{name}: geometry count differs")
    if evaluation["valid_geometry"] + evaluation["invalid_geometry"] != count:
        raise ValueError(f"{name}: invalid geometry count differs")
    seen = set()
    for path in sorted(directory.glob("pred-*.json")):
        for row in read_json(path):
            index = row["index"]
            if index in seen:
                raise ValueError(f"{name}: duplicate prediction")
            seen.add(index)
    if seen != set(range(count)):
        raise ValueError(f"{name}: incomplete prediction shards")
    return {"name": name, "binding": binding, "rows": rows}


def describe(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid_geometry"]]
    result = {"count": len(rows), "valid_geometry": len(valid)}
    for metric in METRICS:
        measured = valid if metric in ("additional_crop_fraction", "extra_background_fraction") else rows
        values = [float(row[metric]) for row in measured]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Non-finite or negative {metric}")
        p90 = None
        if values:
            ordered = sorted(values)
            position = .9 * (len(ordered) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            p90 = (ordered[lower] * (upper - position) + ordered[upper] * (position - lower)) if upper != lower else ordered[lower]
        result[metric] = {
            "n": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p90": p90,
        }
    return result


def summarize(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("At least one run is required")
    first = runs[0]
    scene_keys = ("kind", "integrity", "background_family", "input_missing_fraction")
    scenes = [tuple(row[key] for key in scene_keys) for row in first["rows"]]
    development = first["binding"]["development"]
    protocol_keys = ("runtime_code", "batch", "prediction_backend", "cudnn_tf32", "matmul_tf32")
    protocol = {key: first["binding"].get(key) for key in protocol_keys}
    results = []
    for run in runs:
        if run["binding"]["development"] != development:
            raise ValueError("Development dataset bindings differ")
        if any(run["binding"].get(key) != value for key, value in protocol.items()):
            raise ValueError("Evaluation runtime protocols differ")
        if [tuple(row[key] for key in scene_keys) for row in run["rows"]] != scenes:
            raise ValueError("Ordered development scenes differ")
        rows = run["rows"]
        results.append({
            "label": run["name"],
            "step": run["binding"]["step"],
            "checkpoint_sha256": run["binding"]["checkpoint_sha256"],
            "prediction_backend": run["binding"]["prediction_backend"],
            "batch": run["binding"]["batch"],
            "all": describe(rows),
            "photo": describe([row for row in rows if row["kind"] == "photo"]),
            "photo_by_integrity": {
                label: describe([row for row in rows if row["kind"] == "photo" and row["integrity"] == label])
                for label in sorted({row["integrity"] for row in rows if row["kind"] == "photo"})
            },
        })
    code = protocol.pop("runtime_code")
    protocol["runtime_code_sha256"] = hashlib.sha256(json.dumps(code, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return {"scope": "Exposed full development set; diagnostic aggregate, not independent real-photo accuracy",
            "count": len(scenes), "evaluation_protocol": protocol, "runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="LABEL=DIR")
    parser.add_argument("--count", type=int, default=5736, help="Expected complete development size")
    parser.add_argument("--output", type=Path, required=True, help="Ignored local aggregate JSON path")
    args = parser.parse_args()
    if len({name for name, _ in args.run}) != len(args.run):
        parser.error("Run labels must be unique")
    if args.count != 5736:
        parser.error("The complete frozen development set contains 5736 views")
    if args.output.exists():
        parser.error("Output already exists")
    report = summarize([checked_run(name, directory, args.count) for name, directory in args.run])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Compared {len(report['runs'])} checkpoints across {report['count']} exposed development views")


if __name__ == "__main__":
    main()
