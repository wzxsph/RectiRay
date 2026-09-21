from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Hip PTRN local training and visual evaluation")
    parser.add_argument("--config", type=Path, default=Path("configs/hip.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", action="append", required=True, help="name=/path/to/images; repeat for each pool")
    commands.add_parser("preview")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--steps", type=int, default=100)
    benchmark.add_argument("--batch-size", type=int)
    commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--inputs", type=Path, required=True)
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path(".local/eval47"))
    restore = commands.add_parser("restore")
    restore.add_argument("--input", type=Path, required=True)
    restore.add_argument("--output", type=Path)
    restore.add_argument("--source-kind", choices=["digital", "phone", "unknown"], default="unknown")
    restore.add_argument("--checkpoint", type=Path)
    commands.add_parser("audit-digital")
    commands.add_parser("report-digital")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == "prepare":
        from .data import prepare
        prepare({key: Path(value) for key, value in (entry.split("=", 1) for entry in args.source)}, Path(config["data"]))
    elif args.command == "preview":
        from .reports import preview
        preview(config)
    elif args.command == "benchmark":
        from .engine import benchmark
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        benchmark(config, args.steps)
    elif args.command == "train":
        from .engine import train
        train(config)
    elif args.command == "evaluate":
        from .engine import evaluate_all
        evaluate_all(config, args.checkpoint)
    elif args.command == "compare":
        from .reports import compare
        compare(config, args.inputs, args.checkpoint, args.output)
    elif args.command == "audit-digital":
        from .v2_reports import audit_digital
        audit_digital(config)
    elif args.command == "report-digital":
        from .v2_reports import digital_comparison
        digital_comparison(config)
    elif args.command == "restore":
        import json
        from .pipeline import process_image
        predictor = None
        if args.source_kind == "phone":
            if args.checkpoint is None:
                parser.error("phone requires --checkpoint")
            from .model import build, configure, infer
            configure(mixed=False)
            model, _ = build(args.checkpoint)
            predictor = lambda image: infer(model, image[None], 1)[0]
        print(json.dumps(process_image(args.input, args.output, args.source_kind, predictor)))


if __name__ == "__main__":
    main()
