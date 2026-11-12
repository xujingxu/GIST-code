from __future__ import annotations

import argparse
import json
from pathlib import Path

from gist_fusion.config import load_config
from gist_fusion.evaluation import reproduce_paper_metrics
from gist_fusion.training import retrain_final_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gist-fusion",
        description="Reproduce gGIST multiscale-fusion analyses from deidentified feature tables.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML experiment configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser("reproduce", help="Evaluate frozen paper predictions")
    reproduce.add_argument("--output", default="runs/paper_reproduction")
    reproduce.add_argument("--bootstrap", type=int, default=None)

    train = subparsers.add_parser("train", help="Retrain final models from frozen feature inputs")
    train.add_argument("--output", default="runs/retrained_models")
    train.add_argument("--bootstrap", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "reproduce":
        summary = reproduce_paper_metrics(config, args.output, replicates=args.bootstrap)
    else:
        summary = retrain_final_models(config, args.output, replicates=args.bootstrap)
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
