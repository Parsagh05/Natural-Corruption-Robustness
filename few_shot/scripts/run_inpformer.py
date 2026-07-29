"""Run selected official INP-Former few-shot robustness evaluations."""

import argparse
from pathlib import Path

import torch

from few_shot.harness.models import discover_official_checkpoints
from few_shot.harness.runner import run_official_evaluations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate official INP-Former 1/2/4-shot checkpoints on categorized "
            "MVTec AD and VisA corruptions (two evaluations per shot)."
        )
    )
    parser.add_argument("--mvtec-root", required=True)
    parser.add_argument("--visa-root", required=True)
    parser.add_argument("--inpformer-root", required=True)
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="Root containing the selected official checkpoint directories.",
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        choices=(1, 2, 4),
        default=[1, 2, 4],
        help="Official shot settings to run; each evaluates both datasets.",
    )
    parser.add_argument("--corruption-cache-root")
    parser.add_argument(
        "--corruption-cache-format", choices=("png", "jpeg"), default="png"
    )
    parser.add_argument("--corruption-seed", type=int, default=123)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Omit the clean baseline; categorized corruptions still run.",
    )
    parser.add_argument(
        "--strict-source-commit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the pinned official source commit (default: enabled).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoints = discover_official_checkpoints(
        args.checkpoint_root, shots=args.shots
    )
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    run_official_evaluations(
        mvtec_root=args.mvtec_root,
        visa_root=args.visa_root,
        output_root=args.output_root,
        inpformer_root=args.inpformer_root,
        checkpoint_paths=checkpoints,
        shots=args.shots,
        device=args.device,
        batch_size=args.batch_size,
        corruption_cache_root=args.corruption_cache_root,
        corruption_cache_format=args.corruption_cache_format,
        corruption_seed=args.corruption_seed,
        include_clean=not args.no_clean,
        strict_source_commit=args.strict_source_commit,
    )


if __name__ == "__main__":
    main()
