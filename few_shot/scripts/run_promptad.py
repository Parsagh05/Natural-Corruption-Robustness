"""Run retrained PromptAD few-shot robustness evaluations."""

import argparse

from few_shot.harness.models import discover_promptad_checkpoints
from few_shot.harness.runner import run_promptad_evaluations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired PromptAD CLS/SEG checkpoints on clean and "
            "corrupted MVTec AD and/or VisA test data."
        )
    )
    parser.add_argument("--mvtec-root")
    parser.add_argument("--visa-root")
    parser.add_argument("--promptad-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--dataset", choices=("mvtec", "visa", "both"), default="both"
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--corruption-types", nargs="+")
    parser.add_argument(
        "--severity-levels", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--uncategorized",
        action="store_true",
        help="Interpret corruption names as concrete operations instead of groups.",
    )
    parser.add_argument("--corruption-cache-root")
    parser.add_argument(
        "--corruption-cache-format", choices=("png", "jpeg"), default="png"
    )
    parser.add_argument("--corruption-seed", type=int, default=123)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--verify-checkpoint-hashes", action="store_true")
    parser.add_argument("--no-strict-source-commit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = ("mvtec", "visa") if args.dataset == "both" else (args.dataset,)
    checkpoints = discover_promptad_checkpoints(
        args.checkpoint_root,
        shots=args.shots,
        datasets=datasets,
        verify_hashes=args.verify_checkpoint_hashes,
    )
    run_promptad_evaluations(
        mvtec_root=args.mvtec_root,
        visa_root=args.visa_root,
        output_root=args.output_root,
        promptad_root=args.promptad_root,
        checkpoint_paths=checkpoints,
        shots=args.shots,
        datasets=datasets,
        device=args.device,
        batch_size=args.batch_size,
        corruption_types=args.corruption_types,
        severity_levels=args.severity_levels,
        categorized_corruptions=not args.uncategorized,
        corruption_cache_root=args.corruption_cache_root,
        corruption_cache_format=args.corruption_cache_format,
        corruption_seed=args.corruption_seed,
        include_clean=not args.no_clean,
        strict_source_commit=not args.no_strict_source_commit,
    )


if __name__ == "__main__":
    main()
