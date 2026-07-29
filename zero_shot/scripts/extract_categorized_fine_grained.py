#!/usr/bin/env python3
"""Extract fine-grained metrics from categorized evaluation artifacts.

Categorized evaluation stores one category-level artifact for each object class
and severity.  Every metadata record identifies the concrete corruption that
was assigned to that image.  This script joins that metadata with the matching
entry in ``raw_scores.npy`` and computes metrics for each concrete corruption.

The resulting metrics describe the assigned subsets of the categorized
protocol.  They are not equivalent to the uncategorized protocol, where every
concrete corruption is applied to every image.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from harness.config import MVTEC_CATEGORIES, VISA_CATEGORIES  # noqa: E402
from harness.corruption import (  # noqa: E402
    CATEGORIZED_CORRUPTIONS,
    apply_corruption_to_mask,
)
from harness.metrics import (  # noqa: E402
    DEFAULT_PIXEL_METRIC_SIZE,
    compute_image_metrics,
    compute_pixel_metrics,
    resize_anomaly_map,
    resize_mask,
)


DATASET_CATEGORIES = {
    "MVTec": MVTEC_CATEGORIES,
    "VisA": VISA_CATEGORIES,
}

CATEGORY_CORRUPTIONS = CATEGORIZED_CORRUPTIONS

FINE_GRAINED_CORRUPTIONS = tuple(
    corruption
    for category in ("noise", "blur", "photometric", "geometric")
    for corruption in CATEGORY_CORRUPTIONS[category]
)

SP_METRIC_NAMES = ("auroc_sp", "ap_sp", "f1_sp", "threshold_sp")
PX_METRIC_NAMES = ("auroc_px", "aupro_px", "f1_px", "threshold_px")


def _atomic_write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, allow_nan=False)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _condition_sort_key(condition: Tuple[str, int]) -> Tuple[int, int]:
    corruption, severity = condition
    return FINE_GRAINED_CORRUPTIONS.index(corruption), severity


def _metric_header(categories: Sequence[str], metric_names: Sequence[str]) -> List[str]:
    header = ["transformation_level"]
    for category in categories:
        header.extend(f"{category}_{metric_name}" for metric_name in metric_names)
    header.extend(f"mean_{metric_name}" for metric_name in metric_names)
    return header


def _metric_rows(
    results: Mapping[Tuple[str, int], Mapping[str, Mapping[str, float]]],
    categories: Sequence[str],
    metric_names: Sequence[str],
) -> List[List[object]]:
    rows: List[List[object]] = []
    for corruption, severity in sorted(results, key=_condition_sort_key):
        condition_results = results[(corruption, severity)]
        missing = [category for category in categories if category not in condition_results]
        if missing:
            raise ValueError(
                f"{corruption}_level {severity} is missing object classes: {missing}"
            )

        row: List[object] = [f"{corruption}_level {severity}"]
        metric_columns: Dict[str, List[float]] = {
            metric_name: [] for metric_name in metric_names
        }
        for category in categories:
            metrics = condition_results[category]
            for metric_name in metric_names:
                value = metrics[metric_name]
                row.append(value)
                metric_columns[metric_name].append(value)

        for metric_name in metric_names:
            precision = 6 if metric_name.startswith("threshold_") else 2
            row.append(round(float(np.mean(metric_columns[metric_name])), precision))
        rows.append(row)
    return rows


def _parse_artifact_path(
    dataset_root: Path, metadata_path: Path
) -> Tuple[str, str, int]:
    relative_parts = metadata_path.relative_to(dataset_root).parts
    if len(relative_parts) != 4 or relative_parts[-1] != "metadata.json":
        raise ValueError(f"Unexpected artifact path: {metadata_path}")
    class_name, category, level_name, _ = relative_parts
    if category not in CATEGORY_CORRUPTIONS:
        raise ValueError(f"Unknown corruption category {category!r}: {metadata_path}")
    if not level_name.startswith("level_"):
        raise ValueError(f"Invalid severity directory {level_name!r}: {metadata_path}")
    try:
        severity = int(level_name.removeprefix("level_"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid severity directory {level_name!r}: {metadata_path}"
        ) from exc
    return class_name, category, severity


def _mask_candidate(
    stored_path: Optional[str], dataset_root: Path, class_name: str
) -> Optional[Path]:
    if not stored_path:
        return None
    direct_path = Path(stored_path)
    if direct_path.exists():
        return direct_path

    normalized_parts = PurePosixPath(stored_path.replace("\\", "/")).parts
    try:
        class_index = normalized_parts.index(class_name)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Cannot map stored mask path for class {class_name!r}: {stored_path}"
        ) from exc
    candidate = dataset_root.joinpath(*normalized_parts[class_index:])
    if not candidate.exists():
        raise FileNotFoundError(
            f"Ground-truth mask not found. Tried stored path {stored_path!r} "
            f"and remapped path {candidate}"
        )
    return candidate


def _relative_dataset_path(stored_path: str, class_name: str) -> str:
    """Recover the dataset-relative path used for deterministic corruption."""
    normalized_parts = PurePosixPath(stored_path.replace("\\", "/")).parts
    try:
        class_index = normalized_parts.index(class_name)
    except ValueError as exc:
        raise ValueError(
            f"Cannot recover dataset-relative path for class {class_name!r}: "
            f"{stored_path}"
        ) from exc
    return PurePosixPath(*normalized_parts[class_index:]).as_posix()


def extract_dataset(
    categorized_root: Path,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    pixel_dataset_root: Optional[Path] = None,
    corruption_seed: int = 111,
) -> Dict[str, Path]:
    """Extract one dataset and return the paths written."""
    if dataset_name not in DATASET_CATEGORIES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    dataset_artifacts = categorized_root / dataset_name
    if not dataset_artifacts.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {dataset_artifacts}")

    categories = DATASET_CATEGORIES[dataset_name]
    metadata_paths = sorted(dataset_artifacts.rglob("metadata.json"))
    expected_artifact_count = len(categories) * len(CATEGORY_CORRUPTIONS) * 4
    if len(metadata_paths) != expected_artifact_count:
        raise ValueError(
            f"Expected {expected_artifact_count} metadata files for {dataset_name}; "
            f"found {len(metadata_paths)}"
        )

    sp_groups: Dict[Tuple[str, int, str], Dict[str, List]] = defaultdict(
        lambda: {"labels": [], "scores": []}
    )
    px_groups: Dict[Tuple[str, int, str], Dict[str, List]] = defaultdict(
        lambda: {"masks": [], "maps": []}
    )
    per_image_records: List[Dict[str, object]] = []
    seen_artifacts = set()

    for metadata_path in metadata_paths:
        class_name, corruption_category, severity = _parse_artifact_path(
            dataset_artifacts, metadata_path
        )
        if class_name not in categories:
            raise ValueError(f"Unexpected {dataset_name} object class: {class_name}")
        artifact_key = (class_name, corruption_category, severity)
        if artifact_key in seen_artifacts:
            raise ValueError(f"Duplicate artifact group: {artifact_key}")
        seen_artifacts.add(artifact_key)

        scores_path = metadata_path.parent / "raw_scores.npy"
        maps_path = metadata_path.parent / "lowres_maps.npy"
        if not scores_path.is_file() or not maps_path.is_file():
            raise FileNotFoundError(
                f"Incomplete artifact group beside {metadata_path}: expected "
                "raw_scores.npy and lowres_maps.npy"
            )

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        scores = np.load(scores_path, allow_pickle=False)
        lowres_maps = np.load(maps_path, allow_pickle=False)
        if scores.ndim != 1 or lowres_maps.ndim != 3:
            raise ValueError(
                f"Unexpected array shapes beside {metadata_path}: "
                f"scores={scores.shape}, maps={lowres_maps.shape}"
            )
        if not (len(metadata) == len(scores) == len(lowres_maps)):
            raise ValueError(
                f"Artifact length mismatch beside {metadata_path}: "
                f"metadata={len(metadata)}, scores={len(scores)}, "
                f"maps={len(lowres_maps)}"
            )
        if not np.isfinite(scores).all() or not np.isfinite(lowres_maps).all():
            raise ValueError(f"Non-finite score or map found beside {metadata_path}")

        relative_scores_path = scores_path.relative_to(categorized_root).as_posix()
        relative_maps_path = maps_path.relative_to(categorized_root).as_posix()
        for index, (meta, score, lowres_map) in enumerate(
            zip(metadata, scores, lowres_maps)
        ):
            corruption = meta.get("selected_corruption")
            if corruption not in CATEGORY_CORRUPTIONS[corruption_category]:
                raise ValueError(
                    f"Invalid concrete corruption {corruption!r} for category "
                    f"{corruption_category!r} in {metadata_path} record {index}"
                )
            label = int(meta["label"])
            if label not in (0, 1):
                raise ValueError(
                    f"Invalid binary label {label!r} in {metadata_path} record {index}"
                )

            group_key = (corruption, severity, class_name)
            sp_groups[group_key]["labels"].append(label)
            sp_groups[group_key]["scores"].append(float(score))

            if pixel_dataset_root is not None:
                if label == 0:
                    mask = np.zeros(
                        (DEFAULT_PIXEL_METRIC_SIZE, DEFAULT_PIXEL_METRIC_SIZE),
                        dtype=np.float32,
                    )
                else:
                    mask_path = _mask_candidate(
                        meta.get("mask_path"), pixel_dataset_root, class_name
                    )
                    if mask_path is None:
                        raise FileNotFoundError(
                            f"An anomalous image has no mask path in {metadata_path} "
                            f"record {index}"
                        )
                    mask_image = Image.open(mask_path).convert("L")
                    stored_image_path = meta.get("image_path")
                    if not stored_image_path:
                        raise ValueError(
                            f"An anomalous image has no image path in "
                            f"{metadata_path} record {index}"
                        )
                    relative_image_path = _relative_dataset_path(
                        stored_image_path, class_name
                    )
                    mask_image = apply_corruption_to_mask(
                        mask_image,
                        corruption,
                        severity,
                        relative_image_path,
                        corruption_seed,
                    )
                    mask = np.asarray(mask_image)
                    mask = resize_mask(
                        (mask > 0).astype(np.float32),
                        DEFAULT_PIXEL_METRIC_SIZE,
                        DEFAULT_PIXEL_METRIC_SIZE,
                    )
                anomaly_map = resize_anomaly_map(
                    lowres_map,
                    DEFAULT_PIXEL_METRIC_SIZE,
                    DEFAULT_PIXEL_METRIC_SIZE,
                )
                px_groups[group_key]["masks"].append(mask)
                px_groups[group_key]["maps"].append(anomaly_map)

            per_image_records.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "corruption_category": corruption_category,
                    "selected_corruption": corruption,
                    "severity": severity,
                    "transformation_level": (
                        f"{corruption}_level {severity}"
                    ),
                    "sample_id": meta.get("sample_id"),
                    "label": label,
                    "image_path": meta.get("image_path"),
                    "mask_path": meta.get("mask_path"),
                    "image_score": float(score),
                    "raw_scores_file": relative_scores_path,
                    "raw_scores_index": index,
                    "lowres_maps_file": relative_maps_path,
                    "lowres_maps_index": index,
                }
            )

    sp_results: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (corruption, severity, class_name), values in sp_groups.items():
        labels = np.asarray(values["labels"], dtype=np.int64)
        scores = np.asarray(values["scores"], dtype=np.float32)
        if np.unique(labels).size != 2:
            raise ValueError(
                f"Cannot compute image metrics for {dataset_name}/{class_name}/"
                f"{corruption}_level {severity}: assigned subset does not contain "
                "both normal and anomalous images"
            )
        sp_results[(corruption, severity)][class_name] = compute_image_metrics(
            labels, scores
        )

    expected_conditions = {
        (corruption, severity)
        for corruption in FINE_GRAINED_CORRUPTIONS
        for severity in range(1, 5)
    }
    if set(sp_results) != expected_conditions:
        missing = sorted(expected_conditions - set(sp_results), key=_condition_sort_key)
        extra = sorted(set(sp_results) - expected_conditions, key=_condition_sort_key)
        raise ValueError(
            f"Fine-grained condition mismatch for {dataset_name}; "
            f"missing={missing}, extra={extra}"
        )

    base_name = f"{model_name}_{dataset_name}_FINE_GRAINED"
    sp_path = output_dir / f"{base_name}_SP.csv"
    json_path = output_dir / f"{base_name}_PER_IMAGE.json"
    _atomic_write_csv(
        sp_path,
        _metric_header(categories, SP_METRIC_NAMES),
        _metric_rows(sp_results, categories, SP_METRIC_NAMES),
    )
    _atomic_write_json(
        json_path,
        {
            "model": model_name,
            "dataset": dataset_name,
            "protocol": "categorized_fine_grained_assigned_subsets",
            "description": (
                "Each concrete-corruption metric uses only the images assigned "
                "that corruption by the categorized protocol."
            ),
            "records": per_image_records,
        },
    )

    written = {"sp_csv": sp_path, "per_image_json": json_path}
    if pixel_dataset_root is not None:
        px_results: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = defaultdict(dict)
        for (corruption, severity, class_name), values in px_groups.items():
            px_results[(corruption, severity)][class_name] = compute_pixel_metrics(
                values["masks"], values["maps"]
            )
        px_path = output_dir / f"{base_name}_PX.csv"
        _atomic_write_csv(
            px_path,
            _metric_header(categories, PX_METRIC_NAMES),
            _metric_rows(px_results, categories, PX_METRIC_NAMES),
        )
        written["px_csv"] = px_path

    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-corruption metrics and per-image scores from "
            "categorized evaluation artifacts."
        )
    )
    parser.add_argument(
        "--categorized-root",
        type=Path,
        default=REPOSITORY_ROOT / "AF-CLIP" / "categorized",
        help="Directory containing the MVTec and VisA artifact trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <categorized-root>/fine_grained).",
    )
    parser.add_argument("--model-name", default="AF-CLIP")
    parser.add_argument(
        "--mvtec-root",
        type=Path,
        help="Optional MVTec dataset root; enables fine-grained PX metrics.",
    )
    parser.add_argument(
        "--visa-root",
        type=Path,
        help="Optional VisA dataset root; enables fine-grained PX metrics.",
    )
    parser.add_argument(
        "--corruption-seed",
        type=int,
        default=None,
        help=(
            "Seed used by the categorized evaluation for deterministic "
            "geometric mask transforms. Required when a dataset root is "
            "provided."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    categorized_root = args.categorized_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else categorized_root / "fine_grained"
    )
    dataset_roots = {
        "MVTec": args.mvtec_root.resolve() if args.mvtec_root else None,
        "VisA": args.visa_root.resolve() if args.visa_root else None,
    }
    if any(dataset_roots.values()) and args.corruption_seed is None:
        print(
            "ERROR: --corruption-seed is required with --mvtec-root or "
            "--visa-root so geometric masks use the original run seed.",
            file=sys.stderr,
        )
        return 2

    try:
        for dataset_name in DATASET_CATEGORIES:
            written = extract_dataset(
                categorized_root=categorized_root,
                output_dir=output_dir,
                model_name=args.model_name,
                dataset_name=dataset_name,
                pixel_dataset_root=dataset_roots[dataset_name],
                corruption_seed=(
                    args.corruption_seed
                    if args.corruption_seed is not None
                    else 111
                ),
            )
            print(f"{dataset_name}:")
            for output_type, path in written.items():
                print(f"  {output_type}: {path}")
            if dataset_roots[dataset_name] is None:
                print(
                    "  px_csv: skipped (provide the dataset root to access "
                    "ground-truth masks)"
                )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
