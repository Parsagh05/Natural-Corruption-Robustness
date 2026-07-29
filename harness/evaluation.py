# -*- coding: utf-8 -*-
"""
evaluation.py - Evaluation harness core that computes and structures CSV outputs
matching the survey's schema.

Output CSV format:
    - {model_name}_{dataset_name}_PX.csv  (pixel-level metrics)
    - {model_name}_{dataset_name}_SP.csv  (image-level metrics)
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .config import DatasetConfig, HarnessConfig
from .aaclip_scoring import normalize_aaclip_maps
from .metrics import (
    DEFAULT_PIXEL_METRIC_SIZE,
    compute_image_metrics,
    compute_pixel_metrics,
    resize_anomaly_map,
    resize_mask,
)
from .result_order import condition_sort_key


class EvaluationHarness:
    """
    Computes and exports evaluation metrics in the survey's CSV schema.

    SP columns: AUROC, AP, F1-Max, optimum threshold.
    PX columns: AUROC, AUPRO, F1-Max, optimum threshold.
    """

    def __init__(self, output_root: Path, config: HarnessConfig):
        self.output_root = output_root
        self.config = config
        self.results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
        self.fine_grained_results: Dict[
            str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
        ] = {}
        self.per_image_scores: Dict[str, Dict[str, list]] = {}

    @staticmethod
    def _record_in_store(
        store: Dict,
        model_name: str,
        dataset_name: str,
        category: str,
        corruption_type: str,
        severity: int,
        pixel_metrics: Dict[str, float],
        image_metrics: Dict[str, float],
    ) -> None:
        level_key = f"{corruption_type}_level {severity}"
        store.setdefault(model_name, {})
        store[model_name].setdefault(dataset_name, {})
        store[model_name][dataset_name].setdefault(level_key, {})
        store[model_name][dataset_name][level_key][category] = {
            **pixel_metrics,
            **image_metrics,
        }

    def record_metrics(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        corruption_type: str,
        severity: int,
        pixel_metrics: Dict[str, float],
        image_metrics: Dict[str, float],
    ) -> None:
        """Record computed metrics for a single evaluation condition."""
        self._record_in_store(
            self.results,
            model_name,
            dataset_name,
            category,
            corruption_type,
            severity,
            pixel_metrics,
            image_metrics,
        )

    def record_fine_grained_metrics(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        corruption_type: str,
        severity: int,
        pixel_metrics: Dict[str, float],
        image_metrics: Dict[str, float],
    ) -> None:
        """Record one concrete-corruption result from a categorized run."""
        self._record_in_store(
            self.fine_grained_results,
            model_name,
            dataset_name,
            category,
            corruption_type,
            severity,
            pixel_metrics,
            image_metrics,
        )

    def record_per_image_scores(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        corruption_category: str,
        severity: int,
        scores: np.ndarray,
        metadata: list,
    ) -> None:
        """Save JSON-serializable image scores in inference-array order."""
        if len(scores) != len(metadata):
            raise ValueError(
                "Per-image score/metadata length mismatch: "
                f"{len(scores)} scores for {len(metadata)} records"
            )
        records = self.per_image_scores.setdefault(model_name, {}).setdefault(
            dataset_name, []
        )
        artifact_dir = (
            Path(dataset_name)
            / category
            / corruption_category
            / f"level_{severity}"
        )
        for index, (score, meta) in enumerate(zip(scores, metadata)):
            selected_corruption = (
                meta.get("selected_corruption") or corruption_category
            )
            records.append({
                "model": model_name,
                "dataset": dataset_name,
                "class_name": category,
                "corruption_category": corruption_category,
                "selected_corruption": selected_corruption,
                "severity": severity,
                "transformation_level": (
                    f"{selected_corruption}_level {severity}"
                ),
                "sample_id": meta.get("sample_id"),
                "label": int(meta["label"]),
                "image_path": meta.get("image_path"),
                "mask_path": meta.get("mask_path"),
                "image_score": float(score),
                "raw_scores_file": (
                    artifact_dir / "raw_scores.npy"
                ).as_posix(),
                "raw_scores_index": index,
                "lowres_maps_file": (
                    artifact_dir / "lowres_maps.npy"
                ).as_posix(),
                "lowres_maps_index": index,
            })

    def _export_csv_store(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
        result_store: Dict,
        filename_tag: str = "",
        require_all_categories: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Export results to CSV files matching the survey schema.

        Returns:
            Tuple of (px_csv_path, sp_csv_path).
        """
        dataset_name = dataset_config.name
        categories = dataset_config.categories

        if model_name not in result_store:
            return None, None
        if dataset_name not in result_store[model_name]:
            return None, None

        model_results = result_store[model_name][dataset_name]
        if require_all_categories:
            for level_key, category_results in model_results.items():
                missing_categories = [
                    category
                    for category in categories
                    if category not in category_results
                ]
                if missing_categories:
                    raise ValueError(
                        f"Fine-grained condition {level_key!r} is missing "
                        f"object classes: {missing_categories}"
                    )
        csv_dir = self.output_root / model_name
        csv_dir.mkdir(parents=True, exist_ok=True)

        px_path = csv_dir / f"{model_name}_{dataset_name}{filename_tag}_PX.csv"
        sp_path = csv_dir / f"{model_name}_{dataset_name}{filename_tag}_SP.csv"

        px_header = ["transformation_level"]
        for cat in categories:
            px_header.extend([
                f"{cat}_auroc_px",
                f"{cat}_aupro_px",
                f"{cat}_f1_px",
                f"{cat}_threshold_px",
            ])
        px_header.extend([
            "mean_auroc_px",
            "mean_aupro_px",
            "mean_f1_px",
            "mean_threshold_px",
        ])

        px_rows = []
        for level_key in sorted(model_results.keys(), key=condition_sort_key):
            row = [level_key]
            cat_aurocs, cat_aupros, cat_f1s, cat_thresholds = [], [], [], []

            for cat in categories:
                metrics = model_results[level_key].get(cat, {})
                auroc = metrics.get("auroc_px", 0.0)
                aupro = metrics.get("aupro_px", 0.0)
                f1 = metrics.get("f1_px", 0.0)
                threshold = metrics.get("threshold_px", 0.0)
                row.extend([auroc, aupro, f1, threshold])
                cat_aurocs.append(auroc)
                cat_aupros.append(aupro)
                cat_f1s.append(f1)
                cat_thresholds.append(threshold)

            row.extend([
                round(np.mean(cat_aurocs), 2),
                round(np.mean(cat_aupros), 2),
                round(np.mean(cat_f1s), 2),
                round(np.mean(cat_thresholds), 6),
            ])
            px_rows.append(row)

        with open(px_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(px_header)
            writer.writerows(px_rows)

        sp_header = ["transformation_level"]
        for cat in categories:
            sp_header.extend([
                f"{cat}_auroc_sp",
                f"{cat}_ap_sp",
                f"{cat}_f1_sp",
                f"{cat}_threshold_sp",
            ])
        sp_header.extend([
            "mean_auroc_sp",
            "mean_ap_sp",
            "mean_f1_sp",
            "mean_threshold_sp",
        ])

        sp_rows = []
        for level_key in sorted(model_results.keys(), key=condition_sort_key):
            row = [level_key]
            cat_aurocs, cat_aps, cat_f1s, cat_thresholds = [], [], [], []

            for cat in categories:
                metrics = model_results[level_key].get(cat, {})
                auroc = metrics.get("auroc_sp", 0.0)
                ap = metrics.get("ap_sp", 0.0)
                f1 = metrics.get("f1_sp", 0.0)
                threshold = metrics.get("threshold_sp", 0.0)
                row.extend([auroc, ap, f1, threshold])
                cat_aurocs.append(auroc)
                cat_aps.append(ap)
                cat_f1s.append(f1)
                cat_thresholds.append(threshold)

            row.extend([
                round(np.mean(cat_aurocs), 2),
                round(np.mean(cat_aps), 2),
                round(np.mean(cat_f1s), 2),
                round(np.mean(cat_thresholds), 6),
            ])
            sp_rows.append(row)

        with open(sp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(sp_header)
            writer.writerows(sp_rows)

        return px_path, sp_path

    def export_csv(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
    ) -> Tuple[Path, Path]:
        """Export the standard category-level result CSVs."""
        return self._export_csv_store(
            model_name,
            dataset_config,
            self.results,
        )

    def export_categorized_fine_grained(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Export categorized subtype SP/PX CSVs and per-image score JSON."""
        px_path, sp_path = self._export_csv_store(
            model_name,
            dataset_config,
            self.fine_grained_results,
            filename_tag="_FINE_GRAINED",
            require_all_categories=True,
        )
        records = self.per_image_scores.get(model_name, {}).get(
            dataset_config.name, []
        )
        if not records:
            return px_path, sp_path, None

        json_dir = self.output_root / model_name
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / (
            f"{model_name}_{dataset_config.name}_FINE_GRAINED_PER_IMAGE.json"
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=json_dir,
                prefix=f".{json_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temporary_path = Path(output_file.name)
                json.dump(
                    {
                        "model": model_name,
                        "dataset": dataset_config.name,
                        "protocol": (
                            "categorized_fine_grained_assigned_subsets"
                        ),
                        "description": (
                            "Concrete-corruption metrics use the image subsets "
                            "assigned by the categorized corruption plan."
                        ),
                        "records": records,
                    },
                    output_file,
                    indent=2,
                    allow_nan=False,
                )
                output_file.write("\n")
            os.replace(temporary_path, json_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return px_path, sp_path, json_path

    def discard_model_results(self, model_name: str) -> None:
        """Release completed result buffers after their files are archived."""
        self.results.pop(model_name, None)
        self.fine_grained_results.pop(model_name, None)
        self.per_image_scores.pop(model_name, None)

    def evaluate_from_artifacts(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
        artifacts_root: Path,
    ) -> None:
        """Load saved artifacts and compute all metrics."""
        dataset_name = dataset_config.name

        for corruption_type, severity in self.config.evaluation_conditions:
            for category in dataset_config.categories:
                artifact_dir = (
                    artifacts_root
                    / model_name
                    / dataset_name
                    / category
                    / corruption_type
                    / f"level_{severity}"
                )

                if not artifact_dir.exists():
                    continue

                scores_path = artifact_dir / "raw_scores.npy"
                maps_path = artifact_dir / "lowres_maps.npy"
                meta_path = artifact_dir / "metadata.json"

                if not all(p.exists() for p in [scores_path, maps_path, meta_path]):
                    continue

                scores = np.load(scores_path)
                lowres_maps = np.load(maps_path)
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                labels = np.array([m["label"] for m in metadata])
                image_metrics = compute_image_metrics(labels, scores)

                if lowres_maps.ndim == 3 and lowres_maps.shape[0] > 0:
                    metric_size = DEFAULT_PIXEL_METRIC_SIZE
                    resized_masks = []
                    resized_maps = []
                    for meta in metadata:
                        mask_path = meta.get("mask_path")
                        if mask_path and Path(mask_path).exists():
                            mask = np.array(Image.open(mask_path).convert("L"))
                            mask = (mask > 0).astype(np.float32)
                            mask_resized = resize_mask(mask, metric_size, metric_size)
                        else:
                            mask_resized = np.zeros(
                                (metric_size, metric_size), dtype=np.float32
                            )
                        resized_masks.append(mask_resized)

                    for lowres_map in lowres_maps:
                        resized_maps.append(
                            resize_anomaly_map(
                                lowres_map,
                                metric_size,
                                metric_size,
                                align_corners=(model_name == "AA-CLIP"),
                            )
                        )

                    if model_name == "AA-CLIP":
                        # New AA-CLIP artifacts store already-fused image
                        # scores and raw low-resolution maps. Reapply only the
                        # official per-class map normalization here.
                        resized_maps = normalize_aaclip_maps(resized_maps)

                    pixel_metrics = compute_pixel_metrics(
                        resized_masks,
                        resized_maps,
                        aupro_device=self.config.device,
                    )
                else:
                    pixel_metrics = {
                        "auroc_px": 0.0,
                        "f1_px": 0.0,
                        "aupro_px": 0.0,
                        "threshold_px": 0.0,
                    }

                self.record_metrics(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    category=category,
                    corruption_type=corruption_type,
                    severity=severity,
                    pixel_metrics=pixel_metrics,
                    image_metrics=image_metrics,
                )
