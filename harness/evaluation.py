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
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from .config import DatasetConfig, HarnessConfig
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
        level_key = f"{corruption_type}_level {severity}"

        self.results.setdefault(model_name, {})
        self.results[model_name].setdefault(dataset_name, {})
        self.results[model_name][dataset_name].setdefault(level_key, {})
        self.results[model_name][dataset_name][level_key][category] = {
            **pixel_metrics,
            **image_metrics,
        }

    def export_csv(
        self,
        model_name: str,
        dataset_config: DatasetConfig,
    ) -> Tuple[Path, Path]:
        """
        Export results to CSV files matching the survey schema.

        Returns:
            Tuple of (px_csv_path, sp_csv_path).
        """
        dataset_name = dataset_config.name
        categories = dataset_config.categories

        if model_name not in self.results:
            return None, None
        if dataset_name not in self.results[model_name]:
            return None, None

        model_results = self.results[model_name][dataset_name]
        csv_dir = self.output_root / model_name
        csv_dir.mkdir(parents=True, exist_ok=True)

        px_path = csv_dir / f"{model_name}_{dataset_name}_PX.csv"
        sp_path = csv_dir / f"{model_name}_{dataset_name}_SP.csv"

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
                            resize_anomaly_map(lowres_map, metric_size, metric_size)
                        )

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
