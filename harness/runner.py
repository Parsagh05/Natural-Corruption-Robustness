# -*- coding: utf-8 -*-
"""
runner.py - Main execution pipeline orchestrating model inference,
corruption application, artifact storage, and evaluation.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import (
    HarnessConfig,
    DatasetConfig,
    CORRUPTION_TYPES,
    SEVERITY_LEVELS,
    COMPLETED_MODELS,
)
from .seed import set_global_seed
from .corruption import apply_corruption
from .dataset import AnomalyDetectionDataset
from .models import get_model, BaseModelWrapper
from .storage import ArtifactStorage, IncrementalAccumulator
from .evaluation import EvaluationHarness
from .metrics import (
    DEFAULT_PIXEL_METRIC_SIZE,
    compute_image_metrics,
    compute_pixel_metrics,
    resize_anomaly_map,
    resize_mask,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _first_existing_root(paths: List[Optional[str]]) -> Optional[str]:
    for path in paths:
        if path and Path(path).exists():
            return path
    return None


def _normalize_dataset_selector(dataset: str) -> str:
    normalized = dataset.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "both": "both",
        "all": "both",
        "mvtec": "mvtec",
        "mvtec_ad": "mvtec",
        "mvt_ec": "mvtec",
        "visa": "visa",
    }
    if normalized not in aliases:
        raise ValueError(
            "dataset must be one of: 'both', 'mvtec', or 'visa'"
        )
    return aliases[normalized]


class RobustnessRunner:
    """
    Main orchestrator for the robustness evaluation pipeline.

    Processes models sequentially, applying corruptions on demand,
    saving artifacts incrementally, and computing evaluation metrics.
    """

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.storage = ArtifactStorage(config.output_root)
        self.eval_harness = EvaluationHarness(config.output_root, config)

        # Enforce deterministic execution
        set_global_seed(config.seed)

    def run(
        self,
        models: Optional[List[str]] = None,
        datasets: Optional[List[DatasetConfig]] = None,
    ) -> None:
        """
        Execute the full evaluation pipeline.

        Args:
            models: List of model names to evaluate. Defaults to all models.
            datasets: List of dataset configs. Defaults to config.datasets.
        """
        models = models or self.config.models
        datasets = datasets or self.config.datasets

        total_start = time.time()
        logger.info(
            f"Starting robustness evaluation: "
            f"{len(models)} models × {len(datasets)} datasets × "
            f"{len(self.config.corruption_types)} corruptions × "
            f"{len(self.config.severity_levels)} levels"
        )

        for model_name in models:
            self._run_single_model(model_name, datasets)

        elapsed = time.time() - total_start
        logger.info(f"Full evaluation completed in {elapsed:.1f}s")

    def _run_single_model(
        self, model_name: str, datasets: List[DatasetConfig]
    ) -> None:
        """Run evaluation for a single model across all datasets and corruptions."""
        logger.info(f"{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        # Load model
        try:
            model_wrapper = get_model(
                model_name,
                device=self.config.device,
                **self.config.model_kwargs.get(model_name, {}),
            )
            model_wrapper.load_model()
        except Exception as e:
            logger.exception(f"Failed to load model {model_name}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

        for dataset_config in datasets:
            self._run_model_on_dataset(model_wrapper, dataset_config)

            # Export CSVs for this model+dataset
            px_path, sp_path = self.eval_harness.export_csv(
                model_name, dataset_config
            )
            if px_path:
                logger.info(f"Exported: {px_path}")
            if sp_path:
                logger.info(f"Exported: {sp_path}")

        # Release model and clean up
        model_wrapper.release()

        # Zip artifacts and cleanup disk
        zip_path = self.storage.zip_and_cleanup_model(model_name)
        if zip_path:
            logger.info(f"Archived: {zip_path}")

        ArtifactStorage.cleanup_memory()
        logger.info(f"Completed model: {model_name}")

    def _run_model_on_dataset(
        self, model_wrapper: BaseModelWrapper, dataset_config: DatasetConfig
    ) -> None:
        """Run model on all categories and corruptions for a single dataset."""
        dataset_name = dataset_config.name
        model_name = model_wrapper.model_name

        logger.info(f"  Dataset: {dataset_name}")

        for corruption_type in self.config.corruption_types:
            for severity in self.config.severity_levels:
                logger.info(
                    f"    Corruption: {corruption_type} | Level: {severity}"
                )

                level_datasets = []
                for category in dataset_config.categories:
                    category_dataset = AnomalyDetectionDataset(
                        config=dataset_config,
                        corruption_type=corruption_type,
                        severity=severity,
                        base_seed=(
                            self.config.seed
                            if self.config.corruption_seed is None
                            else self.config.corruption_seed
                        ),
                        category=category,
                        corruption_cache_root=self.config.corruption_cache_root,
                        corruption_cache_format=self.config.corruption_cache_format,
                        categorized_corruptions=self.config.categorized_corruptions,
                        categorized_corruption_plan=self.config.categorized_corruption_plans.get(
                            dataset_config.name.lower()
                        ),
                    )
                    if len(category_dataset) > 0:
                        level_datasets.append((category, category_dataset))

                total_images = sum(len(category_dataset) for _, category_dataset in level_datasets)
                progress_desc = f"{dataset_name} {corruption_type} L{severity}"
                with tqdm(
                    total=total_images,
                    desc=progress_desc,
                    unit="img",
                    leave=True,
                    dynamic_ncols=True,
                ) as progress_bar:
                    for category, category_dataset in level_datasets:
                        self._run_category(
                            model_wrapper=model_wrapper,
                            dataset_config=dataset_config,
                            category=category,
                            corruption_type=corruption_type,
                            severity=severity,
                            dataset=category_dataset,
                            progress_bar=progress_bar,
                        )

                # Periodic memory cleanup
                ArtifactStorage.cleanup_memory()

    def _run_category(
        self,
        model_wrapper: BaseModelWrapper,
        dataset_config: DatasetConfig,
        category: str,
        corruption_type: str,
        severity: int,
        dataset: Optional[AnomalyDetectionDataset] = None,
        progress_bar: Optional[tqdm] = None,
    ) -> None:
        """Run inference on a single category with a specific corruption."""
        if dataset is None:
            dataset = AnomalyDetectionDataset(
                config=dataset_config,
                corruption_type=corruption_type,
                severity=severity,
                base_seed=(
                    self.config.seed
                    if self.config.corruption_seed is None
                    else self.config.corruption_seed
                ),
                category=category,
                corruption_cache_root=self.config.corruption_cache_root,
                corruption_cache_format=self.config.corruption_cache_format,
                categorized_corruptions=self.config.categorized_corruptions,
                categorized_corruption_plan=self.config.categorized_corruption_plans.get(
                    dataset_config.name.lower()
                ),
            )

        if len(dataset) == 0:
            return

        accumulator = IncrementalAccumulator()
        metric_masks = []
        model_name = model_wrapper.model_name
        dataset_name = dataset_config.name

        batch_size = max(1, int(self.config.batch_size))
        for start_idx in range(0, len(dataset), batch_size):
            batch_indices = range(start_idx, min(start_idx + batch_size, len(dataset)))
            samples = [dataset[idx] for idx in batch_indices]
            images = [sample["image"] for sample in samples]

            try:
                scores, lowres_maps = model_wrapper.forward_raw_batch(
                    images, category=category
                )
            except Exception as e:
                logger.warning(
                    f"      Batch inference error on {category} "
                    f"indices {start_idx}-{start_idx + len(samples) - 1}: {e}. "
                    "Falling back to single-image inference."
                )
                scores, lowres_maps = [], []
                for sample in samples:
                    try:
                        score, lowres_map = model_wrapper.forward_raw(
                            sample["image"], category=category
                        )
                    except Exception as single_error:
                        logger.warning(
                            f"      Inference error on {sample['sample_id']}: "
                            f"{single_error}"
                        )
                        score = 0.0
                        lowres_map = np.zeros(
                            (
                                self.config.default_map_resolution,
                                self.config.default_map_resolution,
                            ),
                            dtype=np.float32,
                        )
                    scores.append(score)
                    lowres_maps.append(lowres_map)
                scores = np.asarray(scores, dtype=np.float32)
                lowres_maps = np.stack(lowres_maps)

            for local_idx, sample in enumerate(samples):
                mask_path = dataset.samples[start_idx + local_idx].get("mask_path")
                image_path = dataset.samples[start_idx + local_idx].get("image_path")
                gt_mask = (np.asarray(sample["mask"]) > 0).astype(np.float32)
                metric_masks.append(
                    resize_mask(
                        gt_mask,
                        DEFAULT_PIXEL_METRIC_SIZE,
                        DEFAULT_PIXEL_METRIC_SIZE,
                    )
                )
                accumulator.add(
                    score=float(scores[local_idx]),
                    lowres_map=lowres_maps[local_idx],
                    label=sample["label"],
                    sample_id=sample["sample_id"],
                    mask_path=str(mask_path) if mask_path else None,
                    image_path=str(image_path) if image_path else None,
                    selected_corruption=sample.get("selected_corruption"),
                )
            if progress_bar is not None:
                progress_bar.update(len(samples))

        # Flush and save
        scores, lowres_maps, metadata = accumulator.flush()

        self.storage.save_artifacts(
            model_name=model_name,
            dataset_name=dataset_name,
            category=category,
            noise_type=corruption_type,
            severity=severity,
            scores=scores,
            lowres_maps=lowres_maps,
            metadata=metadata,
        )

        # Compute metrics immediately
        labels = np.array([m["label"] for m in metadata])

        # Image-level metrics
        image_metrics = compute_image_metrics(labels, scores)

        # Pixel-level metrics
        if lowres_maps.ndim == 3 and lowres_maps.shape[0] > 0:
            metric_size = DEFAULT_PIXEL_METRIC_SIZE
            if len(metric_masks) != len(metadata):
                raise RuntimeError(
                    "Metric-mask count does not match inference metadata: "
                    f"{len(metric_masks)} masks for {len(metadata)} samples."
                )
            resized_masks = []
            resized_maps = []
            for i in range(len(metadata)):
                # Use the mask returned by the dataset, not the original mask
                # path. Geometric corruptions transform that in-memory mask
                # with the same deterministic parameters as the image.
                resized_masks.append(metric_masks[i])
                resized_maps.append(
                    resize_anomaly_map(lowres_maps[i], metric_size, metric_size)
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

        self.eval_harness.record_metrics(
            model_name=model_name,
            dataset_name=dataset_name,
            category=category,
            corruption_type=corruption_type,
            severity=severity,
            pixel_metrics=pixel_metrics,
            image_metrics=image_metrics,
        )


def run_evaluation(
    mvtec_root: Optional[str] = "/kaggle/input/mvtec-ad",
    visa_root: Optional[str] = "/kaggle/input/visa-anomaly-detection",
    output_root: str = "/kaggle/working/outputs",
    models: Optional[List[str]] = None,
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    device: str = "cuda",
    dataset: str = "both",
    corruption_types: Optional[List[str]] = None,
    severity_levels: Optional[List[int]] = None,
    batch_size: int = 1,
    corruption_cache_root: Optional[str] = None,
    corruption_cache_format: str = "png",
    categorized_corruptions: bool = False,
    categorized_corruption_plans: Optional[Dict[str, str]] = None,
    corruption_seed: Optional[int] = None,
) -> None:
    """
    Convenience function to run the full evaluation pipeline.

    Args:
        mvtec_root: Path to MVTec AD dataset root, or None to skip it.
        visa_root: Path to VisA dataset root, or None to skip it.
        output_root: Output directory for artifacts and CSVs.
        models: List of model names (None = completed models only).
        model_kwargs: Optional per-model constructor arguments, e.g.
            {"AnomalyCLIP": {"anomalyclip_root": "...", "checkpoint_path": "..."}}.
        device: PyTorch device.
        dataset: Dataset selector: "both", "mvtec", or "visa".
        corruption_types: Optional subset of Hendrycks corruptions to run.
        severity_levels: Optional subset of severity levels to run.
        batch_size: Number of images per model forward pass. Corruptions and
            image loading remain CPU-side, but compatible wrappers can use this
            to improve GPU utilization during inference.
        corruption_cache_root: Optional directory for corrupted image cache.
            This is useful on Kaggle when rerunning a model or evaluating
            multiple models against the same corruptions.
        corruption_cache_format: Cache encoding. Use "png" to avoid extra
            compression, or "jpeg" to mirror ImageNet-C-style file storage.
        categorized_corruptions: If true, treat ``corruption_types`` as the
            four grouped protocols and use a persistent assignment CSV.
        categorized_corruption_plans: Mapping from dataset name (``mvtec`` or
            ``visa``) to its CSV generated by demo_categorized.ipynb.
        corruption_seed: Per-image corruption seed.  Set this to the
            ``base_seed`` used to generate a persistent categorized plan.
    """
    from .dataset import build_dataset_configs

    dataset = _normalize_dataset_selector(dataset)

    resolved_mvtec_root = None
    if dataset in {"both", "mvtec"} and mvtec_root is not None:
        resolved_mvtec_root = _first_existing_root([
            mvtec_root,
            "/kaggle/input/mvtec-ad",
            "/kaggle/input/datasets/ipythonx/mvtec-ad",
        ])

    resolved_visa_root = None
    if dataset in {"both", "visa"} and visa_root is not None:
        resolved_visa_root = _first_existing_root([
            visa_root,
            "/kaggle/input/visa-anomaly-detection",
            "/kaggle/input/visa",
            "/kaggle/input/datasets/ess1004/visa-anomaly-detection",
        ])

    dataset_configs = build_dataset_configs(
        mvtec_root=resolved_mvtec_root,
        visa_root=resolved_visa_root,
    )
    if not dataset_configs:
        raise FileNotFoundError(
            f"No dataset roots found for dataset='{dataset}'. "
            "Check mvtec_root/visa_root or Kaggle input mounts."
        )

    config = HarnessConfig(
        output_root=Path(output_root),
        datasets=dataset_configs,
        device=device,
        models=models or COMPLETED_MODELS,
        model_kwargs=model_kwargs or {},
        corruption_types=corruption_types or CORRUPTION_TYPES,
        severity_levels=severity_levels or SEVERITY_LEVELS,
        batch_size=batch_size,
        corruption_cache_root=Path(corruption_cache_root) if corruption_cache_root else None,
        corruption_cache_format=corruption_cache_format,
        categorized_corruptions=categorized_corruptions,
        categorized_corruption_plans={
            name: Path(path)
            for name, path in (categorized_corruption_plans or {}).items()
        },
        corruption_seed=corruption_seed,
    )

    runner = RobustnessRunner(config)
    runner.run()
