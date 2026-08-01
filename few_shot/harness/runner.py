"""Few-shot robustness execution built on the zero-shot logging/output contract."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from shared import corruption_plan_path
from zero_shot.harness.dataset import build_dataset_configs
from zero_shot.harness.runner import (
    RobustnessRunner as CommonRobustnessRunner,
    _first_existing_root,
    _normalize_dataset_selector,
    logger as pipeline_logger,
)
from zero_shot.harness.storage import ArtifactStorage

from .config import (
    CATEGORIZED_CORRUPTION_TYPES,
    COMPLETED_MODELS,
    CORRUPTION_TYPES,
    FewShotHarnessConfig,
    OFFICIAL_SHOTS,
    SEVERITY_LEVELS,
)
from .models import (
    PROMPTAD_DATASET_CATEGORIES,
    get_model,
    normalize_dataset_name,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


class RobustnessRunner(CommonRobustnessRunner):
    """Common corruption runner with a few-shot model factory and provenance."""

    config: FewShotHarnessConfig

    def _create_model(self, model_name: str):
        kwargs = dict(self.config.model_kwargs.get(model_name, {}))
        configured_shot = kwargs.setdefault("shot", self.config.shot)
        if int(configured_shot) != self.config.shot:
            raise ValueError(
                f"Harness shot={self.config.shot} conflicts with "
                f"{model_name} shot={configured_shot}."
            )
        return get_model(model_name, device=self.config.device, **kwargs)

    def _run_single_model(self, model_name, datasets) -> None:
        pipeline_logger.info("%s", "=" * 60)
        pipeline_logger.info("Few-shot model: %s | shot=%s", model_name, self.config.shot)
        pipeline_logger.info("%s", "=" * 60)

        model_wrapper = self._create_model(model_name)
        result_name = model_wrapper.model_name
        model_output_dir = self.config.output_root / result_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_output_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        pipeline_logger.addHandler(file_handler)
        pipeline_logger.info(
            "Logging %s (%s-shot) to %s", model_name, self.config.shot, log_path
        )

        manifest_path = model_output_dir / "run_manifest.json"
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "status": "starting",
            "started_at_utc": _utc_timestamp(),
            "requested_model": model_name,
            "result_name": result_name,
            "shot": self.config.shot,
            "device": self.config.device,
            "seed": self.config.seed,
            "corruption_seed": (
                self.config.seed
                if self.config.corruption_seed is None
                else self.config.corruption_seed
            ),
            "categorized_corruptions": self.config.categorized_corruptions,
            "include_clean": self.config.include_clean,
            "conditions": [
                {"corruption": corruption, "severity": severity}
                for corruption, severity in self.config.evaluation_conditions
            ],
            "batch_size": self.config.batch_size,
            "corruption_cache_root": (
                str(self.config.corruption_cache_root)
                if self.config.corruption_cache_root else None
            ),
            "corruption_cache_format": self.config.corruption_cache_format,
            "evaluations": [],
        }
        _write_manifest(manifest_path, manifest)

        succeeded = False
        try:
            model_wrapper.load_model()
            manifest["status"] = "running"
            for dataset_config in datasets:
                evaluation_start = time.time()
                pipeline_logger.info(
                    "Activating official %s-shot checkpoint for %s",
                    self.config.shot,
                    dataset_config.name,
                )
                model_wrapper.prepare_for_dataset(dataset_config.name)
                provenance = model_wrapper.inference_provenance()
                pipeline_logger.info(
                    "Checkpoint: %s",
                    provenance.get("checkpoint", {}).get("path"),
                )

                self._run_model_on_dataset(model_wrapper, dataset_config)
                px_path, sp_path = self.eval_harness.export_csv(
                    result_name, dataset_config
                )
                outputs = {
                    "px_csv": (
                        px_path.relative_to(model_output_dir).as_posix()
                        if px_path else None
                    ),
                    "sp_csv": (
                        sp_path.relative_to(model_output_dir).as_posix()
                        if sp_path else None
                    ),
                }
                if self.config.categorized_corruptions:
                    fine_px, fine_sp, per_image = (
                        self.eval_harness.export_categorized_fine_grained(
                            result_name, dataset_config
                        )
                    )
                    outputs.update({
                        "fine_grained_px_csv": (
                            fine_px.relative_to(model_output_dir).as_posix()
                            if fine_px else None
                        ),
                        "fine_grained_sp_csv": (
                            fine_sp.relative_to(model_output_dir).as_posix()
                            if fine_sp else None
                        ),
                        "fine_grained_per_image_json": (
                            per_image.relative_to(model_output_dir).as_posix()
                            if per_image else None
                        ),
                    })
                evaluation_record = {
                    "dataset": dataset_config.name,
                    "dataset_root": str(dataset_config.root_path),
                    "category_count": len(dataset_config.categories),
                    "elapsed_seconds": round(time.time() - evaluation_start, 3),
                    "model_provenance": provenance,
                    "outputs": outputs,
                }
                manifest["evaluations"].append(evaluation_record)
                _write_manifest(manifest_path, manifest)
                pipeline_logger.info(
                    "Completed %s-shot %s evaluation in %.1fs",
                    self.config.shot,
                    dataset_config.name,
                    evaluation_record["elapsed_seconds"],
                )

            manifest["status"] = "evaluation_completed"
            manifest["completed_at_utc"] = _utc_timestamp()
            manifest["archive"] = f"{result_name}_artifacts.zip"
            _write_manifest(manifest_path, manifest)
            succeeded = True
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failed_at_utc"] = _utc_timestamp()
            manifest["error_type"] = type(exc).__name__
            manifest["error"] = str(exc)
            _write_manifest(manifest_path, manifest)
            pipeline_logger.exception(
                "Few-shot evaluation failed for %s", result_name
            )
            raise
        finally:
            model_wrapper.release()
            pipeline_logger.removeHandler(file_handler)
            file_handler.close()

        if succeeded:
            zip_path = self.storage.zip_and_cleanup_model(result_name)
            if zip_path:
                pipeline_logger.info("Archived: %s", zip_path)
            self.eval_harness.discard_model_results(result_name)
            ArtifactStorage.cleanup_memory()


def run_evaluation(
    mvtec_root: Optional[str] = "/kaggle/input/mvtec-ad",
    visa_root: Optional[str] = "/kaggle/input/visa-anomaly-detection",
    output_root: str = "/kaggle/working/outputs",
    models: Optional[List[str]] = None,
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    shot: int = 1,
    device: str = "cuda",
    dataset: str = "both",
    corruption_types: Optional[List[str]] = None,
    severity_levels: Optional[List[int]] = None,
    batch_size: int = 1,
    corruption_cache_root: Optional[str] = None,
    corruption_cache_format: str = "png",
    categorized_corruptions: bool = True,
    categorized_corruption_plans: Optional[Dict[str, str]] = None,
    corruption_seed: Optional[int] = 123,
    include_clean: bool = True,
) -> None:
    """Run one shot setting on one or both supported datasets."""
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
            f"No dataset roots found for dataset={dataset!r}. Check the "
            "MVTec AD and VisA paths."
        )

    if categorized_corruption_plans is None and categorized_corruptions:
        categorized_corruption_plans = {
            config.name.lower(): str(
                corruption_plan_path(normalize_dataset_name(config.name))
            )
            for config in dataset_configs
        }

    config = FewShotHarnessConfig(
        shot=shot,
        output_root=Path(output_root),
        datasets=dataset_configs,
        device=device,
        models=models or COMPLETED_MODELS,
        model_kwargs=model_kwargs or {},
        corruption_types=(
            corruption_types
            if corruption_types is not None
            else (
                CATEGORIZED_CORRUPTION_TYPES
                if categorized_corruptions else CORRUPTION_TYPES
            )
        ),
        severity_levels=(
            severity_levels if severity_levels is not None else SEVERITY_LEVELS
        ),
        include_clean=include_clean,
        batch_size=batch_size,
        corruption_cache_root=(
            Path(corruption_cache_root) if corruption_cache_root else None
        ),
        corruption_cache_format=corruption_cache_format,
        categorized_corruptions=categorized_corruptions,
        categorized_corruption_plans={
            name: Path(path)
            for name, path in (categorized_corruption_plans or {}).items()
        },
        corruption_seed=corruption_seed,
    )
    RobustnessRunner(config).run()


def run_official_evaluations(
    mvtec_root: Optional[str],
    visa_root: Optional[str],
    output_root: str,
    inpformer_root: str,
    checkpoint_paths: Mapping[int, Mapping[str, str]],
    shots: Sequence[int] = OFFICIAL_SHOTS,
    datasets: Sequence[str] = ("mvtec", "visa"),
    device: str = "cuda",
    batch_size: int = 4,
    corruption_types: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[int]] = None,
    categorized_corruptions: bool = True,
    corruption_cache_root: Optional[str] = None,
    corruption_cache_format: str = "png",
    corruption_seed: int = 123,
    include_clean: bool = True,
    strict_source_commit: bool = False,
) -> None:
    """Execute each requested shot for the selected evaluation datasets.

    Corruption mode, enabled corruption names, severities, and the clean
    baseline are explicit controls. Categorized mode uses the repository's
    fixed per-image assignment plans.
    """
    dataset_values = (datasets,) if isinstance(datasets, str) else datasets
    selected_datasets = tuple(
        normalize_dataset_name(dataset) for dataset in dataset_values
    )
    if not selected_datasets:
        raise ValueError("Select at least one evaluation dataset.")
    if len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError(
            f"Evaluation datasets must be unique; got {selected_datasets}."
        )
    root_values = {"mvtec": mvtec_root, "visa": visa_root}
    dataset_labels = {"mvtec": "MVTec AD", "visa": "VisA"}
    dataset_roots = {
        dataset: (
            Path(root_values[dataset]).expanduser()
            if root_values[dataset] is not None else None
        )
        for dataset in selected_datasets
    }
    missing_roots = [
        f"{dataset_labels[name]}: {path}"
        for name, path in dataset_roots.items()
        if path is None or not path.is_dir()
    ]
    if missing_roots:
        raise FileNotFoundError(
            "The selected official evaluation datasets are missing:\n  - "
            + "\n  - ".join(missing_roots)
        )
    source_root = Path(inpformer_root).expanduser()
    if not (
        (source_root / "models" / "uad.py").is_file()
        and (source_root / "dinov2" / "models" / "vision_transformer.py").is_file()
    ):
        raise FileNotFoundError(
            f"Official INP-Former source tree is incomplete: {source_root}"
        )

    selected_shots = tuple(int(shot) for shot in shots)
    if not selected_shots:
        raise ValueError("At least one shot setting must be selected.")
    if len(set(selected_shots)) != len(selected_shots):
        raise ValueError(f"Shot settings must be unique; got {selected_shots}.")
    invalid_shots = sorted(set(selected_shots) - set(OFFICIAL_SHOTS))
    if invalid_shots:
        raise ValueError(
            f"Only official shot settings {OFFICIAL_SHOTS} are supported; "
            f"got {invalid_shots}."
        )

    allowed_corruptions = (
        CATEGORIZED_CORRUPTION_TYPES
        if categorized_corruptions else CORRUPTION_TYPES
    )
    selected_corruptions = list(
        corruption_types
        if corruption_types is not None else allowed_corruptions
    )
    if len(set(selected_corruptions)) != len(selected_corruptions):
        raise ValueError(
            f"Corruption types must be unique; got {selected_corruptions}."
        )
    invalid_corruptions = sorted(
        set(selected_corruptions) - set(allowed_corruptions)
    )
    if invalid_corruptions:
        mode = "categorized" if categorized_corruptions else "uncategorized"
        raise ValueError(
            f"Invalid {mode} corruption types {invalid_corruptions}; "
            f"choose from {allowed_corruptions}."
        )

    selected_severities = [
        int(level)
        for level in (
            severity_levels if severity_levels is not None else SEVERITY_LEVELS
        )
    ]
    if len(set(selected_severities)) != len(selected_severities):
        raise ValueError(
            f"Severity levels must be unique; got {selected_severities}."
        )
    invalid_severities = sorted(set(selected_severities) - set(SEVERITY_LEVELS))
    if invalid_severities:
        raise ValueError(
            f"This benchmark supports severity levels {SEVERITY_LEVELS}; "
            f"got {invalid_severities}."
        )
    if selected_corruptions and not selected_severities:
        raise ValueError(
            "Select at least one severity when corruptions are enabled."
        )
    if not include_clean and not selected_corruptions:
        raise ValueError(
            "No evaluation conditions selected: enable clean or a corruption."
        )

    normalized_paths: Dict[int, Dict[str, str]] = {}
    for shot in selected_shots:
        raw_mapping = checkpoint_paths.get(shot)
        if raw_mapping is None:
            raw_mapping = checkpoint_paths.get(str(shot))  # type: ignore[arg-type]
        if raw_mapping is None:
            raise KeyError(f"Missing official {shot}-shot checkpoint mapping.")
        available_paths = {
            normalize_dataset_name(name): str(path)
            for name, path in raw_mapping.items()
        }
        missing_datasets = set(selected_datasets) - set(available_paths)
        if missing_datasets:
            raise KeyError(
                f"{shot}-shot checkpoint mapping is missing {missing_datasets}."
            )
        normalized_paths[shot] = {
            dataset: available_paths[dataset]
            for dataset in selected_datasets
        }

    missing_checkpoints = [
        f"{shot}-shot {dataset}: {path}"
        for shot, dataset_paths in normalized_paths.items()
        for dataset, path in dataset_paths.items()
        if not Path(path).expanduser().exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Official checkpoint preflight failed. Missing:\n  - "
            + "\n  - ".join(missing_checkpoints)
        )

    plans = (
        {
            dataset: str(corruption_plan_path(dataset))
            for dataset in selected_datasets
        }
        if categorized_corruptions else None
    )
    for shot_index, shot in enumerate(selected_shots, start=1):
        pipeline_logger.info(
            "Launching official INP-Former suite %s/%s: %s-shot x %s",
            shot_index,
            len(selected_shots),
            shot,
            [dataset_labels[name] for name in selected_datasets],
        )
        run_evaluation(
            mvtec_root=(mvtec_root if "mvtec" in selected_datasets else None),
            visa_root=(visa_root if "visa" in selected_datasets else None),
            output_root=output_root,
            models=["INP-Former"],
            model_kwargs={
                "INP-Former": {
                    "inpformer_root": inpformer_root,
                    "checkpoint_paths": normalized_paths[shot],
                    "strict_source_commit": strict_source_commit,
                }
            },
            shot=shot,
            device=device,
            dataset=(
                "both" if len(selected_datasets) == 2
                else selected_datasets[0]
            ),
            corruption_types=selected_corruptions,
            severity_levels=selected_severities,
            batch_size=batch_size,
            corruption_cache_root=corruption_cache_root,
            corruption_cache_format=corruption_cache_format,
            categorized_corruptions=categorized_corruptions,
            categorized_corruption_plans=plans,
            corruption_seed=corruption_seed,
            include_clean=include_clean,
        )


def run_promptad_evaluations(
    mvtec_root: Optional[str],
    visa_root: Optional[str],
    output_root: str,
    promptad_root: str,
    checkpoint_paths: Mapping[
        int, Mapping[str, Mapping[str, Mapping[str, str]]]
    ],
    shots: Sequence[int] = OFFICIAL_SHOTS,
    datasets: Sequence[str] = ("mvtec", "visa"),
    device: str = "cuda",
    batch_size: int = 32,
    corruption_types: Optional[Sequence[str]] = None,
    severity_levels: Optional[Sequence[int]] = None,
    categorized_corruptions: bool = True,
    corruption_cache_root: Optional[str] = None,
    corruption_cache_format: str = "png",
    corruption_seed: int = 123,
    include_clean: bool = True,
    strict_source_commit: bool = False,
) -> None:
    """Run class-paired PromptAD CLS/SEG buffers for selected shot suites."""
    dataset_values = (datasets,) if isinstance(datasets, str) else datasets
    selected_datasets = tuple(
        normalize_dataset_name(dataset) for dataset in dataset_values
    )
    if not selected_datasets or len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError(
            f"PromptAD evaluation datasets must be non-empty and unique: {selected_datasets}."
        )

    root_values = {"mvtec": mvtec_root, "visa": visa_root}
    dataset_labels = {"mvtec": "MVTec AD", "visa": "VisA"}
    missing_roots = [
        f"{dataset_labels[name]}: {root_values[name]}"
        for name in selected_datasets
        if root_values[name] is None
        or not Path(str(root_values[name])).expanduser().is_dir()
    ]
    if missing_roots:
        raise FileNotFoundError(
            "The selected PromptAD evaluation datasets are missing:\n  - "
            + "\n  - ".join(missing_roots)
        )

    source_root = Path(promptad_root).expanduser()
    if not (
        (source_root / "PromptAD" / "model.py").is_file()
        and (source_root / "PromptAD" / "CLIPAD" / "factory.py").is_file()
    ):
        raise FileNotFoundError(
            f"Official PromptAD source tree is incomplete: {source_root}"
        )

    selected_shots = tuple(int(shot) for shot in shots)
    if not selected_shots or len(set(selected_shots)) != len(selected_shots):
        raise ValueError(
            f"PromptAD shot settings must be non-empty and unique: {selected_shots}."
        )
    invalid_shots = sorted(set(selected_shots) - set(OFFICIAL_SHOTS))
    if invalid_shots:
        raise ValueError(
            f"Only PromptAD shot settings {OFFICIAL_SHOTS} are supported; "
            f"got {invalid_shots}."
        )

    allowed_corruptions = (
        CATEGORIZED_CORRUPTION_TYPES if categorized_corruptions else CORRUPTION_TYPES
    )
    selected_corruptions = list(
        corruption_types if corruption_types is not None else allowed_corruptions
    )
    if len(set(selected_corruptions)) != len(selected_corruptions):
        raise ValueError(f"Corruption types must be unique: {selected_corruptions}.")
    invalid_corruptions = sorted(set(selected_corruptions) - set(allowed_corruptions))
    if invalid_corruptions:
        raise ValueError(
            f"Invalid PromptAD corruption types {invalid_corruptions}; "
            f"choose from {allowed_corruptions}."
        )

    selected_severities = [
        int(level)
        for level in (
            severity_levels if severity_levels is not None else SEVERITY_LEVELS
        )
    ]
    if len(set(selected_severities)) != len(selected_severities):
        raise ValueError(f"Severity levels must be unique: {selected_severities}.")
    invalid_severities = sorted(set(selected_severities) - set(SEVERITY_LEVELS))
    if invalid_severities:
        raise ValueError(
            f"This benchmark supports severity levels {SEVERITY_LEVELS}; "
            f"got {invalid_severities}."
        )
    if selected_corruptions and not selected_severities:
        raise ValueError("Select at least one severity when corruptions are enabled.")
    if not include_clean and not selected_corruptions:
        raise ValueError("No PromptAD evaluation conditions were selected.")

    normalized_paths: Dict[
        int, Dict[str, Dict[str, Dict[str, str]]]
    ] = {}
    for shot in selected_shots:
        raw_shot_mapping = checkpoint_paths.get(shot)
        if raw_shot_mapping is None:
            raw_shot_mapping = checkpoint_paths.get(str(shot))  # type: ignore[arg-type]
        if raw_shot_mapping is None:
            raise KeyError(f"Missing PromptAD {shot}-shot checkpoint mapping.")
        available_datasets = {
            normalize_dataset_name(name): categories
            for name, categories in raw_shot_mapping.items()
        }
        normalized_paths[shot] = {}
        for dataset in selected_datasets:
            if dataset not in available_datasets:
                raise KeyError(
                    f"PromptAD {shot}-shot mapping is missing dataset {dataset}."
                )
            category_mapping: Dict[str, Dict[str, str]] = {}
            for category in PROMPTAD_DATASET_CATEGORIES[dataset]:
                raw_tasks = available_datasets[dataset].get(category)
                if raw_tasks is None:
                    raise KeyError(
                        f"PromptAD {dataset}/{shot}-shot is missing {category}."
                    )
                tasks = {str(task).lower(): str(path) for task, path in raw_tasks.items()}
                if set(tasks) != {"cls", "seg"}:
                    raise KeyError(
                        f"PromptAD {dataset}/{shot}-shot/{category} needs exactly "
                        f"CLS and SEG checkpoints; got {sorted(tasks)}."
                    )
                missing_files = [path for path in tasks.values() if not Path(path).is_file()]
                if missing_files:
                    raise FileNotFoundError(
                        "PromptAD checkpoint preflight failed. Missing:\n  - "
                        + "\n  - ".join(missing_files)
                    )
                category_mapping[category] = tasks
            normalized_paths[shot][dataset] = category_mapping

    plans = (
        {
            dataset: str(corruption_plan_path(dataset))
            for dataset in selected_datasets
        }
        if categorized_corruptions else None
    )
    for shot_index, shot in enumerate(selected_shots, start=1):
        pipeline_logger.info(
            "Launching PromptAD suite %s/%s: %s-shot x %s",
            shot_index,
            len(selected_shots),
            shot,
            [dataset_labels[name] for name in selected_datasets],
        )
        run_evaluation(
            mvtec_root=(mvtec_root if "mvtec" in selected_datasets else None),
            visa_root=(visa_root if "visa" in selected_datasets else None),
            output_root=output_root,
            models=["PromptAD"],
            model_kwargs={
                "PromptAD": {
                    "promptad_root": promptad_root,
                    "checkpoint_paths": normalized_paths[shot],
                    "strict_source_commit": strict_source_commit,
                }
            },
            shot=shot,
            device=device,
            dataset=(
                "both" if len(selected_datasets) == 2 else selected_datasets[0]
            ),
            corruption_types=selected_corruptions,
            severity_levels=selected_severities,
            batch_size=batch_size,
            corruption_cache_root=corruption_cache_root,
            corruption_cache_format=corruption_cache_format,
            categorized_corruptions=categorized_corruptions,
            categorized_corruption_plans=plans,
            corruption_seed=corruption_seed,
            include_clean=include_clean,
        )


__all__ = [
    "RobustnessRunner",
    "run_evaluation",
    "run_official_evaluations",
    "run_promptad_evaluations",
]
