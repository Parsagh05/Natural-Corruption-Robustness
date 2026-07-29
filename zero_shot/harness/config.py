# -*- coding: utf-8 -*-
"""
config.py - Central configuration for the robustness evaluation harness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.corruption import CATEGORIZED_CORRUPTIONS


# Global seed
GLOBAL_SEED: int = 111

# Corruption config
CORRUPTION_TYPES: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "brightness",
    "contrast",
]

# Grouped protocol: every image receives one concrete corruption sampled from
# its group.  The assignment is balanced independently for each object class
# and image label, so the corruption choice cannot be confounded with the
# normal/anomalous split.
CATEGORIZED_CORRUPTION_GROUPS: Dict[str, Tuple[str, ...]] = (
    CATEGORIZED_CORRUPTIONS.copy()
)
CATEGORIZED_CORRUPTION_TYPES: List[str] = list(
    CATEGORIZED_CORRUPTION_GROUPS
)
CATEGORIZED_FINE_GRAINED_CORRUPTION_TYPES: List[str] = [
    corruption
    for corruptions in CATEGORIZED_CORRUPTION_GROUPS.values()
    for corruption in corruptions
]

SEVERITY_LEVELS: List[int] = [1, 2, 3, 4]

# The optional baseline evaluates unmodified source images. Keeping this
# separate prevents severity 0 from reaching corruption implementations,
# which only support severities 1-5.
CLEAN_CONDITION: Tuple[str, int] = ("clean", 0)

# Model registry
LEARNABLE_MODELS: List[str] = [
    "VCP-CLIP",
    "Crane",
    "FAPrompt",
    "AnomalyCLIP",
    "AdaCLIP",
    "AA-CLIP",
    "Bayes-PFL",
    "AF-CLIP",
    "CoPS",
]

TRAINING_FREE_MODELS: List[str] = [
    "WinCLIP",
    "AnoVL",
    "MRAD",
    "AnomalyAgent",
]

ALL_MODELS: List[str] = LEARNABLE_MODELS + TRAINING_FREE_MODELS

# Only these wrappers have executable model-specific inference implemented.
# The remaining registered models are kept as placeholders for future work.
COMPLETED_MODELS: List[str] = ["AnomalyCLIP", "AA-CLIP", "AF-CLIP"]

# Dataset config
MVTEC_CATEGORIES: List[str] = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

VISA_CATEGORIES: List[str] = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


@dataclass
class DatasetConfig:
    """Configuration for a single AD dataset."""

    name: str
    root_path: Path
    categories: List[str]
    test_subdir: str = "test"
    mask_subdir: str = "ground_truth"
    layout: str = "auto"


@dataclass
class HarnessConfig:
    """Top-level configuration for the evaluation harness."""

    seed: int = GLOBAL_SEED
    corruption_seed: Optional[int] = None
    corruption_types: List[str] = field(default_factory=lambda: CORRUPTION_TYPES)
    severity_levels: List[int] = field(default_factory=lambda: SEVERITY_LEVELS)
    include_clean: bool = True
    models: List[str] = field(default_factory=lambda: COMPLETED_MODELS)
    output_root: Path = Path("outputs")
    datasets: List[DatasetConfig] = field(default_factory=list)
    model_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    batch_size: int = 1
    corruption_cache_root: Optional[Path] = None
    corruption_cache_format: str = "png"
    categorized_corruptions: bool = False
    categorized_corruption_plans: Dict[str, Path] = field(default_factory=dict)
    device: str = "cuda"
    default_map_resolution: int = 14

    @property
    def evaluation_conditions(self) -> List[Tuple[str, int]]:
        """Return the enabled clean baseline and corrupted conditions."""
        conditions = [CLEAN_CONDITION] if self.include_clean else []
        conditions.extend(
            (corruption_type, severity)
            for corruption_type in self.corruption_types
            if corruption_type != CLEAN_CONDITION[0]
            for severity in self.severity_levels
        )
        return conditions

    def __post_init__(self):
        if not self.evaluation_conditions:
            raise ValueError(
                "At least one evaluation condition is required: enable the "
                "clean baseline or provide a corruption type and severity."
            )
        if self.categorized_corruptions:
            invalid = [
                name for name in self.corruption_types
                if name not in CATEGORIZED_CORRUPTION_TYPES
            ]
            if invalid:
                raise ValueError(
                    "categorized_corruptions=True requires only categorized "
                    "corruption types "
                    f"{CATEGORIZED_CORRUPTION_TYPES}; got invalid values {invalid}"
                )
            normalized_plans = {
                str(dataset_name).lower(): Path(plan_path)
                for dataset_name, plan_path in self.categorized_corruption_plans.items()
            }
            required_datasets = {
                dataset.name.lower() for dataset in self.datasets
            }
            missing_plans = sorted(required_datasets - set(normalized_plans))
            if missing_plans:
                raise ValueError(
                    "categorized_corruptions=True requires one persistent "
                    "corruption CSV per dataset; missing plans for "
                    f"{missing_plans}"
                )
            missing_files = [
                str(normalized_plans[name]) for name in required_datasets
                if not normalized_plans[name].is_file()
            ]
            if missing_files:
                raise FileNotFoundError(
                    "Persistent corruption CSV not found: "
                    + ", ".join(missing_files)
                )
            self.categorized_corruption_plans = normalized_plans
        self.output_root.mkdir(parents=True, exist_ok=True)
