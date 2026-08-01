"""Configuration for few-shot robustness evaluation."""

from dataclasses import dataclass, field
from typing import List

from zero_shot.harness.config import (  # re-export the shared protocol types
    CATEGORIZED_CORRUPTION_GROUPS,
    CATEGORIZED_CORRUPTION_TYPES,
    CLEAN_CONDITION,
    CORRUPTION_TYPES,
    DatasetConfig,
    GLOBAL_SEED,
    HarnessConfig,
    MVTEC_CATEGORIES,
    SEVERITY_LEVELS,
    VISA_CATEGORIES,
)


REGISTERED_MODELS: List[str] = ["INP-Former", "PromptAD"]
COMPLETED_MODELS: List[str] = ["INP-Former"]
OFFICIAL_SHOTS = (1, 2, 4)


@dataclass
class FewShotHarnessConfig(HarnessConfig):
    """Harness configuration with an explicit few-shot setting."""

    shot: int = 1
    models: List[str] = field(default_factory=lambda: COMPLETED_MODELS.copy())
    require_anomaly_masks: bool = True

    def __post_init__(self) -> None:
        if self.shot not in OFFICIAL_SHOTS:
            raise ValueError(
                f"shot must be one of the official settings {OFFICIAL_SHOTS}; "
                f"got {self.shot}."
            )
        super().__post_init__()


__all__ = [
    "CATEGORIZED_CORRUPTION_GROUPS",
    "CATEGORIZED_CORRUPTION_TYPES",
    "CLEAN_CONDITION",
    "COMPLETED_MODELS",
    "CORRUPTION_TYPES",
    "DatasetConfig",
    "FewShotHarnessConfig",
    "GLOBAL_SEED",
    "MVTEC_CATEGORIES",
    "OFFICIAL_SHOTS",
    "REGISTERED_MODELS",
    "SEVERITY_LEVELS",
    "VISA_CATEGORIES",
]
