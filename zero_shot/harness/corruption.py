"""Compatibility exports for the repository-level corruption pipeline."""

from shared.corruption import (
    CATEGORIZED_CORRUPTIONS,
    GEOMETRIC_CORRUPTIONS,
    apply_corruption,
    apply_corruption_to_mask,
    fixed_numpy_seed,
    is_corruption_category,
    stable_corruption_seed,
)


__all__ = [
    "CATEGORIZED_CORRUPTIONS",
    "GEOMETRIC_CORRUPTIONS",
    "apply_corruption",
    "apply_corruption_to_mask",
    "fixed_numpy_seed",
    "is_corruption_category",
    "stable_corruption_seed",
]
