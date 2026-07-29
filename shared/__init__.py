"""Assets shared by zero-shot and few-shot evaluation pipelines."""

from pathlib import Path


SHARED_ROOT = Path(__file__).resolve().parent
CORRUPTION_PLAN_ROOT = SHARED_ROOT / "corruption_plans"


def corruption_plan_path(dataset: str) -> Path:
    """Return the persistent categorized-corruption plan for a dataset."""
    dataset_name = dataset.lower().strip()
    if dataset_name not in {"mvtec", "visa"}:
        raise ValueError("dataset must be either 'mvtec' or 'visa'")
    return CORRUPTION_PLAN_ROOT / f"{dataset_name}_corruption_plan.csv"


__all__ = [
    "CORRUPTION_PLAN_ROOT",
    "SHARED_ROOT",
    "corruption_plan_path",
]
