# -*- coding: utf-8 -*-
"""Canonical ordering helpers for exported robustness conditions."""

import re
from typing import Tuple

from .config import CATEGORIZED_CORRUPTION_TYPES, CORRUPTION_TYPES


_CONDITION_PATTERN = re.compile(
    r"^(?P<corruption>.+)_level\s+(?P<severity>\d+)$"
)
_CORRUPTION_RANK = {
    corruption: rank
    for rank, corruption in enumerate(
        [*CORRUPTION_TYPES, *CATEGORIZED_CORRUPTION_TYPES]
    )
}


def parse_condition(condition: str) -> Tuple[str, int]:
    """Split ``<corruption>_level <severity>`` into its two components."""
    normalized = condition.strip()
    match = _CONDITION_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "Expected condition in '<corruption>_level <severity>' format; "
            f"got {condition!r}."
        )
    return match.group("corruption"), int(match.group("severity"))


def condition_sort_key(condition: str) -> Tuple[int, int, str, int]:
    """Return a stable protocol-aware sort key for a result condition.

    The clean baseline is first. Known corruptions then follow the configured
    concrete-corruption order followed by the configured categorized order,
    with severities ascending. Future/custom corruption names remain supported
    and are placed afterward in alphabetical order.
    """
    try:
        corruption, severity = parse_condition(condition)
    except ValueError:
        # Exporting a custom preformatted key should remain possible. The
        # normalization CLI validates stored CSV rows more strictly.
        return (3, 0, condition.strip(), 0)

    if corruption == "clean":
        return (0, 0, "", severity)

    rank = _CORRUPTION_RANK.get(corruption)
    if rank is not None:
        return (1, rank, "", severity)

    return (2, 0, corruption, severity)
