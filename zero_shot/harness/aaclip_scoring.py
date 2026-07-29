# -*- coding: utf-8 -*-
"""Paper-faithful AA-CLIP condition-level inference scoring helpers."""

from typing import List, Sequence, Tuple

import numpy as np


def _official_minmax(values: np.ndarray) -> np.ndarray:
    """Match AA-CLIP's test-time min-max normalization safely.

    The official implementation normalizes whenever the maximum is not
    exactly one.  A constant non-one input would make its denominator zero;
    returning zeros is the finite equivalent and preserves the absence of any
    ranking information.
    """
    normalized = np.asarray(values, dtype=np.float32).copy()
    if normalized.size == 0:
        return normalized
    if not np.isfinite(normalized).all():
        raise ValueError("AA-CLIP scores must be finite before normalization.")

    maximum = float(normalized.max())
    if maximum == 1.0:
        return normalized
    minimum = float(normalized.min())
    value_range = maximum - minimum
    if value_range <= 0.0:
        normalized.fill(0.0)
        return normalized
    normalized -= minimum
    normalized /= value_range
    return normalized


def normalize_aaclip_maps(
    anomaly_maps: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Apply AA-CLIP's global per-class map normalization in place."""
    maps = [np.asarray(anomaly_map, dtype=np.float32) for anomaly_map in anomaly_maps]
    if not maps:
        return maps
    if any(anomaly_map.ndim != 2 or anomaly_map.size == 0 for anomaly_map in maps):
        raise ValueError("AA-CLIP anomaly maps must be non-empty 2D arrays.")
    if any(not np.isfinite(anomaly_map).all() for anomaly_map in maps):
        raise ValueError("AA-CLIP anomaly maps must be finite before normalization.")

    maximum = max(float(anomaly_map.max()) for anomaly_map in maps)
    if maximum == 1.0:
        return maps
    minimum = min(float(anomaly_map.min()) for anomaly_map in maps)
    value_range = maximum - minimum
    if value_range <= 0.0:
        for anomaly_map in maps:
            anomaly_map.fill(0.0)
        return maps
    for anomaly_map in maps:
        anomaly_map -= minimum
        anomaly_map /= value_range
    return maps


def postprocess_aaclip_industrial_condition(
    image_scores: np.ndarray,
    anomaly_maps: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Normalize and fuse AA-CLIP industrial image/map scores.

    This implements ``metrics_eval`` from the official AA-CLIP repository for
    industrial datasets: normalize image and pixel predictions independently,
    take each image's maximum normalized pixel score, and average it 50/50
    with the normalized image prediction.
    """
    scores = np.asarray(image_scores, dtype=np.float32)
    if scores.ndim != 1:
        raise ValueError(f"AA-CLIP image scores must be 1D, got {scores.shape}.")
    maps = normalize_aaclip_maps(anomaly_maps)
    if len(scores) != len(maps):
        raise ValueError(
            "AA-CLIP score/map count mismatch: "
            f"{len(scores)} scores and {len(maps)} maps."
        )

    normalized_scores = _official_minmax(scores)
    map_max_scores = np.asarray(
        [float(anomaly_map.max()) for anomaly_map in maps], dtype=np.float32
    )
    fused_scores = 0.5 * map_max_scores + 0.5 * normalized_scores
    return fused_scores.astype(np.float32, copy=False), maps
