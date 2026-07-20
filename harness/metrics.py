# -*- coding: utf-8 -*-
"""
metrics.py - Evaluation metrics for pixel-level and image-level anomaly detection.

Pixel-level: AUROC, F1-Max, AUPRO
Image-level: AUROC, F1-Max, AP
"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)
from skimage import measure

# Numerical stability constant for F1 computation
_F1_EPSILON = 1e-10
DEFAULT_AUPRO_FPR_LIMIT = 0.3
DEFAULT_AUPRO_MAX_STEP = 200
DEFAULT_PIXEL_METRIC_SIZE = 518

ArrayLikeImages = Union[np.ndarray, Sequence[np.ndarray]]


def compute_f1_max(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute the maximum F1 score and its best threshold."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + _F1_EPSILON)
    if thresholds.size == 0:
        return float(np.max(f1_scores)), 0.0

    threshold_f1_scores = f1_scores[:-1]
    best_idx = int(np.argmax(threshold_f1_scores))
    return float(threshold_f1_scores[best_idx]), float(thresholds[best_idx])


def compute_aupro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    fpr_limit: float = DEFAULT_AUPRO_FPR_LIMIT,
    max_step: int = DEFAULT_AUPRO_MAX_STEP,
    device: str = "auto",
) -> float:
    """
    Compute AUPRO using the FasterAUPRO/Gudovskiy-style threshold sweep.

    Args:
        masks: Binary ground truth masks, shape (N, H, W).
        anomaly_maps: Predicted anomaly maps, shape (N, H, W).
        fpr_limit: FPR integration limit. FasterAUPRO default is 0.3.
        max_step: Number of threshold steps. FasterAUPRO default is 200.
        device: "auto", "cuda", or "cpu". Uses CUDA when available.

    Returns:
        AUPRO score in [0, 1] under the requested FPR limit.
    """
    masks = np.asarray(masks, dtype=np.float32)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)

    if masks.shape != anomaly_maps.shape:
        raise ValueError(
            f"masks and anomaly_maps must have the same shape, got "
            f"{masks.shape} and {anomaly_maps.shape}"
        )
    if masks.ndim != 3 or masks.shape[0] == 0:
        return 0.0

    masks = (masks > 0).astype(np.float32)
    min_th = float(anomaly_maps.min())
    max_th = float(anomaly_maps.max())
    delta = (max_th - min_th) / max_step

    if delta <= 0:
        return 0.0

    regionprops_list = [measure.regionprops(measure.label(mask)) for mask in masks]
    coords_list: List[List[Tuple[np.ndarray, np.ndarray, int]]] = [
        [
            (region.coords[:, 0], region.coords[:, 1], len(region.coords))
            for region in regionprops
        ]
        for regionprops in regionprops_list
    ]
    if not any(coords_list):
        return 0.0

    device_name = str(device).lower()
    use_cuda = device_name.startswith("cuda") or (
        device_name == "auto" and torch.cuda.is_available()
    )
    torch_device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    if torch_device.type == "cuda":
        coords_list = [
            [
                (
                    torch.as_tensor(rows, dtype=torch.long, device=torch_device),
                    torch.as_tensor(cols, dtype=torch.long, device=torch_device),
                    region_area,
                )
                for rows, cols, region_area in regions_coords
            ]
            for regions_coords in coords_list
        ]

    mask_tensor = torch.as_tensor(masks, dtype=torch.float32, device=torch_device)
    amap_tensor = torch.as_tensor(anomaly_maps, dtype=torch.float32, device=torch_device)
    inverse_masks = 1.0 - mask_tensor
    tn_pixel = float(inverse_masks.sum().item())
    if tn_pixel <= 0:
        return 0.0

    pros, fprs = [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps = amap_tensor > float(th)
        pro_values = []

        for image_idx, regions_coords in enumerate(coords_list):
            binary_amap = binary_amaps[image_idx]
            for rows, cols, region_area in regions_coords:
                tp_pixels = binary_amap[rows, cols].sum().item()
                pro_values.append(tp_pixels / region_area)

        fp_pixels = torch.logical_and(inverse_masks.bool(), binary_amaps).sum().item()
        fprs.append(fp_pixels / tn_pixel)
        pros.append(float(np.mean(pro_values)) if pro_values else 0.0)

    pros = np.asarray(pros, dtype=np.float64)
    fprs = np.asarray(fprs, dtype=np.float64)
    valid = fprs < fpr_limit
    fprs = fprs[valid]
    pros = pros[valid]
    if fprs.size == 0:
        return 0.0

    if np.ptp(fprs) == 0:
        return float(np.mean(pros))

    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    return float(auc(fprs, pros))


def _as_image_list(images: ArrayLikeImages) -> List[np.ndarray]:
    if isinstance(images, np.ndarray) and images.ndim == 3:
        return [images[i] for i in range(images.shape[0])]
    return [np.asarray(image) for image in images]


def compute_pixel_metrics(
    masks: ArrayLikeImages,
    anomaly_maps: ArrayLikeImages,
    aupro_device: str = "auto",
) -> Dict[str, float]:
    """
    Compute pixel-level metrics.

    Args:
        masks: Ground-truth binary masks, shape (N, H, W) or list of (H, W).
        anomaly_maps: Predicted anomaly maps, same shape as masks.
        aupro_device: "auto", "cuda", or "cpu" for FasterAUPRO.

    Returns:
        Dict with keys: auroc_px, f1_px, aupro_px, threshold_px
    """
    mask_list = _as_image_list(masks)
    map_list = _as_image_list(anomaly_maps)
    if len(mask_list) != len(map_list):
        raise ValueError(
            f"masks and anomaly_maps length mismatch: "
            f"{len(mask_list)} vs {len(map_list)}"
        )

    for idx, (mask, amap) in enumerate(zip(mask_list, map_list)):
        if mask.shape != amap.shape:
            raise ValueError(
                f"Mask/map shape mismatch at index {idx}: "
                f"{mask.shape} vs {amap.shape}"
            )

    masks_array = np.stack([(mask > 0).astype(np.float32) for mask in mask_list])
    maps_array = np.stack([np.asarray(amap, dtype=np.float32) for amap in map_list])
    flat_masks = masks_array.reshape(-1)
    flat_maps = maps_array.reshape(-1)

    # Skip if all one class
    if flat_masks.sum() == 0 or flat_masks.sum() == len(flat_masks):
        return {"auroc_px": 0.0, "f1_px": 0.0, "aupro_px": 0.0, "threshold_px": 0.0}

    auroc = roc_auc_score(flat_masks, flat_maps)
    f1_max, threshold = compute_f1_max(flat_masks, flat_maps)
    aupro = compute_aupro(masks_array, maps_array, device=aupro_device)

    return {
        "auroc_px": round(auroc * 100, 2),
        "f1_px": round(f1_max * 100, 2),
        "aupro_px": round(aupro * 100, 2),
        "threshold_px": round(threshold, 6),
    }


def compute_image_metrics(
    labels: np.ndarray, scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute image-level metrics.

    Args:
        labels: Ground-truth binary labels, shape (N,).
        scores: Predicted anomaly scores, shape (N,).

    Returns:
        Dict with keys: auroc_sp, f1_sp, ap_sp, threshold_sp
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc_sp": 0.0, "f1_sp": 0.0, "ap_sp": 0.0, "threshold_sp": 0.0}

    auroc = roc_auc_score(labels, scores)
    f1_max, threshold = compute_f1_max(labels, scores)
    ap = average_precision_score(labels, scores)

    return {
        "auroc_sp": round(auroc * 100, 2),
        "f1_sp": round(f1_max * 100, 2),
        "ap_sp": round(ap * 100, 2),
        "threshold_sp": round(threshold, 6),
    }


def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a ground-truth mask to the target dimensions.

    Args:
        mask: Binary mask of shape (H, W).
        target_h: Target height.
        target_w: Target width.

    Returns:
        Resized binary mask of shape (target_h, target_w).
    """
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        mask_tensor, size=(target_h, target_w), mode="nearest"
    )
    return (resized.squeeze().numpy() > 0.5).astype(np.float32)


def resize_mask_to_lowres(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Backward-compatible alias for older low-res metric paths."""
    return resize_mask(mask, target_h, target_w)


def resize_anomaly_map(
    anomaly_map: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Resize an anomaly map to the target dimensions with bilinear sampling."""
    map_tensor = torch.from_numpy(anomaly_map).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        map_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    return resized.squeeze().numpy().astype(np.float32)
