# -*- coding: utf-8 -*-
"""Deterministic image and mask corruptions shared by all shot modes."""

from contextlib import contextmanager
import hashlib
from typing import Dict
import numpy as np
from PIL import Image

# Lazy-loaded corruption functions from the shared frozen ImageNet-C copy.
_CORRUPTION_FNS = None


CATEGORIZED_CORRUPTIONS: Dict[str, tuple[str, ...]] = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "motion_blur", "zoom_blur"),
    "photometric": ("brightness", "contrast"),
    "geometric": ("rotation", "zooming", "shifting"),
}
GEOMETRIC_CORRUPTIONS = frozenset(CATEGORIZED_CORRUPTIONS["geometric"])


def stable_corruption_seed(
    base_seed: int,
    rel_path: str,
    corruption: str,
    severity: int,
) -> int:
    """Generate a deterministic per-sample corruption seed."""
    key = f"{base_seed}|{rel_path}|{corruption}|{severity}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _load_corruption_functions():
    """Lazily import corruption functions from the imagenet_c package."""
    global _CORRUPTION_FNS
    if _CORRUPTION_FNS is not None:
        return _CORRUPTION_FNS

    from .imagenet_c.corruptions import (
        gaussian_noise,
        shot_noise,
        impulse_noise,
        defocus_blur,
        motion_blur,
        zoom_blur,
        brightness,
        contrast,
    )

    def rotation(image: Image.Image, severity: int = 1) -> np.ndarray:
        degrees = (5, 10, 15, 20, 25)[severity - 1]
        angle = float(np.random.uniform(-degrees, degrees))
        return np.asarray(image.rotate(angle, resample=Image.Resampling.BILINEAR))

    def zooming(image: Image.Image, severity: int = 1) -> np.ndarray:
        scale = (1.05, 1.10, 1.15, 1.20, 1.25)[severity - 1]
        width, height = image.size
        enlarged = image.resize(
            (round(width * scale), round(height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
        left = (enlarged.width - width) // 2
        top = (enlarged.height - height) // 2
        return np.asarray(enlarged.crop((left, top, left + width, top + height)))

    def shifting(image: Image.Image, severity: int = 1) -> np.ndarray:
        max_fraction = (0.02, 0.04, 0.06, 0.08, 0.10)[severity - 1]
        width, height = image.size
        dx = int(round(np.random.uniform(-max_fraction, max_fraction) * width))
        dy = int(round(np.random.uniform(-max_fraction, max_fraction) * height))
        return np.asarray(image.transform(
            (width, height),
            Image.Transform.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        ))

    _CORRUPTION_FNS = {
        "gaussian_noise": gaussian_noise,
        "shot_noise": shot_noise,
        "impulse_noise": impulse_noise,
        "defocus_blur": defocus_blur,
        "motion_blur": motion_blur,
        "zoom_blur": zoom_blur,
        "brightness": brightness,
        "contrast": contrast,
        "rotation": rotation,
        "zooming": zooming,
        "shifting": shifting,
    }
    return _CORRUPTION_FNS


@contextmanager
def fixed_numpy_seed(seed: int):
    """Context manager to temporarily fix numpy random state."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: int,
    rel_path: str,
    base_seed: int = 111,
) -> Image.Image:
    """
    Apply a single Hendrycks corruption to a PIL image.

    Args:
        image: Clean PIL RGB image.
        corruption_type: Name of one of the supported corruption functions.
        severity: Severity level (1-5).
        rel_path: Relative path of image for reproducible seeding.
        base_seed: Base seed for stable per-sample randomness.

    Returns:
        Corrupted PIL RGB image.
    """
    fns = _load_corruption_functions()

    if corruption_type not in fns:
        raise ValueError(
            f"Unknown corruption: {corruption_type}. "
            f"Available: {list(fns.keys())}"
        )

    seed = stable_corruption_seed(base_seed, rel_path, corruption_type, severity)

    with fixed_numpy_seed(seed):
        corrupted = fns[corruption_type](image, severity=severity)

    corrupted = np.uint8(np.clip(corrupted, 0, 255))
    return Image.fromarray(corrupted).convert("RGB")


def apply_corruption_to_mask(
    mask: Image.Image,
    corruption_type: str,
    severity: int,
    rel_path: str,
    base_seed: int = 111,
) -> Image.Image:
    """Transform a ground-truth mask consistently with geometric corruption.

    Noise, blur, and photometric operations only alter appearance, so their
    masks remain unchanged.  Geometric transforms use nearest-neighbor
    resampling to retain a binary mask and the same deterministic seed as the
    corresponding image operation.
    """
    if corruption_type not in GEOMETRIC_CORRUPTIONS:
        return mask.convert("L")

    seed = stable_corruption_seed(base_seed, rel_path, corruption_type, severity)
    mask = mask.convert("L")
    with fixed_numpy_seed(seed):
        if corruption_type == "rotation":
            degrees = (5, 10, 15, 20, 25)[severity - 1]
            angle = float(np.random.uniform(-degrees, degrees))
            transformed = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        elif corruption_type == "zooming":
            scale = (1.05, 1.10, 1.15, 1.20, 1.25)[severity - 1]
            width, height = mask.size
            enlarged = mask.resize(
                (round(width * scale), round(height * scale)),
                resample=Image.Resampling.NEAREST,
            )
            left = (enlarged.width - width) // 2
            top = (enlarged.height - height) // 2
            transformed = enlarged.crop((left, top, left + width, top + height))
        else:  # shifting
            max_fraction = (0.02, 0.04, 0.06, 0.08, 0.10)[severity - 1]
            width, height = mask.size
            dx = int(round(np.random.uniform(-max_fraction, max_fraction) * width))
            dy = int(round(np.random.uniform(-max_fraction, max_fraction) * height))
            transformed = mask.transform(
                (width, height),
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
    return transformed


def is_corruption_category(name: str) -> bool:
    """Return whether *name* identifies a categorized protocol group."""
    return name in CATEGORIZED_CORRUPTIONS


__all__ = [
    "CATEGORIZED_CORRUPTIONS",
    "GEOMETRIC_CORRUPTIONS",
    "apply_corruption",
    "apply_corruption_to_mask",
    "fixed_numpy_seed",
    "is_corruption_category",
    "stable_corruption_seed",
]
