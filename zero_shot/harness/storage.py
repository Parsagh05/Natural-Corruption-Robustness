# -*- coding: utf-8 -*-
"""
storage.py - Incremental artifact serialization and memory management.

Handles the disk layout:
    outputs/{model_name}/{dataset_name}/{category_name}/{noise_type}/level_{severity}/
        metadata.json
        raw_scores.npy
        lowres_maps.npy
"""

import gc
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class ArtifactStorage:
    """Manages incremental serialization of inference artifacts."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def get_artifact_dir(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        noise_type: str,
        severity: int,
    ) -> Path:
        """Get the directory path for a specific artifact set."""
        path = (
            self.output_root
            / model_name
            / dataset_name
            / category
            / noise_type
            / f"level_{severity}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_artifacts(
        self,
        model_name: str,
        dataset_name: str,
        category: str,
        noise_type: str,
        severity: int,
        scores: np.ndarray,
        lowres_maps: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> Path:
        """
        Save inference artifacts for a single model/dataset/category/noise/level.
        """
        artifact_dir = self.get_artifact_dir(
            model_name, dataset_name, category, noise_type, severity
        )

        np.save(artifact_dir / "raw_scores.npy", scores.astype(np.float32))
        np.save(artifact_dir / "lowres_maps.npy", lowres_maps.astype(np.float32))
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return artifact_dir

    def zip_and_cleanup_model(self, model_name: str) -> Optional[Path]:
        """Zip the output directory of a completed model and clean it up."""
        model_dir = self.output_root / model_name
        if not model_dir.exists():
            return None

        zip_path = self.output_root / f"{model_name}_artifacts.zip"
        temporary_zip_path = zip_path.with_suffix(zip_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(
                temporary_zip_path, "w", zipfile.ZIP_DEFLATED
            ) as zf:
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.output_root)
                        zf.write(file_path, arcname)
            temporary_zip_path.replace(zip_path)
        finally:
            if temporary_zip_path.exists():
                temporary_zip_path.unlink()

        shutil.rmtree(model_dir)
        return zip_path

    @staticmethod
    def cleanup_memory() -> None:
        """Force garbage collection and CUDA cache clearing."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class IncrementalAccumulator:
    """
    Accumulates scores and maps per category to avoid out-of-memory failures.
    """

    def __init__(self):
        self.scores: List[float] = []
        self.maps: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        score: float,
        lowres_map: np.ndarray,
        label: int,
        sample_id: str,
        mask_path: Optional[str] = None,
        image_path: Optional[str] = None,
        selected_corruption: Optional[str] = None,
    ) -> None:
        """Add a single sample's results."""
        self.scores.append(score)
        self.maps.append(lowres_map)
        self.metadata.append({
            "label": label,
            "sample_id": sample_id,
            "mask_path": str(mask_path) if mask_path else None,
            "image_path": str(image_path) if image_path else None,
            "selected_corruption": selected_corruption,
        })

    def flush(self) -> tuple:
        """Return accumulated data and clear buffers."""
        scores = np.array(self.scores, dtype=np.float32)

        if len(self.maps) > 0:
            max_h = max(m.shape[0] for m in self.maps)
            max_w = max(m.shape[1] for m in self.maps)
            padded_maps = np.zeros(
                (len(self.maps), max_h, max_w), dtype=np.float32
            )
            for i, m in enumerate(self.maps):
                padded_maps[i, : m.shape[0], : m.shape[1]] = m
        else:
            padded_maps = np.zeros((0, 14, 14), dtype=np.float32)

        metadata = self.metadata.copy()
        self.scores.clear()
        self.maps.clear()
        self.metadata.clear()

        return scores, padded_maps, metadata

    @property
    def size(self) -> int:
        return len(self.scores)
