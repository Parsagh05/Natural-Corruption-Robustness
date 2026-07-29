# -*- coding: utf-8 -*-
"""
dataset.py - Dataset handler for MVTec AD and VisA test subsets with optional
cached Hendrycks corruptions.
"""

from collections import deque
import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from shared.corruption import (
    apply_corruption,
    apply_corruption_to_mask,
    is_corruption_category,
)

from .config import DatasetConfig, MVTEC_CATEGORIES, VISA_CATEGORIES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MVTEC_STYLE = "mvtec_style"
VISA_ORIGINAL = "visa_original"
SUPPORTED_CACHE_FORMATS = {"png", "jpeg", "jpg"}
VISA_SPLIT_FILES = (
    "split_csv/1cls.csv",
    "split_csv/test.csv",
    "metadata/visa/test.jsonl",
    "metadata/VisA/test.jsonl",
    "test.jsonl",
    "test.csv",
)
PERSISTENT_CORRUPTION_COLUMNS = {
    "relative_path",
    "cls_name",
    "category",
    "severity",
    "selected_corruption",
}


def _iter_image_files(root: Path) -> Iterable[Path]:
    """Yield image files below root with case-insensitive extension matching."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _candidate_roots(root: Path, max_depth: int = 4) -> Iterable[Path]:
    """Yield possible dataset roots without recursing into category payloads."""
    if not root.exists():
        return

    queue: deque[Tuple[Path, int]] = deque([(root, 0)])
    seen = set()

    while queue:
        current, depth = queue.popleft()
        resolved = current.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield current

        if depth >= max_depth:
            continue

        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue

        for child in sorted(children):
            queue.append((child, depth + 1))


def _mvtec_style_score(root: Path, categories: List[str]) -> int:
    return sum(1 for cat in categories if (root / cat / "test").is_dir())


def _visa_original_score(root: Path, categories: List[str]) -> int:
    return sum(
        1 for cat in categories if (root / cat / "Data" / "Images").is_dir()
    )


def _resolve_dataset_root(
    root: Path, categories: List[str], preferred_layout: str = "auto"
) -> Tuple[Path, str]:
    """
    Resolve Kaggle/KaggleHub wrapper paths to the directory that contains data.

    Supported layouts:
      - MVTec style: {root}/{category}/test/{defect_type}/*
      - Original VisA: {root}/{category}/Data/Images/{Normal,Anomaly}/*
    """
    if preferred_layout not in {"auto", MVTEC_STYLE, VISA_ORIGINAL}:
        raise ValueError(f"Unsupported dataset layout: {preferred_layout}")

    best_mvtec: Tuple[int, Path] = (0, root)
    best_visa: Tuple[int, Path] = (0, root)

    for candidate in _candidate_roots(root):
        if preferred_layout in {"auto", MVTEC_STYLE}:
            score = _mvtec_style_score(candidate, categories)
            if score > best_mvtec[0]:
                best_mvtec = (score, candidate)

        if preferred_layout in {"auto", VISA_ORIGINAL}:
            score = _visa_original_score(candidate, categories)
            if score > best_visa[0]:
                best_visa = (score, candidate)

    if preferred_layout == MVTEC_STYLE and best_mvtec[0] > 0:
        return best_mvtec[1], MVTEC_STYLE
    if preferred_layout == VISA_ORIGINAL and best_visa[0] > 0:
        return best_visa[1], VISA_ORIGINAL
    if preferred_layout == "auto":
        if best_mvtec[0] >= best_visa[0] and best_mvtec[0] > 0:
            return best_mvtec[1], MVTEC_STYLE
        if best_visa[0] > 0:
            return best_visa[1], VISA_ORIGINAL

    return root, preferred_layout


def _first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _normalize_relative_path(path: Path | str) -> str:
    return Path(path).as_posix()


def _load_persistent_corruption_assignments(
    plan_path: Path,
    dataset_category: str,
    corruption_category: str,
    severity: int,
) -> Dict[str, str]:
    """Load exactly one category/severity/class slice of a plan CSV."""
    with plan_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"Persistent corruption CSV has no header: {plan_path}")
        missing_columns = PERSISTENT_CORRUPTION_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Persistent corruption CSV is missing columns {sorted(missing_columns)}: "
                f"{plan_path}"
            )

        assignments: Dict[str, str] = {}
        for row in reader:
            if (
                row["cls_name"] != dataset_category
                or row["category"] != corruption_category
                or int(row["severity"]) != int(severity)
            ):
                continue
            relative_path = _normalize_relative_path(row["relative_path"])
            if relative_path in assignments:
                raise ValueError(
                    "Persistent corruption CSV has duplicate assignments for "
                    f"{relative_path}, {corruption_category}, severity {severity}: "
                    f"{plan_path}"
                )
            assignments[relative_path] = row["selected_corruption"]
    return assignments


def _find_visa_split_file(root: Path) -> Optional[Path]:
    for rel_path in VISA_SPLIT_FILES:
        split_path = root / rel_path
        if split_path.exists():
            return split_path
    return None


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _is_test_split(row: Dict[str, Any]) -> bool:
    split_value = _norm_text(
        row.get("split")
        or row.get("Split")
        or row.get("phase")
        or row.get("subset")
        or row.get("set")
    ).lower()
    if split_value:
        return split_value in {"test", "testing"}
    return True


def _row_value(row: Dict[str, Any], names: Iterable[str]) -> str:
    lower_row = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name in row and _norm_text(row[name]):
            return _norm_text(row[name])
        value = lower_row.get(name.lower())
        if _norm_text(value):
            return _norm_text(value)
    return ""


def _read_split_rows(split_path: Path) -> List[Dict[str, Any]]:
    if split_path.suffix.lower() == ".jsonl":
        rows = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with split_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))


def _resolve_visa_row_path(root: Path, raw_path: str, category: str) -> Optional[Path]:
    if not raw_path:
        return None

    path = Path(raw_path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            root / path,
            root / category / path,
            root / category / "Data" / "Images" / path,
            root / category / "Data" / "Masks" / path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _mask_candidates(mask_dir: Path, img_path: Path, source_root: Path) -> List[Path]:
    try:
        rel_path = img_path.relative_to(source_root)
    except ValueError:
        rel_path = Path(img_path.name)

    stem_rel = rel_path.with_suffix("")
    candidates = [
        mask_dir / rel_path,
        mask_dir / rel_path.with_suffix(".png"),
        mask_dir / f"{img_path.stem}.png",
        mask_dir / f"{img_path.stem}_mask.png",
        mask_dir / stem_rel.with_name(f"{stem_rel.name}_mask.png"),
    ]

    for ext in MASK_EXTENSIONS:
        candidates.append(mask_dir / stem_rel.with_suffix(ext))
        candidates.append(mask_dir / stem_rel.with_name(f"{stem_rel.name}_mask{ext}"))

    return candidates


class AnomalyDetectionDataset(Dataset):
    """
    Generic anomaly detection dataset that loads test images and masks,
    applying Hendrycks corruptions on demand and optionally caching them to disk.
    """

    def __init__(
        self,
        config: DatasetConfig,
        corruption_type: Optional[str] = None,
        severity: Optional[int] = None,
        base_seed: int = 111,
        transform=None,
        category: Optional[str] = None,
        corruption_cache_root: Optional[Path] = None,
        corruption_cache_format: str = "png",
        categorized_corruptions: bool = False,
        categorized_corruption_plan: Optional[Path] = None,
    ):
        """
        Args:
            config: DatasetConfig with root path and category info.
            corruption_type: If None, return clean images. Otherwise apply corruption.
            severity: Severity level for corruption (1-5).
            base_seed: Base seed for reproducible corruption.
            transform: Optional torchvision transform to apply after corruption.
            category: If specified, only load this category. Otherwise load all.
            corruption_cache_root: Optional directory where corrupted images are
                persisted and reused across models/runs.
            corruption_cache_format: Cache encoding, either "png" or "jpeg".
            categorized_corruptions: Interpret ``corruption_type`` as a
                category (noise, blur, photometric, or geometric), then load
                each image's concrete operation from the persistent CSV plan.
            categorized_corruption_plan: CSV generated by the categorized demo.
        """
        self.config = config
        self.corruption_type = corruption_type
        self.severity = severity
        self.base_seed = base_seed
        self.transform = transform
        self.corruption_cache_root = Path(corruption_cache_root) if corruption_cache_root else None
        self.categorized_corruptions = categorized_corruptions
        self.categorized_corruption_plan = (
            Path(categorized_corruption_plan)
            if categorized_corruption_plan is not None else None
        )
        self.corruption_cache_format = corruption_cache_format.lower()
        if self.corruption_cache_format == "jpg":
            self.corruption_cache_format = "jpeg"
        if self.corruption_cache_format not in SUPPORTED_CACHE_FORMATS:
            raise ValueError(
                "corruption_cache_format must be one of: png, jpeg, jpg"
            )
        self.root_path, self.layout = _resolve_dataset_root(
            config.root_path,
            config.categories,
            config.layout,
        )

        categories = [category] if category else config.categories
        self.samples: List[Dict[str, Any]] = []

        if self.layout == VISA_ORIGINAL:
            self._load_visa_original_samples(categories)
        else:
            self._load_mvtec_style_samples(categories)

        if self.categorized_corruptions:
            if not self.corruption_type or not is_corruption_category(self.corruption_type):
                raise ValueError(
                    "categorized_corruptions=True requires corruption_type to be "
                    "one of: noise, blur, photometric, geometric"
                )
            if self.severity is None:
                raise ValueError("categorized_corruptions=True requires a severity")
            if self.categorized_corruption_plan is None:
                raise ValueError(
                    "categorized_corruptions=True requires categorized_corruption_plan"
                )
            assignments = _load_persistent_corruption_assignments(
                self.categorized_corruption_plan,
                category or "",
                self.corruption_type,
                self.severity,
            )
            sample_paths = {
                _normalize_relative_path(sample["image_path"].relative_to(self.root_path))
                for sample in self.samples
            }
            plan_paths = set(assignments)
            if sample_paths != plan_paths:
                missing_from_plan = sorted(sample_paths - plan_paths)
                unexpected_in_plan = sorted(plan_paths - sample_paths)
                raise ValueError(
                    "Persistent corruption plan and loaded evaluation samples differ "
                    f"for dataset class '{category}', category '{self.corruption_type}', "
                    f"severity {self.severity}. Missing from plan: "
                    f"{missing_from_plan[:3]} (total {len(missing_from_plan)}); "
                    f"unexpected in plan: {unexpected_in_plan[:3]} "
                    f"(total {len(unexpected_in_plan)})."
                )
            for sample in self.samples:
                relative_path = _normalize_relative_path(
                    sample["image_path"].relative_to(self.root_path)
                )
                sample["selected_corruption"] = assignments[relative_path]

    def _corruption_cache_path(self, rel_path: str) -> Path:
        suffix = ".jpg" if self.corruption_cache_format == "jpeg" else ".png"
        relative = Path(rel_path).with_suffix(suffix)
        return (
            self.corruption_cache_root
            / self.config.name
            / f"seed_{self.base_seed}"
            / self.corruption_type
            / f"level_{self.severity}"
            / relative
        )

    def _load_or_create_cached_corruption(
        self, img: Image.Image, rel_path: str, concrete_corruption: str
    ) -> Image.Image:
        if self.corruption_cache_root is None:
            return apply_corruption(
                img,
                concrete_corruption,
                self.severity,
                rel_path,
                self.base_seed,
            )

        cache_path = self._corruption_cache_path(rel_path)
        if cache_path.exists():
            with Image.open(cache_path) as cached:
                return cached.convert("RGB").copy()

        corrupted = apply_corruption(
            img,
            concrete_corruption,
            self.severity,
            rel_path,
            self.base_seed,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
        save_kwargs = {}
        save_format = self.corruption_cache_format.upper()
        if self.corruption_cache_format == "jpeg":
            save_format = "JPEG"
            save_kwargs = {"quality": 95, "subsampling": 0}
        corrupted.save(temp_path, format=save_format, **save_kwargs)
        temp_path.replace(cache_path)
        return corrupted

    def _load_mvtec_style_samples(self, categories: List[str]) -> None:
        for cat in categories:
            test_dir = self.root_path / cat / self.config.test_subdir
            mask_dir = self.root_path / cat / self.config.mask_subdir

            if not test_dir.exists():
                continue

            # Iterate over defect type subdirectories
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue

                defect_type = defect_dir.name
                is_normal = defect_type.lower() == "good"

                for img_path in _iter_image_files(defect_dir):
                    mask_path = None
                    if not is_normal:
                        # Try to find corresponding mask
                        defect_mask_dir = mask_dir / defect_type
                        mask_path = _first_existing_path(
                            _mask_candidates(defect_mask_dir, img_path, defect_dir)
                        )

                    self.samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "category": cat,
                        "defect_type": defect_type,
                        "is_anomaly": not is_normal,
                        "sample_id": f"{cat}/{defect_type}/{img_path.name}",
                    })

    def _load_visa_original_samples(self, categories: List[str]) -> None:
        split_path = _find_visa_split_file(self.root_path)
        if split_path is not None:
            self._load_visa_split_samples(categories, split_path)
            return

        for cat in categories:
            category_root = self.root_path / cat
            image_root = category_root / "Data" / "Images"
            mask_root = category_root / "Data" / "Masks" / "Anomaly"

            if not image_root.exists():
                continue

            normal_dir = image_root / "Normal"
            for img_path in _iter_image_files(normal_dir):
                rel_path = img_path.relative_to(normal_dir)
                self.samples.append({
                    "image_path": img_path,
                    "mask_path": None,
                    "category": cat,
                    "defect_type": "good",
                    "is_anomaly": False,
                    "sample_id": f"{cat}/good/{rel_path.as_posix()}",
                })

            anomaly_dir = image_root / "Anomaly"
            for img_path in _iter_image_files(anomaly_dir):
                rel_path = img_path.relative_to(anomaly_dir)
                defect_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "bad"
                mask_path = _first_existing_path(
                    _mask_candidates(mask_root, img_path, anomaly_dir)
                )
                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "category": cat,
                    "defect_type": defect_type,
                    "is_anomaly": True,
                    "sample_id": f"{cat}/{defect_type}/{rel_path.as_posix()}",
                })

    def _load_visa_split_samples(self, categories: List[str], split_path: Path) -> None:
        category_set = set(categories)
        rows = _read_split_rows(split_path)

        for row in rows:
            if not _is_test_split(row):
                continue

            cat = _row_value(
                row,
                ("object", "category", "class", "class_name", "classname"),
            )
            image_raw = _row_value(
                row,
                ("image", "image_path", "img_path", "file", "filename", "path"),
            )
            if not cat and image_raw:
                parts = Path(image_raw).parts
                for part in parts:
                    if part in category_set:
                        cat = part
                        break
            if cat not in category_set:
                continue

            img_path = _resolve_visa_row_path(self.root_path, image_raw, cat)
            if img_path is None or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_text = _row_value(
                row,
                ("label", "is_anomaly", "anomaly", "type", "defect_type"),
            ).lower()
            is_anomaly = label_text not in {"", "normal", "good", "0", "false"}

            mask_raw = _row_value(row, ("mask", "mask_path", "gt", "ground_truth"))
            mask_path = _resolve_visa_row_path(self.root_path, mask_raw, cat)
            if is_anomaly and mask_path is None:
                anomaly_root = self.root_path / cat / "Data" / "Images" / "Anomaly"
                mask_root = self.root_path / cat / "Data" / "Masks" / "Anomaly"
                try:
                    rel_path = img_path.relative_to(anomaly_root)
                    defect_mask_dir = mask_root / rel_path.parent
                    mask_path = _first_existing_path(
                        _mask_candidates(defect_mask_dir, img_path, anomaly_root)
                    )
                except ValueError:
                    mask_path = _first_existing_path(
                        _mask_candidates(mask_root, img_path, self.root_path)
                    )

            defect_type = "good"
            if is_anomaly:
                defect_type = _row_value(row, ("defect_type", "defect", "type"))
                if defect_type.lower() in {"", "anomaly", "1", "true"}:
                    try:
                        rel_path = img_path.relative_to(
                            self.root_path / cat / "Data" / "Images" / "Anomaly"
                        )
                        defect_type = rel_path.parts[0] if len(rel_path.parts) > 1 else "bad"
                    except ValueError:
                        defect_type = "bad"

            try:
                rel_sample = img_path.relative_to(self.root_path).as_posix()
            except ValueError:
                rel_sample = img_path.name

            self.samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "category": cat,
                "defect_type": defect_type,
                "is_anomaly": is_anomaly,
                "sample_id": f"{cat}/{defect_type}/{rel_sample}",
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        rel_path = str(sample["image_path"].relative_to(self.root_path))
        concrete_corruption = sample.get("selected_corruption", self.corruption_type)

        # Apply corruption if specified
        if self.corruption_type is not None and self.corruption_type != "clean":
            img = self._load_or_create_cached_corruption(
                img, rel_path, concrete_corruption
            )

        # Load mask
        mask = None
        if sample["mask_path"] is not None:
            mask_image = Image.open(sample["mask_path"]).convert("L")
            if self.corruption_type is not None and self.corruption_type != "clean":
                mask_image = apply_corruption_to_mask(
                    mask_image,
                    concrete_corruption,
                    self.severity,
                    rel_path,
                    self.base_seed,
                )
            mask = np.array(mask_image)
            mask = (mask > 0).astype(np.float32)
        else:
            # Normal sample -> zero mask
            w, h = img.size
            mask = np.zeros((h, w), dtype=np.float32)

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "mask": mask,
            "label": int(sample["is_anomaly"]),
            "category": sample["category"],
            "sample_id": sample["sample_id"],
            "defect_type": sample["defect_type"],
            "selected_corruption": sample.get("selected_corruption"),
        }


def build_dataset_configs(
    mvtec_root: Optional[str] = None,
    visa_root: Optional[str] = None,
) -> List[DatasetConfig]:
    """Build dataset configurations for MVTec and VisA Kaggle layouts."""
    configs = []

    if mvtec_root:
        root_path, layout = _resolve_dataset_root(
            Path(mvtec_root), MVTEC_CATEGORIES, MVTEC_STYLE
        )
        configs.append(DatasetConfig(
            name="MVTec",
            root_path=root_path,
            categories=MVTEC_CATEGORIES,
            test_subdir="test",
            mask_subdir="ground_truth",
            layout=layout,
        ))

    if visa_root:
        root_path, layout = _resolve_dataset_root(
            Path(visa_root), VISA_CATEGORIES, "auto"
        )
        configs.append(DatasetConfig(
            name="VisA",
            root_path=root_path,
            categories=VISA_CATEGORIES,
            test_subdir="test",
            mask_subdir="ground_truth",
            layout=layout,
        ))

    return configs
