"""Few-shot model wrappers.

INP-Former is constructed from the official source tree and evaluated with the
official few-shot checkpoint and test-time scoring path.  New few-shot models
can be added with :func:`register_model` without changing the runner.
"""

from functools import partial
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from zero_shot.harness.models import APRILGANWrapper, BaseModelWrapper


OFFICIAL_INPFORMER_COMMIT = "17d265381d9b323a2ef6e05aab0665a85edebe84"
OFFICIAL_PROMPTAD_COMMIT = "0f86ce0dc1ed59007d51348d8d566aed31360cf9"
OFFICIAL_AFCLIP_COMMIT = "bb7edec4128a76f29cb573cd3002538bf250b2fe"
OFFICIAL_APRILGAN_COMMIT = APRILGANWrapper.OFFICIAL_SOURCE_COMMIT
OFFICIAL_CHECKPOINT_URLS: Dict[int, Dict[str, str]] = {
    1: {
        "mvtec": "https://drive.google.com/file/d/1ymAywov3JFFVzwDpcdt9Tj_iFv-mk32c/view?usp=sharing",
        "visa": "https://drive.google.com/file/d/1mwpzXjLmjYLWFDx4dUF1yuErzL37K21p/view?usp=sharing",
    },
    2: {
        "mvtec": "https://drive.google.com/file/d/1K9X8-v1bSy_mgrbVSK0w6Fx525clSTtz/view?usp=sharing",
        "visa": "https://drive.google.com/file/d/1_vlO4OSQSze095ddhkkyRWCOA2IRVLia/view?usp=sharing",
    },
    4: {
        "mvtec": "https://drive.google.com/file/d/15UtpeFveG2azUQmhogoET2HifEyIKSvX/view?usp=sharing",
        "visa": "https://drive.google.com/file/d/1MFZcRNwALdPPv1Wemk5_1WLq76BINdky/view?usp=sharing",
    },
}

_DATASET_ALIASES = {
    "mvtec": "mvtec",
    "mvtecad": "mvtec",
    "mvtecadataset": "mvtec",
    "visa": "visa",
}

PROMPTAD_DATASET_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "mvtec": (
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    ),
    "visa": (
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
        "pipe_fryum",
    ),
}

# Preserve the category order in the released AF-CLIP datasets.  Its random
# support sampler advances one NumPy RNG across classes, so the order is part
# of a reproducible few-shot support selection.
AFCLIP_DATASET_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "mvtec": (
        "carpet", "grid", "leather", "tile", "wood", "bottle", "cable",
        "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush",
        "transistor", "zipper",
    ),
    "visa": PROMPTAD_DATASET_CATEGORIES["visa"],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize harness and official dataset labels to checkpoint keys."""
    compact = "".join(character for character in dataset_name.lower() if character.isalnum())
    try:
        return _DATASET_ALIASES[compact]
    except KeyError as exc:
        raise ValueError(
            f"INP-Former few-shot checkpoints support MVTec AD and VisA; "
            f"got {dataset_name!r}."
        ) from exc


def official_checkpoint_directory(shot: int, dataset_name: str) -> str:
    """Return the directory name used by the official INP-Former code."""
    dataset = normalize_dataset_name(dataset_name)
    official_dataset = "MVTec-AD" if dataset == "mvtec" else "VisA"
    return (
        f"INP-Former-Few-Shot-{shot}_dataset={official_dataset}_"
        "Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6"
    )


def discover_official_checkpoints(
    checkpoint_root: str,
    shots: Sequence[int] = (1, 2, 4),
    datasets: Sequence[str] = ("mvtec", "visa"),
) -> Dict[int, Dict[str, str]]:
    """Locate the selected official ``model.pth`` files below one root.

    The files all have the same basename, so their official parent-directory
    names are required to prevent silently pairing a shot with the wrong
    dataset checkpoint.
    """
    root = Path(checkpoint_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    selected_shots = tuple(int(shot) for shot in shots)
    invalid_shots = sorted(set(selected_shots) - {1, 2, 4})
    if not selected_shots or invalid_shots:
        raise ValueError(
            "shots must contain one or more official settings from (1, 2, 4); "
            f"got {selected_shots}."
        )
    dataset_values = (datasets,) if isinstance(datasets, str) else datasets
    selected_datasets = tuple(
        normalize_dataset_name(dataset) for dataset in dataset_values
    )
    if not selected_datasets:
        raise ValueError("Select at least one checkpoint dataset.")
    if len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError(
            f"Checkpoint datasets must be unique; got {selected_datasets}."
        )

    resolved: Dict[int, Dict[str, str]] = {}
    missing = []
    all_model_files = list(root.rglob("model.pth"))
    for shot in selected_shots:
        resolved[shot] = {}
        for dataset in selected_datasets:
            directory_name = official_checkpoint_directory(shot, dataset)
            direct_path = root / directory_name / "model.pth"
            matches = [
                path for path in all_model_files
                if path.parent.name == directory_name
            ]
            if direct_path.is_file():
                checkpoint_path = direct_path
            elif len(matches) == 1:
                checkpoint_path = matches[0]
            elif len(matches) > 1:
                raise RuntimeError(
                    f"Multiple copies of {directory_name}/model.pth found: "
                    f"{matches}"
                )
            else:
                missing.append(str(direct_path))
                continue
            resolved[shot][dataset] = str(checkpoint_path.resolve())

    if missing:
        raise FileNotFoundError(
            "The selected official checkpoint suite is incomplete. Missing:\n  - "
            + "\n  - ".join(missing)
        )
    return resolved


def discover_promptad_checkpoints(
    checkpoint_root: str,
    shots: Sequence[int] = (1, 2, 4),
    datasets: Sequence[str] = ("mvtec", "visa"),
    verify_hashes: bool = False,
) -> Dict[int, Dict[str, Dict[str, Dict[str, str]]]]:
    """Discover class/task PromptAD buffers from exported checkpoint indexes.

    Each training artifact contains a ``checkpoint_index.json`` whose keys use
    ``dataset/<shot>-shot/category/{cls,seg}``.  Reading those indexes avoids
    guessing from identical directory layouts when several Kaggle shards are
    attached at once.
    """
    root = Path(checkpoint_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"PromptAD checkpoint root does not exist: {root}")

    selected_shots = tuple(int(shot) for shot in shots)
    invalid_shots = sorted(set(selected_shots) - {1, 2, 4})
    if not selected_shots or invalid_shots:
        raise ValueError(
            "shots must contain one or more PromptAD settings from (1, 2, 4); "
            f"got {selected_shots}."
        )
    if len(set(selected_shots)) != len(selected_shots):
        raise ValueError(f"PromptAD shot settings must be unique: {selected_shots}.")

    dataset_values = (datasets,) if isinstance(datasets, str) else datasets
    selected_datasets = tuple(
        normalize_dataset_name(dataset) for dataset in dataset_values
    )
    if not selected_datasets:
        raise ValueError("Select at least one PromptAD checkpoint dataset.")
    if len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError(
            f"PromptAD checkpoint datasets must be unique: {selected_datasets}."
        )

    index_paths = sorted(root.rglob("checkpoint_index.json"))
    if not index_paths:
        raise FileNotFoundError(
            "No checkpoint_index.json was found below the PromptAD checkpoint "
            f"root: {root}"
        )

    resolved: Dict[int, Dict[str, Dict[str, Dict[str, str]]]] = {
        shot: {dataset: {} for dataset in selected_datasets}
        for shot in selected_shots
    }
    indexed_entries = 0
    for index_path in index_paths:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError(f"PromptAD checkpoint index must be an object: {index_path}")
        package_root = index_path.parent
        for key, metadata in payload.items():
            parts = str(key).split("/")
            if len(parts) != 4 or not parts[1].endswith("-shot"):
                raise ValueError(f"Malformed PromptAD checkpoint key {key!r} in {index_path}")
            dataset = normalize_dataset_name(parts[0])
            try:
                shot = int(parts[1][:-5])
            except ValueError as exc:
                raise ValueError(
                    f"Malformed PromptAD shot key {parts[1]!r} in {index_path}"
                ) from exc
            category, task = parts[2], parts[3].lower()
            if (
                shot not in selected_shots
                or dataset not in selected_datasets
            ):
                continue
            if task not in {"cls", "seg"}:
                raise ValueError(f"Unknown PromptAD task {task!r} in {index_path}")
            if not isinstance(metadata, Mapping) or not metadata.get("path"):
                raise ValueError(f"Missing path metadata for {key!r} in {index_path}")
            checkpoint = package_root / str(metadata["path"])
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Indexed PromptAD checkpoint is missing: {checkpoint}"
                )
            if verify_hashes and metadata.get("sha256"):
                actual_hash = _sha256_file(checkpoint)
                if actual_hash.lower() != str(metadata["sha256"]).lower():
                    raise RuntimeError(
                        f"PromptAD checkpoint SHA-256 mismatch for {checkpoint}: "
                        f"expected {metadata['sha256']}, got {actual_hash}."
                    )
            category_mapping = resolved[shot][dataset].setdefault(category, {})
            if task in category_mapping:
                raise RuntimeError(
                    "Duplicate PromptAD checkpoint entry for "
                    f"{dataset}/{shot}-shot/{category}/{task}: "
                    f"{category_mapping[task]} and {checkpoint}"
                )
            category_mapping[task] = str(checkpoint.resolve())
            indexed_entries += 1

    missing = []
    for shot in selected_shots:
        for dataset in selected_datasets:
            for category in PROMPTAD_DATASET_CATEGORIES[dataset]:
                tasks = resolved[shot][dataset].get(category, {})
                for task in ("cls", "seg"):
                    if task not in tasks:
                        missing.append(f"{dataset}/{shot}-shot/{category}/{task}")
    if missing:
        raise FileNotFoundError(
            "The selected PromptAD checkpoint suite is incomplete. Missing:\n  - "
            + "\n  - ".join(missing)
        )
    if indexed_entries == 0:
        raise FileNotFoundError(
            "The PromptAD indexes contained no entries for the selected shots "
            f"and datasets: shots={selected_shots}, datasets={selected_datasets}."
        )
    return resolved


def discover_afclip_checkpoints(
    checkpoint_root: str,
) -> Dict[str, Dict[str, str]]:
    """Locate the four released AF-CLIP prompt/adaptor files.

    The official repository checks these small trained components into its
    ``weight`` directory.  AF-CLIP+ does not train another checkpoint: target
    normal images populate an in-memory nearest-neighbour gallery at runtime.
    """
    root = Path(checkpoint_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"AF-CLIP checkpoint root does not exist: {root}")

    resolved: Dict[str, Dict[str, str]] = {}
    missing = []
    for source_dataset in ("mvtec", "visa"):
        resolved[source_dataset] = {}
        for component in ("prompt", "adaptor"):
            filename = f"{source_dataset}_{component}.pt"
            candidates = sorted(path for path in root.rglob(filename) if path.is_file())
            if len(candidates) == 1:
                resolved[source_dataset][component] = str(candidates[0].resolve())
            elif len(candidates) > 1:
                raise RuntimeError(
                    f"Multiple AF-CLIP files named {filename} found: {candidates}"
                )
            else:
                missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "The released AF-CLIP checkpoint suite is incomplete. Missing: "
            + ", ".join(missing)
        )
    return resolved


def discover_aprilgan_checkpoints(checkpoint_root: str) -> Dict[str, str]:
    """Locate APRIL-GAN's two released cross-dataset linear-layer weights."""
    root = Path(checkpoint_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"APRIL-GAN checkpoint root does not exist: {root}")
    resolved: Dict[str, str] = {}
    missing = []
    for source_dataset in ("mvtec", "visa"):
        filename = f"{source_dataset}_pretrained.pth"
        candidates = sorted(path for path in root.rglob(filename) if path.is_file())
        if len(candidates) == 1:
            resolved[source_dataset] = str(candidates[0].resolve())
        elif len(candidates) > 1:
            raise RuntimeError(
                f"Multiple APRIL-GAN files named {filename} found: {candidates}"
            )
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(
            "The released APRIL-GAN checkpoint suite is incomplete. Missing: "
            + ", ".join(missing)
        )
    return resolved


def _resolve_git_commit(root: Path) -> Optional[str]:
    git_dir = root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head[5:]
    loose_ref = git_dir / reference
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    return None


def _resolve_checkpoint_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_dir():
        direct = path / "model.pth"
        candidates = sorted(path.glob("*.pth"))
        if direct.is_file():
            return direct.resolve()
        if len(candidates) == 1:
            return candidates[0].resolve()
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError(f"INP-Former checkpoint not found: {path}")


def _official_gaussian_kernel(
    kernel_size: int = 5, sigma: float = 4.0
) -> torch.Tensor:
    """Create the normalized kernel used by official ``evaluation_batch``."""
    coordinates = torch.arange(kernel_size)
    x_grid = coordinates.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1) / 2.0
    variance = sigma**2
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2.0 * variance)
    )
    kernel = kernel / torch.sum(kernel)
    return kernel.view(1, 1, kernel_size, kernel_size)


class INPFormerWrapper(BaseModelWrapper):
    """Paper-faithful inference wrapper for official INP-Former few-shot weights."""

    OFFICIAL_ENCODER = "dinov2reg_vit_base_14"
    OFFICIAL_INPUT_SIZE = 448
    OFFICIAL_CROP_SIZE = 392
    OFFICIAL_METRIC_SIZE = 256
    OFFICIAL_INP_NUM = 6
    OFFICIAL_TOP_RATIO = 0.01
    fail_on_inference_error = True

    def __init__(
        self,
        inpformer_root: str = "",
        checkpoint_paths: Optional[Mapping[str, str]] = None,
        shot: int = 1,
        encoder: str = OFFICIAL_ENCODER,
        input_size: int = OFFICIAL_INPUT_SIZE,
        crop_size: int = OFFICIAL_CROP_SIZE,
        inp_num: int = OFFICIAL_INP_NUM,
        strict_source_commit: bool = False,
        **kwargs: Any,
    ) -> None:
        if shot not in (1, 2, 4):
            raise ValueError("Official INP-Former few-shot weights use shot=1, 2, or 4.")
        super().__init__(f"INP-Former-{shot}-shot", **kwargs)
        self.inpformer_root = inpformer_root
        self.checkpoint_paths = {
            normalize_dataset_name(name): str(path)
            for name, path in (checkpoint_paths or {}).items()
        }
        self.shot = shot
        self.encoder_name = encoder
        self.input_size = input_size
        self.crop_size = crop_size
        self.inp_num = inp_num
        self.strict_source_commit = strict_source_commit
        self.source_root: Optional[Path] = None
        self.source_commit: Optional[str] = None
        self.active_dataset: Optional[str] = None
        self.active_checkpoint: Optional[Path] = None
        self._normalization_mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        )[:, None, None]
        self._normalization_std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        )[:, None, None]
        self._gaussian_kernel = _official_gaussian_kernel()

    def _center_crop_box(self) -> Tuple[int, int, int, int]:
        offset = int(round((self.input_size - self.crop_size) / 2.0))
        return (
            offset,
            offset,
            offset + self.crop_size,
            offset + self.crop_size,
        )

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Equivalent of the official Resize/ToTensor/CenterCrop/Normalize."""
        resized = image.convert("RGB").resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        left, top, right, bottom = self._center_crop_box()
        tensor = tensor[:, top:bottom, left:right]
        return (tensor - self._normalization_mean) / self._normalization_std

    def _find_source_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.inpformer_root,
            os.environ.get("INPFORMER_ROOT"),
            self.kwargs.get("model_root"),
            "INP-Former",
            "../INP-Former",
            "../../INP-Former",
            str(harness_dir.parent / "INP-Former"),
            str(harness_dir.parent.parent / "INP-Former"),
            str(harness_dir.parent.parent.parent / "INP-Former"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if (
                (path / "models" / "uad.py").is_file()
                and (path / "dinov2" / "models" / "vision_transformer.py").is_file()
            ):
                return path.resolve()
        raise FileNotFoundError(
            "Could not locate the official INP-Former source. Pass "
            "inpformer_root=... or set INPFORMER_ROOT to a clone of "
            "https://github.com/luow23/INP-Former."
        )

    @staticmethod
    def _prepare_imports(root: Path) -> None:
        # The pinned upstream fork's xFormers branch returns only the attention
        # output, while its Block implementation unconditionally unpacks
        # ``(output, attention)``.  Its standard attention branch returns the
        # required pair and is the path used by the official environment.
        os.environ["XFORMERS_DISABLED"] = "1"
        root_string = str(root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)
        importlib.invalidate_caches()

        for module_name in list(sys.modules):
            if not (
                module_name == "models"
                or module_name.startswith("models.")
                or module_name == "dinov2"
                or module_name.startswith("dinov2.")
            ):
                continue
            # Always evict these top-level packages. In particular, a cached
            # official ``dinov2`` imported before XFORMERS_DISABLED was set
            # would retain its broken xFormers branch even though the source
            # path itself is correct.
            sys.modules.pop(module_name, None)

    def _validate_official_configuration(self) -> None:
        actual = (
            self.encoder_name,
            self.input_size,
            self.crop_size,
            self.inp_num,
        )
        expected = (
            self.OFFICIAL_ENCODER,
            self.OFFICIAL_INPUT_SIZE,
            self.OFFICIAL_CROP_SIZE,
            self.OFFICIAL_INP_NUM,
        )
        if actual != expected:
            raise ValueError(
                "Official few-shot checkpoints require encoder/input/crop/INP "
                f"configuration {expected}; got {actual}."
            )

    def load_model(self) -> None:
        """Construct the exact official architecture without downloading DINO weights.

        The official ``model.pth`` is a strict full-model state dict, including
        the frozen DINOv2 encoder.  Constructing the encoder directly avoids an
        unnecessary second backbone download while retaining identical final
        parameters.
        """
        self._validate_official_configuration()
        root = self._find_source_root()
        self._prepare_imports(root)

        uad_module = importlib.import_module("models.uad")
        blocks_module = importlib.import_module("models.vision_transformer")
        dino_module = importlib.import_module("dinov2.models.vision_transformer")

        encoder = dino_module.vit_base(
            patch_size=14,
            img_size=518,
            block_chunks=0,
            init_values=1e-8,
            num_register_tokens=4,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        embed_dim, num_heads = 768, 12
        bottleneck = nn.ModuleList([
            blocks_module.Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)
        ])
        prototype_token = nn.ParameterList([
            nn.Parameter(torch.randn(self.inp_num, embed_dim))
        ])
        aggregation = nn.ModuleList([
            blocks_module.Aggregation_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
        ])
        decoder = nn.ModuleList([
            blocks_module.Prototype_Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            for _ in range(8)
        ])
        self.model = uad_module.INP_Former(
            encoder=encoder,
            bottleneck=bottleneck,
            aggregation=aggregation,
            decoder=decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            remove_class_token=True,
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            prototype_token=prototype_token,
        )
        self.source_root = root
        self.source_commit = _resolve_git_commit(root)
        if (
            self.strict_source_commit
            and self.source_commit != OFFICIAL_INPFORMER_COMMIT
        ):
            raise RuntimeError(
                "Official source commit mismatch: expected "
                f"{OFFICIAL_INPFORMER_COMMIT}, found {self.source_commit}."
            )

    @staticmethod
    def _load_state_dict(checkpoint: Path) -> Mapping[str, torch.Tensor]:
        try:
            state_dict = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        except TypeError:
            state_dict = torch.load(checkpoint, map_location="cpu")
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Official checkpoint must contain a raw state dict: {checkpoint}"
            )
        if "state_dict" in state_dict or "model" in state_dict:
            raise ValueError(
                "Expected the official raw model.pth state dict, but found a "
                f"wrapped checkpoint at {checkpoint}."
            )
        return state_dict

    def prepare_for_dataset(self, dataset_name: str) -> None:
        if self.model is None:
            raise RuntimeError("load_model() must be called before selecting a checkpoint.")
        dataset = normalize_dataset_name(dataset_name)
        if dataset not in self.checkpoint_paths:
            raise FileNotFoundError(
                f"No {self.shot}-shot INP-Former checkpoint configured for "
                f"{dataset_name}. Expected checkpoint_paths['{dataset}']."
            )
        checkpoint = _resolve_checkpoint_path(self.checkpoint_paths[dataset])

        self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state_dict = self._load_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict
        self.model.to(self.device).eval()
        self.active_dataset = dataset
        self.active_checkpoint = checkpoint

    @staticmethod
    def _raw_anomaly_maps(
        encoder_features: Sequence[torch.Tensor],
        decoder_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        maps = [
            1.0 - F.cosine_similarity(encoder, decoder, dim=1)
            for encoder, decoder in zip(encoder_features, decoder_features)
        ]
        return torch.stack(maps, dim=1).mean(dim=1)

    def _postprocess_raw_maps(self, raw_maps: torch.Tensor) -> torch.Tensor:
        """Apply the official 392 -> 256 -> Gaussian test-time map path."""
        maps = raw_maps.unsqueeze(1)
        maps = F.interpolate(
            maps,
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=True,
        )
        maps = F.interpolate(
            maps,
            size=(self.OFFICIAL_METRIC_SIZE, self.OFFICIAL_METRIC_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        kernel = self._gaussian_kernel.to(device=maps.device, dtype=maps.dtype)
        return F.conv2d(maps, kernel, padding=kernel.shape[-1] // 2)[:, 0]

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.active_checkpoint is None:
            raise RuntimeError(
                "INP-Former requires prepare_for_dataset() before inference."
            )
        image_tensor = torch.stack([
            self._preprocess_image(image) for image in images
        ]).to(self.device)
        with torch.no_grad():
            encoder_features, decoder_features, _ = self.model(image_tensor)
            raw_maps = self._raw_anomaly_maps(
                encoder_features, decoder_features
            )
            metric_maps = self._postprocess_raw_maps(raw_maps)
            flattened = metric_maps.flatten(1)
            top_count = int(flattened.shape[1] * self.OFFICIAL_TOP_RATIO)
            scores = torch.sort(flattened, dim=1, descending=True)[0][
                :, :top_count
            ].mean(dim=1)
        return (
            scores.cpu().numpy().astype(np.float32),
            raw_maps.cpu().numpy().astype(np.float32),
        )

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        raw_map = torch.as_tensor(anomaly_map, dtype=torch.float32)[None]
        with torch.no_grad():
            metric_map = self._postprocess_raw_maps(raw_map)[0]
        return metric_map.cpu().numpy().astype(np.float32)

    def prepare_metric_mask(self, mask: np.ndarray) -> np.ndarray:
        binary_image = Image.fromarray(
            (np.asarray(mask) > 0).astype(np.uint8) * 255
        )
        resized = binary_image.resize(
            (self.input_size, self.input_size),
            resample=Image.Resampling.BILINEAR,
        )
        left, top, right, bottom = self._center_crop_box()
        cropped = resized.crop((left, top, right, bottom))
        mask_tensor = torch.from_numpy(
            np.asarray(cropped, dtype=np.float32) / 255.0
        )[None, None]
        mask_tensor = F.interpolate(
            mask_tensor,
            size=(self.OFFICIAL_METRIC_SIZE, self.OFFICIAL_METRIC_SIZE),
            mode="nearest",
        )
        return (mask_tensor[0, 0].numpy() > 0.5).astype(np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        checkpoint = self.active_checkpoint
        checkpoint_info: Dict[str, Any] = {"path": str(checkpoint) if checkpoint else None}
        if checkpoint and checkpoint.is_file():
            stat = checkpoint.stat()
            checkpoint_info.update({
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
            })
        return {
            "model": "INP-Former",
            "result_name": self.model_name,
            "shot": self.shot,
            "dataset_checkpoint_key": self.active_dataset,
            "checkpoint": checkpoint_info,
            "official_source_root": str(self.source_root) if self.source_root else None,
            "official_source_commit": self.source_commit,
            "expected_source_commit": OFFICIAL_INPFORMER_COMMIT,
            "encoder": self.encoder_name,
            "resize": self.input_size,
            "crop": self.crop_size,
            "inp_num": self.inp_num,
            "raw_map_size": 28,
            "metric_map_size": self.OFFICIAL_METRIC_SIZE,
            "gaussian_kernel": 5,
            "gaussian_sigma": 4,
            "image_score_top_ratio": self.OFFICIAL_TOP_RATIO,
            "xformers_disabled": True,
        }


class PromptADWrapper(BaseModelWrapper):
    """Paper/code-faithful inference for class-specific PromptAD buffers.

    PromptAD saves only two normal visual galleries and one learned text
    gallery.  The official test scripts load a separate CLS checkpoint for
    image scores and SEG checkpoint for pixel maps.  This wrapper keeps one
    frozen OpenCLIP backbone in memory and swaps those three small buffers
    between the two official inference passes.
    """

    OFFICIAL_BACKBONE = "ViT-B-16-plus-240"
    OFFICIAL_PRETRAINED_DATASET = "laion400m_e32"
    OFFICIAL_INPUT_SIZE = 240
    OFFICIAL_METRIC_SIZE = 400
    OFFICIAL_NORMAL_CONTEXT_TOKENS = 4
    OFFICIAL_ANOMALY_CONTEXT_TOKENS = 1
    OFFICIAL_ANOMALY_SUFFIXES = 4
    _CHECKPOINT_KEYS = {"feature_gallery1", "feature_gallery2", "text_features"}
    fail_on_inference_error = True

    def __init__(
        self,
        promptad_root: str = "",
        checkpoint_paths: Optional[
            Mapping[str, Mapping[str, Mapping[str, str]]]
        ] = None,
        shot: int = 1,
        backbone: str = OFFICIAL_BACKBONE,
        pretrained_dataset: str = OFFICIAL_PRETRAINED_DATASET,
        strict_source_commit: bool = False,
        emulate_official_bgr_test_path: bool = True,
        **kwargs: Any,
    ) -> None:
        if shot not in (1, 2, 4):
            raise ValueError("PromptAD few-shot buffers use shot=1, 2, or 4.")
        requested_device = str(kwargs.get("device", "cuda"))
        if not requested_device.startswith("cuda"):
            # The upstream model hard-codes fp16 inference, including on its
            # nominal CPU path. Keep failures early and explicit.
            raise ValueError("PromptAD's official fp16 inference path requires CUDA.")
        super().__init__(f"PromptAD-{shot}-shot", **kwargs)
        self.promptad_root = promptad_root
        self.checkpoint_paths = {
            normalize_dataset_name(dataset): {
                str(category): {
                    str(task).lower(): str(path)
                    for task, path in task_paths.items()
                }
                for category, task_paths in category_paths.items()
            }
            for dataset, category_paths in (checkpoint_paths or {}).items()
        }
        self.shot = shot
        self.backbone = backbone
        self.pretrained_dataset = pretrained_dataset
        self.strict_source_commit = strict_source_commit
        self.emulate_official_bgr_test_path = emulate_official_bgr_test_path
        self.source_root: Optional[Path] = None
        self.source_commit: Optional[str] = None
        self.active_dataset: Optional[str] = None
        self.active_category: Optional[str] = None
        self.active_checkpoints: Dict[str, Path] = {}
        self._promptad_class = None
        self._cls_state: Optional[Mapping[str, torch.Tensor]] = None
        self._seg_state: Optional[Mapping[str, torch.Tensor]] = None

    def _find_source_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.promptad_root,
            os.environ.get("PROMPTAD_ROOT"),
            self.kwargs.get("model_root"),
            "PromptAD",
            "../PromptAD",
            "../../PromptAD",
            str(harness_dir.parent / "PromptAD"),
            str(harness_dir.parent.parent / "PromptAD"),
            str(harness_dir.parent.parent.parent / "PromptAD"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if (
                (path / "PromptAD" / "model.py").is_file()
                and (path / "PromptAD" / "CLIPAD" / "factory.py").is_file()
            ):
                return path.resolve()
        raise FileNotFoundError(
            "Could not locate the official PromptAD source. Pass "
            "promptad_root=... or set PROMPTAD_ROOT to a clone of "
            "https://github.com/FuNz-0/PromptAD."
        )

    @staticmethod
    def _prepare_imports(root: Path) -> None:
        root_string = str(root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if module_name == "PromptAD" or module_name.startswith("PromptAD."):
                sys.modules.pop(module_name, None)

    def _validate_official_configuration(self) -> None:
        actual = (self.backbone, self.pretrained_dataset)
        expected = (self.OFFICIAL_BACKBONE, self.OFFICIAL_PRETRAINED_DATASET)
        if actual != expected:
            raise ValueError(
                f"PromptAD checkpoints require backbone/pretraining {expected}; "
                f"got {actual}."
            )

    def load_model(self) -> None:
        """Pin imports to the official source; construct CLIP lazily once."""
        self._validate_official_configuration()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PromptAD requires a CUDA accelerator because the official "
                "implementation is fp16-only."
            )
        root = self._find_source_root()
        self.source_root = root
        self.source_commit = _resolve_git_commit(root)
        if (
            self.strict_source_commit
            and self.source_commit != OFFICIAL_PROMPTAD_COMMIT
        ):
            raise RuntimeError(
                "Official PromptAD source commit mismatch: expected "
                f"{OFFICIAL_PROMPTAD_COMMIT}, found {self.source_commit}."
            )
        self._prepare_imports(root)
        try:
            module = importlib.import_module("PromptAD.model")
        except ImportError as exc:
            raise ImportError(
                "PromptAD source dependencies are incomplete. Install "
                "open_clip_torch, timm, ftfy, regex, scipy, and torchvision."
            ) from exc
        self._promptad_class = module.PromptAD

    def prepare_for_dataset(self, dataset_name: str) -> None:
        dataset = normalize_dataset_name(dataset_name)
        if dataset not in self.checkpoint_paths:
            raise FileNotFoundError(
                f"No {self.shot}-shot PromptAD checkpoint mapping for {dataset_name}."
            )
        expected_categories = set(PROMPTAD_DATASET_CATEGORIES[dataset])
        available_categories = set(self.checkpoint_paths[dataset])
        missing = sorted(expected_categories - available_categories)
        if missing:
            raise FileNotFoundError(
                f"PromptAD {dataset}/{self.shot}-shot mapping is missing classes: {missing}"
            )
        self.active_dataset = dataset
        self.active_category = None
        self.active_checkpoints = {}
        self._cls_state = None
        self._seg_state = None

    def _load_checkpoint_state(self, path: Path) -> Mapping[str, torch.Tensor]:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, Mapping) or set(state) != self._CHECKPOINT_KEYS:
            keys = sorted(state) if isinstance(state, Mapping) else type(state).__name__
            raise ValueError(
                f"Unexpected PromptAD inference-buffer checkpoint {path}: {keys}"
            )
        expected_gallery_shape = (self.shot * 15 * 15, 896)
        for key in ("feature_gallery1", "feature_gallery2"):
            if tuple(state[key].shape) != expected_gallery_shape:
                raise ValueError(
                    f"{path}: {key} shape {tuple(state[key].shape)} does not "
                    f"match {expected_gallery_shape}."
                )
        if tuple(state["text_features"].shape) != (2, 640):
            raise ValueError(
                f"{path}: text_features must have shape (2, 640), got "
                f"{tuple(state['text_features'].shape)}."
            )
        return state

    def _construct_model(self, category: str) -> None:
        if self._promptad_class is None:
            raise RuntimeError("load_model() must be called before PromptAD inference.")
        self.model = self._promptad_class(
            out_size_h=self.OFFICIAL_METRIC_SIZE,
            out_size_w=self.OFFICIAL_METRIC_SIZE,
            device=self.device,
            backbone=self.backbone,
            pretrained_dataset=self.pretrained_dataset,
            n_ctx=self.OFFICIAL_NORMAL_CONTEXT_TOKENS,
            n_pro=1,
            n_ctx_ab=self.OFFICIAL_ANOMALY_CONTEXT_TOKENS,
            n_pro_ab=self.OFFICIAL_ANOMALY_SUFFIXES,
            class_name=category,
            precision="fp16",
            k_shot=self.shot,
            img_resize=self.OFFICIAL_INPUT_SIZE,
            img_cropsize=self.OFFICIAL_INPUT_SIZE,
        )
        self.model = self.model.to(self.device)
        self.model.eval_mode()

    def _prepare_for_category(self, category: str) -> None:
        if self.active_dataset is None:
            raise RuntimeError(
                "PromptAD requires prepare_for_dataset() before class inference."
            )
        if category == self.active_category:
            return
        category_paths = self.checkpoint_paths[self.active_dataset].get(category)
        if not category_paths or set(category_paths) != {"cls", "seg"}:
            raise FileNotFoundError(
                "PromptAD requires paired CLS and SEG checkpoints for "
                f"{self.active_dataset}/{self.shot}-shot/{category}."
            )
        checkpoints = {
            task: Path(path).expanduser().resolve()
            for task, path in category_paths.items()
        }
        missing = [str(path) for path in checkpoints.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "PromptAD class checkpoints are missing:\n  - " + "\n  - ".join(missing)
            )
        if self.model is None:
            self._construct_model(category)
        self._cls_state = self._load_checkpoint_state(checkpoints["cls"])
        self._seg_state = self._load_checkpoint_state(checkpoints["seg"])
        self.active_category = category
        self.active_checkpoints = checkpoints

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        import cv2

        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.emulate_official_bgr_test_path:
            # Official CLIPDataset reads with cv2 (BGR), then test_cls/test_seg
            # call Image.fromarray without BGR->RGB conversion. Reproduce that
            # released evaluation path after applying corruptions in RGB space.
            pixels = pixels[..., ::-1].copy()
        # CLIPDataset first distorts every test image to 1024 x 1024 with
        # cv2.INTER_LINEAR; PromptAD.transform then resizes that image to 240.
        pixels = cv2.resize(pixels, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(pixels)
        return self.model.transform(image)

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not category:
            raise ValueError("PromptAD inference requires a dataset category name.")
        self._prepare_for_category(category)
        if self.model is None or self._cls_state is None or self._seg_state is None:
            raise RuntimeError("PromptAD class state was not initialized.")
        image_tensor = torch.stack([
            self._preprocess_image(image) for image in images
        ]).to(self.device)
        with torch.no_grad():
            self.model.load_state_dict(self._cls_state, strict=False)
            textual_scores, cls_visual_maps = self.model(image_tensor, "cls")
            self.model.load_state_dict(self._seg_state, strict=False)
            seg_visual_features = self.model.encode_image(image_tensor)
            seg_textual_maps = self.model.calculate_textual_anomaly_score(
                seg_visual_features, "seg"
            )
            seg_visual_maps = self.model.calculate_visual_anomaly_score(
                seg_visual_features
            )
            # This is the native 15 x 15 harmonic fusion immediately before
            # PromptAD.forward performs its 400-pixel interpolation/smoothing.
            pixel_maps = 1.0 / (1.0 / seg_textual_maps + 1.0 / seg_visual_maps)
        textual_scores = np.asarray(textual_scores, dtype=np.float32)
        cls_visual_maps = np.asarray(cls_visual_maps, dtype=np.float32)
        max_visual_scores = cls_visual_maps.reshape(len(images), -1).max(axis=1)
        # Exact metric_cal_img fusion from the released test_cls.py path.
        with np.errstate(divide="ignore", invalid="ignore"):
            image_scores = 1.0 / (
                1.0 / max_visual_scores + 1.0 / textual_scores
            )
        return (
            np.asarray(image_scores, dtype=np.float32),
            np.asarray(pixel_maps[:, 0].numpy(), dtype=np.float32),
        )

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        raw_map = np.asarray(anomaly_map, dtype=np.float32)
        if raw_map.shape != (15, 15):
            raise ValueError(
                f"PromptAD native SEG map must be 15x15, got {raw_map.shape}."
            )
        map_tensor = torch.from_numpy(raw_map)[None, None]
        resized = F.interpolate(
            map_tensor,
            size=(self.OFFICIAL_METRIC_SIZE, self.OFFICIAL_METRIC_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        return gaussian_filter(resized, sigma=4).astype(np.float32)

    def prepare_metric_mask(self, mask: np.ndarray) -> np.ndarray:
        # Match CLIPDataset's nearest 1024 resize and specify_resolution's
        # second nearest resize to the official 400-pixel metric space.
        import cv2

        binary = (np.asarray(mask) > 0).astype(np.uint8)
        resized = cv2.resize(binary, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        resized = cv2.resize(
            resized,
            (self.OFFICIAL_METRIC_SIZE, self.OFFICIAL_METRIC_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        return (resized > 0).astype(np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        selected = self.checkpoint_paths.get(self.active_dataset or "", {})
        checkpoint_pairs = {
            category: {
                task: {
                    "path": str(Path(path).expanduser()),
                    "size_bytes": (
                        Path(path).expanduser().stat().st_size
                        if Path(path).expanduser().is_file() else None
                    ),
                }
                for task, path in task_paths.items()
            }
            for category, task_paths in selected.items()
        }
        return {
            "model": "PromptAD",
            "result_name": self.model_name,
            "shot": self.shot,
            "dataset_checkpoint_key": self.active_dataset,
            "checkpoint_pair_count": len(checkpoint_pairs),
            "checkpoint_pairs": checkpoint_pairs,
            "official_source_root": str(self.source_root) if self.source_root else None,
            "official_source_commit": self.source_commit,
            "expected_source_commit": OFFICIAL_PROMPTAD_COMMIT,
            "backbone": self.backbone,
            "pretrained_dataset": self.pretrained_dataset,
            "resize": self.OFFICIAL_INPUT_SIZE,
            "crop": self.OFFICIAL_INPUT_SIZE,
            "dataset_preresize": 1024,
            "raw_map_size": 15,
            "metric_map_size": self.OFFICIAL_METRIC_SIZE,
            "gaussian_sigma": 4,
            "image_score_source": (
                "harmonic fusion of class-specific CLS text score and "
                "maximum CLS visual-map score"
            ),
            "pixel_map_source": "class-specific SEG text/visual harmonic fusion",
            "official_bgr_test_path": self.emulate_official_bgr_test_path,
        }

    def release(self) -> None:
        self._cls_state = None
        self._seg_state = None
        self._promptad_class = None
        super().release()


class APRILGANFewShotWrapper(APRILGANWrapper):
    """Official APRIL-GAN few-shot memory-bank inference.

    The released cross-dataset linear projections remain unchanged. Clean
    normal target images populate one patch-feature memory bank per category;
    nearest-neighbour distance maps are added to the zero-shot map. Image
    scores follow ``test.py`` and average the text score with a class/condition
    min-max-normalized maximum of the combined anomaly map.
    """

    condition_postprocessing = True

    def __init__(
        self,
        dataset_roots: Optional[Mapping[str, str]] = None,
        shot: int = 1,
        reference_seed: int = 42,
        few_shot_features: Optional[Sequence[int]] = None,
        result_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if shot not in (1, 2, 4):
            raise ValueError(
                "APRIL-GAN few-shot evaluation supports the benchmark's "
                "1-, 2-, and 4-shot settings."
            )
        super().__init__(
            result_name=result_name or f"APRIL-GAN-{shot}-shot",
            **kwargs,
        )
        self.dataset_roots = {
            self._normalize_dataset_name(dataset): str(path)
            for dataset, path in (dataset_roots or {}).items()
        }
        self.shot = int(shot)
        self.reference_seed = int(reference_seed)
        self.few_shot_features = list(
            self.OFFICIAL_FEATURES
            if few_shot_features is None
            else (
                (few_shot_features,)
                if isinstance(few_shot_features, int)
                else tuple(few_shot_features)
            )
        )
        self.support_paths: Dict[str, Tuple[Path, ...]] = {}
        self.active_category: Optional[str] = None
        self.memory_features: Sequence[torch.Tensor] = ()

    @staticmethod
    def _resolve_visa_image(root: Path, raw_path: str) -> Path:
        path = Path(raw_path.replace("\\", "/"))
        candidates = [path] if path.is_absolute() else [root / path]
        if not path.is_absolute() and path.parts and path.parts[0].lower() == "visa":
            candidates.append(root.joinpath(*path.parts[1:]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"VisA support image does not exist: {raw_path}")

    def _normal_training_paths(
        self, dataset: str, category: str
    ) -> Tuple[Path, ...]:
        root_value = self.dataset_roots.get(dataset)
        if not root_value:
            raise FileNotFoundError(
                f"No target root configured for APRIL-GAN few-shot dataset {dataset!r}."
            )
        root = Path(root_value).expanduser()
        if dataset == "mvtec":
            normal_root = root / category / "train" / "good"
            paths = (
                sorted(
                    path.resolve()
                    for path in normal_root.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
                )
                if normal_root.is_dir()
                else []
            )
        else:
            split_path = next(
                (
                    path
                    for path in (
                        root / "split_csv" / "1cls.csv",
                        root / "split_csv" / "test.csv",
                    )
                    if path.is_file()
                ),
                None,
            )
            if split_path is None:
                raise FileNotFoundError(
                    f"APRIL-GAN few-shot requires VisA split_csv/1cls.csv below {root}."
                )
            paths = []
            with split_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                reader = csv.reader(file_obj)
                header = next(reader, None)
                if header is None or len(header) < 4:
                    raise ValueError(f"Malformed VisA split CSV: {split_path}")
                for row in reader:
                    if len(row) < 4 or row[0] != category or row[1].lower() != "train":
                        continue
                    if row[2].strip().lower() not in {"normal", "good", "0", "false"}:
                        continue
                    paths.append(self._resolve_visa_image(root, row[3]))
        if not paths:
            raise ValueError(
                f"APRIL-GAN found no normal training images for {dataset}/{category}."
            )
        return tuple(paths)

    def _select_support_paths(self, dataset: str) -> Dict[str, Tuple[Path, ...]]:
        # This deliberately samples with replacement. It reproduces the
        # released dataset.py call to torch.randint rather than silently
        # changing the official support policy.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.reference_seed)
        selected: Dict[str, Tuple[Path, ...]] = {}
        for category in AFCLIP_DATASET_CATEGORIES[dataset]:
            candidates = self._normal_training_paths(dataset, category)
            indices = torch.randint(
                0,
                len(candidates),
                (self.shot,),
                generator=generator,
            ).tolist()
            selected[category] = tuple(candidates[index] for index in indices)
        return selected

    def load_model(self) -> None:
        if tuple(self.few_shot_features) != self.OFFICIAL_FEATURES:
            raise ValueError(
                "Official APRIL-GAN few-shot inference requires few_shot_features="
                f"{list(self.OFFICIAL_FEATURES)}, got {self.few_shot_features}."
            )
        super().load_model()

    def prepare_for_dataset(self, dataset_name: str) -> None:
        super().prepare_for_dataset(dataset_name)
        if self.active_target_dataset is None:
            raise RuntimeError("APRIL-GAN target dataset was not activated.")
        self.support_paths = self._select_support_paths(self.active_target_dataset)
        self.active_category = None
        self.memory_features = ()

    def _prepare_memory(self, category: str) -> None:
        if category == self.active_category and self.memory_features:
            return
        if category not in self.support_paths:
            raise KeyError(
                f"No APRIL-GAN support selection for "
                f"{self.active_target_dataset}/{category}."
            )
        support_images = []
        for path in self.support_paths[category]:
            with Image.open(path) as image:
                support_images.append(image.convert("RGB").copy())
        support_tensor = self._preprocess_batch(support_images)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=str(self.device).startswith("cuda")
        ):
            _, patch_tokens = self.model.encode_image(
                support_tensor, self.few_shot_features
            )
            memory_features = []
            for tokens in patch_tokens:
                tokens = tokens[:, 1:, :].reshape(-1, tokens.shape[-1])
                memory_features.append(F.normalize(tokens, dim=-1))
        self.memory_features = tuple(memory_features)
        self.active_category = category

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.active_checkpoint is None:
            raise RuntimeError("prepare_for_dataset() must run before APRIL-GAN inference.")
        if not category:
            raise ValueError("APRIL-GAN few-shot inference requires a category name.")
        self._prepare_memory(category)
        image_tensor = self._preprocess_batch(images)
        if image_tensor.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.image_size, self.image_size), dtype=np.float32),
            )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", enabled=str(self.device).startswith("cuda")
        ):
            image_scores, anomaly_map = self._zero_shot_forward_tensor(
                image_tensor, category
            )
            _, patch_tokens = self.model.encode_image(
                image_tensor, self.few_shot_features
            )
            for tokens, memory in zip(patch_tokens, self.memory_features):
                tokens = F.normalize(tokens[:, 1:, :], dim=-1)
                # The released code uses sklearn cosine similarity and takes
                # min(1 - similarity) over every support patch.
                nearest_distance = 1.0 - torch.matmul(
                    tokens, memory.transpose(0, 1)
                ).amax(dim=-1)
                side = int(round(math.sqrt(nearest_distance.shape[1])))
                if side * side != nearest_distance.shape[1]:
                    raise ValueError(
                        "APRIL-GAN few-shot patch count is not square: "
                        f"{nearest_distance.shape[1]}."
                    )
                distance_map = nearest_distance.reshape(-1, 1, side, side)
                anomaly_map = anomaly_map + F.interpolate(
                    distance_map,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
        return (
            image_scores.float().cpu().numpy().astype(np.float32),
            anomaly_map.float().cpu().numpy().astype(np.float32),
        )

    def postprocess_condition_outputs(
        self,
        scores: np.ndarray,
        anomaly_maps: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        scores_array = np.asarray(scores, dtype=np.float32)
        map_peaks = np.asarray(
            [np.asarray(anomaly_map).max() for anomaly_map in anomaly_maps],
            dtype=np.float32,
        )
        value_range = float(map_peaks.max() - map_peaks.min())
        if value_range > np.finfo(np.float32).eps:
            normalized_peaks = (map_peaks - map_peaks.min()) / value_range
        else:
            normalized_peaks = np.zeros_like(map_peaks)
        return 0.5 * (scores_array + normalized_peaks), anomaly_maps

    def inference_provenance(self) -> Dict[str, Any]:
        provenance = super().inference_provenance()
        provenance.update(
            {
                "result_name": self.model_name,
                "shot": self.shot,
                "reference_seed": self.reference_seed,
                "few_shot_feature_layers": self.few_shot_features,
                "support_sampling": "torch.randint with replacement",
                "support_images": {
                    category: [str(path) for path in paths]
                    for category, paths in self.support_paths.items()
                },
                "memory_distance": "minimum cosine distance over support patches",
                "few_shot_map_fusion": "zero-shot map + four memory-distance maps",
                "few_shot_image_score": (
                    "0.5 * (text score + class/condition-normalized map maximum)"
                ),
            }
        )
        return provenance

    def release(self) -> None:
        self.support_paths = {}
        self.active_category = None
        self.memory_features = ()
        super().release()


class AFCLIPFewShotWrapper(BaseModelWrapper):
    """Official AF-CLIP+ inference with target-normal memory banks.

    The released MVTec/VisA prompt and adaptor weights remain zero-shot and
    cross-dataset.  For each target class, AF-CLIP+ samples ``shot`` clean
    training images, stores their multi-level spatially aggregated patch
    features, and adds the nearest-neighbour score to 0.1 times AF-CLIP's
    learned prompt score.  No few-shot optimization is performed.
    """

    OFFICIAL_MODEL = "ViT-L/14@336px"
    OFFICIAL_INPUT_SIZE = 518
    OFFICIAL_PROMPT_LENGTH = 12
    OFFICIAL_FEATURE_LAYERS = (6, 12, 18, 24)
    OFFICIAL_MEMORY_LAYERS = (6, 12, 18, 24)
    OFFICIAL_ALPHA = 0.1
    OFFICIAL_GAUSSIAN_SIGMA = 4.0
    fail_on_inference_error = True

    def __init__(
        self,
        afclip_root: str = "",
        checkpoint_paths: Optional[Mapping[str, Mapping[str, str]]] = None,
        dataset_roots: Optional[Mapping[str, str]] = None,
        shot: int = 1,
        reference_seed: int = 111,
        result_name: Optional[str] = None,
        clip_download_dir: str = "",
        strict_source_commit: bool = True,
        **kwargs: Any,
    ) -> None:
        if shot not in (1, 2, 4):
            raise ValueError("AF-CLIP+ supports the official 1-, 2-, and 4-shot settings.")
        super().__init__(result_name or f"AF-CLIP+-{shot}-shot", **kwargs)
        self.afclip_root = afclip_root
        self.checkpoint_paths = {
            normalize_dataset_name(dataset): {
                str(component).lower(): str(path)
                for component, path in components.items()
            }
            for dataset, components in (checkpoint_paths or {}).items()
        }
        self.dataset_roots = {
            normalize_dataset_name(dataset): str(path)
            for dataset, path in (dataset_roots or {}).items()
        }
        self.shot = shot
        self.reference_seed = int(reference_seed)
        self.clip_download_dir = clip_download_dir
        self.strict_source_commit = strict_source_commit
        self.source_root: Optional[Path] = None
        self.source_commit: Optional[str] = None
        self.active_dataset: Optional[str] = None
        self.active_source_dataset: Optional[str] = None
        self.active_checkpoints: Dict[str, Path] = {}
        self.active_category: Optional[str] = None
        self.support_paths: Dict[str, Tuple[Path, ...]] = {}
        self._clip_tokenize = None
        self._args = None
        self._normalization_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
        )[:, None, None]
        self._normalization_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
        )[:, None, None]

    def _find_source_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.afclip_root,
            os.environ.get("AFCLIP_ROOT"),
            self.kwargs.get("model_root"),
            "AF-CLIP",
            "../AF-CLIP",
            str(harness_dir.parent / "AF-CLIP"),
            str(harness_dir.parent.parent / "AF-CLIP"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate)).expanduser()
            if (
                (path / "clip" / "clip.py").is_file()
                and (path / "clip" / "model.py").is_file()
            ):
                return path.resolve()
        raise FileNotFoundError(
            "Could not locate the official AF-CLIP source. Pass afclip_root=... "
            "or set AFCLIP_ROOT to a clone of Faustinaqq/AF-CLIP."
        )

    @staticmethod
    def _prepare_imports(root: Path) -> None:
        root_string = str(root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)
        importlib.invalidate_caches()
        # AF-CLIP vendors a modified top-level package named ``clip``.  Evict
        # any OpenAI/OpenCLIP module imported by another wrapper first.
        for module_name in list(sys.modules):
            if module_name == "clip" or module_name.startswith("clip."):
                sys.modules.pop(module_name, None)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (self.OFFICIAL_INPUT_SIZE, self.OFFICIAL_INPUT_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return (tensor - self._normalization_mean) / self._normalization_std

    def load_model(self) -> None:
        root = self._find_source_root()
        self._prepare_imports(root)
        clip_module = importlib.import_module("clip.clip")
        download_root = self.clip_download_dir or str(root / "download" / "clip")
        self.model, _ = clip_module.load(
            name=self.OFFICIAL_MODEL,
            jit=False,
            device=self.device,
            download_root=download_root,
        )
        self._args = type("AFCLIPArgs", (), {
            "prompt_len": self.OFFICIAL_PROMPT_LENGTH,
            "feature_layers": list(self.OFFICIAL_FEATURE_LAYERS),
            "memory_layers": list(self.OFFICIAL_MEMORY_LAYERS),
            "alpha": self.OFFICIAL_ALPHA,
        })()
        self.model.insert(
            args=self._args,
            tokenizer=clip_module.tokenize,
            device=self.device,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.source_root = root
        self.source_commit = _resolve_git_commit(root)
        if self.strict_source_commit and self.source_commit != OFFICIAL_AFCLIP_COMMIT:
            raise RuntimeError(
                "Official AF-CLIP source commit mismatch: expected "
                f"{OFFICIAL_AFCLIP_COMMIT}, found {self.source_commit}."
            )

    @staticmethod
    def _torch_load_trusted(path: Path, map_location: str = "cpu") -> Any:
        # The official adaptor is serialized as a Python nn.Module, not a raw
        # state dict.  Only load files discovered from the pinned repository.
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _resolve_visa_image(root: Path, raw_path: str) -> Path:
        path = Path(raw_path.replace("\\", "/"))
        candidates = [path] if path.is_absolute() else [root / path]
        if not path.is_absolute() and path.parts and path.parts[0].lower() == "visa":
            candidates.append(root.joinpath(*path.parts[1:]))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"VisA support image does not exist: {raw_path}")

    def _normal_training_paths(self, dataset: str, category: str) -> Tuple[Path, ...]:
        if dataset not in self.dataset_roots:
            raise FileNotFoundError(
                f"No target root configured for AF-CLIP+ dataset {dataset!r}."
            )
        root = Path(self.dataset_roots[dataset]).expanduser()
        if dataset == "mvtec":
            normal_root = root / category / "train" / "good"
            paths = sorted(
                path.resolve() for path in normal_root.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            ) if normal_root.is_dir() else []
        else:
            split_candidates = (
                root / "split_csv" / "1cls.csv",
                root / "split_csv" / "test.csv",
            )
            split_path = next((path for path in split_candidates if path.is_file()), None)
            if split_path is None:
                raise FileNotFoundError(
                    f"AF-CLIP+ requires VisA's split_csv/1cls.csv below {root}."
                )
            paths = []
            with split_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
                reader = csv.reader(file_obj)
                header = next(reader, None)
                if header is None or len(header) < 4:
                    raise ValueError(f"Malformed VisA split CSV: {split_path}")
                for row in reader:
                    if len(row) < 4 or row[0] != category or row[1].lower() != "train":
                        continue
                    if row[2].strip().lower() not in {"normal", "good", "0", "false"}:
                        continue
                    paths.append(self._resolve_visa_image(root, row[3]))
            paths = sorted(set(paths))
        if len(paths) < self.shot:
            raise ValueError(
                f"AF-CLIP+ needs {self.shot} normal training images for "
                f"{dataset}/{category}, found {len(paths)}."
            )
        return tuple(paths)

    def _select_support_paths(self, dataset: str) -> Dict[str, Tuple[Path, ...]]:
        rng = np.random.RandomState(self.reference_seed)
        selected: Dict[str, Tuple[Path, ...]] = {}
        for category in AFCLIP_DATASET_CATEGORIES[dataset]:
            candidates = self._normal_training_paths(dataset, category)
            indices = rng.choice(len(candidates), size=self.shot, replace=False)
            selected[category] = tuple(candidates[int(index)] for index in indices)
        return selected

    def prepare_for_dataset(self, dataset_name: str) -> None:
        if self.model is None:
            raise RuntimeError("load_model() must be called before selecting AF-CLIP weights.")
        target_dataset = normalize_dataset_name(dataset_name)
        source_dataset = "visa" if target_dataset == "mvtec" else "mvtec"
        components = self.checkpoint_paths.get(source_dataset, {})
        if set(components) != {"prompt", "adaptor"}:
            raise FileNotFoundError(
                f"AF-CLIP+ target {target_dataset} needs the released "
                f"{source_dataset} prompt and adaptor weights."
            )
        paths = {name: Path(path).expanduser().resolve() for name, path in components.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing AF-CLIP checkpoint files: " + ", ".join(missing))

        prompt_object = self._torch_load_trusted(paths["prompt"])
        prompt = prompt_object
        if isinstance(prompt, Mapping):
            prompt = next(
                (
                    prompt[key]
                    for key in ("state_prompt_embedding", "prompt", "weight")
                    if key in prompt
                ),
                prompt,
            )
        if not isinstance(prompt, torch.Tensor):
            raise TypeError(f"AF-CLIP prompt checkpoint must be a tensor: {paths['prompt']}")
        expected_prompt_shape = tuple(self.model.state_prompt_embedding.shape)
        if tuple(prompt.shape) != expected_prompt_shape:
            raise ValueError(
                f"AF-CLIP prompt shape mismatch: expected {expected_prompt_shape}, "
                f"got {tuple(prompt.shape)} from {paths['prompt']}."
            )
        prompt = nn.Parameter(
            prompt.detach().to(
                device=self.device,
                dtype=self.model.token_embedding.weight.dtype,
            ),
            requires_grad=False,
        )
        adaptor_object = self._torch_load_trusted(paths["adaptor"])
        if isinstance(adaptor_object, nn.Module):
            adaptor = adaptor_object
        else:
            state_dict = adaptor_object
            if isinstance(state_dict, Mapping):
                for key in ("adaptor", "state_dict"):
                    if isinstance(state_dict.get(key), Mapping):
                        state_dict = state_dict[key]
                        break
            if not isinstance(state_dict, Mapping):
                raise TypeError(
                    "AF-CLIP adaptor checkpoint must contain an nn.Module or "
                    f"state dict: {paths['adaptor']}"
                )
            adaptor = self.model.adaptor
            adaptor.load_state_dict(state_dict, strict=True)
        self.model.state_prompt_embedding = prompt
        self.model.adaptor = adaptor.to(self.device).eval()
        for parameter in self.model.adaptor.parameters():
            parameter.requires_grad_(False)
        self.model.memorybank = None

        self.active_dataset = target_dataset
        self.active_source_dataset = source_dataset
        self.active_checkpoints = paths
        self.active_category = None
        self.support_paths = self._select_support_paths(target_dataset)

    def _prepare_for_category(self, category: str) -> None:
        if self.model is None or self.active_dataset is None or self._args is None:
            raise RuntimeError("AF-CLIP+ requires prepare_for_dataset() before inference.")
        if category == self.active_category and self.model.memorybank is not None:
            return
        if category not in self.support_paths:
            raise KeyError(f"No AF-CLIP+ support selection for {self.active_dataset}/{category}.")
        support_images = []
        for path in self.support_paths[category]:
            with Image.open(path) as image:
                support_images.append(self._preprocess_image(image.copy()))
        support_tensor = torch.stack(support_images).to(self.device)
        with torch.no_grad():
            self.model.store_memory(support_tensor, self._args)
        self.active_category = category

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not category:
            raise ValueError("AF-CLIP+ inference requires a category name.")
        self._prepare_for_category(category)
        image_tensor = torch.stack([self._preprocess_image(image) for image in images]).to(self.device)
        with torch.no_grad():
            scores, maps = self.model.detect_forward(image_tensor, self._args)
        return (
            scores.detach().cpu().numpy().astype(np.float32),
            maps[:, 0].detach().cpu().numpy().astype(np.float32),
        )

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        raw_map = torch.as_tensor(anomaly_map, dtype=torch.float32)[None, None]
        resized = F.interpolate(
            raw_map,
            size=(self.OFFICIAL_INPUT_SIZE, self.OFFICIAL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        return gaussian_filter(resized, sigma=self.OFFICIAL_GAUSSIAN_SIGMA).astype(np.float32)

    def prepare_metric_mask(self, mask: np.ndarray) -> np.ndarray:
        # The released target transform resizes masks with bicubic interpolation
        # and evaluation thresholds them at 0.5.
        binary_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
        resized = binary_image.resize(
            (self.OFFICIAL_INPUT_SIZE, self.OFFICIAL_INPUT_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
        return (np.asarray(resized, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        checkpoint_info = {
            component: {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for component, path in self.active_checkpoints.items()
        }
        return {
            "model": "AF-CLIP+",
            "result_name": self.model_name,
            "shot": self.shot,
            "reference_seed": self.reference_seed,
            "target_dataset": self.active_dataset,
            "source_checkpoint_dataset": self.active_source_dataset,
            "checkpoint": checkpoint_info,
            "support_images": {
                category: [str(path) for path in paths]
                for category, paths in self.support_paths.items()
            },
            "official_source_root": str(self.source_root) if self.source_root else None,
            "official_source_commit": self.source_commit,
            "expected_source_commit": OFFICIAL_AFCLIP_COMMIT,
            "backbone": self.OFFICIAL_MODEL,
            "input_size": self.OFFICIAL_INPUT_SIZE,
            "feature_layers": list(self.OFFICIAL_FEATURE_LAYERS),
            "memory_layers": list(self.OFFICIAL_MEMORY_LAYERS),
            "spatial_aggregation_kernels": [1, 3, 5],
            "prompt_weight": self.OFFICIAL_ALPHA,
            "raw_map_size": 37,
            "metric_map_size": self.OFFICIAL_INPUT_SIZE,
            "gaussian_sigma": self.OFFICIAL_GAUSSIAN_SIGMA,
            "support_policy": "normal target training images; no corruption",
        }

    def release(self) -> None:
        self.support_paths = {}
        self.active_checkpoints = {}
        self.active_category = None
        super().release()


MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str, wrapper_class: type) -> None:
    """Register a future few-shot wrapper with the common harness."""
    if not issubclass(wrapper_class, BaseModelWrapper):
        raise TypeError("Few-shot wrappers must subclass BaseModelWrapper.")
    MODEL_REGISTRY[name] = wrapper_class


register_model("INP-Former", INPFormerWrapper)
register_model("PromptAD", PromptADWrapper)
register_model("AF-CLIP+", AFCLIPFewShotWrapper)
register_model("APRIL-GAN", APRILGANFewShotWrapper)


def get_model(
    model_name: str, device: str = "cuda", **kwargs: Any
) -> BaseModelWrapper:
    """Instantiate a registered few-shot model wrapper."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown few-shot model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_name](device=device, **kwargs)


__all__ = [
    "AFCLIPFewShotWrapper",
    "APRILGANFewShotWrapper",
    "AFCLIP_DATASET_CATEGORIES",
    "INPFormerWrapper",
    "PromptADWrapper",
    "MODEL_REGISTRY",
    "OFFICIAL_CHECKPOINT_URLS",
    "OFFICIAL_INPFORMER_COMMIT",
    "OFFICIAL_PROMPTAD_COMMIT",
    "OFFICIAL_AFCLIP_COMMIT",
    "OFFICIAL_APRILGAN_COMMIT",
    "PROMPTAD_DATASET_CATEGORIES",
    "discover_official_checkpoints",
    "discover_promptad_checkpoints",
    "discover_afclip_checkpoints",
    "discover_aprilgan_checkpoints",
    "get_model",
    "normalize_dataset_name",
    "official_checkpoint_directory",
    "register_model",
]
