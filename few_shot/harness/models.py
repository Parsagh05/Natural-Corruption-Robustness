"""Few-shot model wrappers.

INP-Former is constructed from the official source tree and evaluated with the
official few-shot checkpoint and test-time scoring path.  New few-shot models
can be added with :func:`register_model` without changing the runner.
"""

from functools import partial
import importlib
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

from zero_shot.harness.models import BaseModelWrapper


OFFICIAL_INPFORMER_COMMIT = "17d265381d9b323a2ef6e05aab0665a85edebe84"
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

    resolved: Dict[int, Dict[str, str]] = {}
    missing = []
    all_model_files = list(root.rglob("model.pth"))
    for shot in selected_shots:
        resolved[shot] = {}
        for dataset in ("mvtec", "visa"):
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


MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str, wrapper_class: type) -> None:
    """Register a future few-shot wrapper with the common harness."""
    if not issubclass(wrapper_class, BaseModelWrapper):
        raise TypeError("Few-shot wrappers must subclass BaseModelWrapper.")
    MODEL_REGISTRY[name] = wrapper_class


register_model("INP-Former", INPFormerWrapper)


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
    "INPFormerWrapper",
    "MODEL_REGISTRY",
    "OFFICIAL_CHECKPOINT_URLS",
    "OFFICIAL_INPFORMER_COMMIT",
    "discover_official_checkpoints",
    "get_model",
    "normalize_dataset_name",
    "official_checkpoint_directory",
    "register_model",
]
