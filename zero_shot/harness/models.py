# -*- coding: utf-8 -*-
"""
models.py - Model registry and wrapper with .forward_raw() extraction.

Each model wrapper intercepts the inference pipeline to extract:
  1. Raw image-level anomaly score (before any normalization).
  2. Low-resolution anomaly map tensor (before final bilinear upsampling).
"""

from abc import ABC, abstractmethod
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Tuple, Optional, Dict, Any, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .aaclip_scoring import postprocess_aaclip_industrial_condition


def _as_tuple(value: Any, default: Sequence[int]) -> Tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, int):
        return (value,)
    return tuple(value)


def _resolve_existing_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    return path if path.exists() else None


class BaseModelWrapper(ABC):
    """
    Abstract base class for all VLM anomaly detection model wrappers.

    Every model wrapper MUST implement:
        - load_model(): Initialize model weights and components.
        - forward_raw(image): Return (anomaly_score, lowres_anomaly_map).
    """

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.kwargs = kwargs

    # All existing wrappers keep the harness's historical interpolation rule.
    # AA-CLIP overrides this because its official test code uses True.
    metric_map_align_corners: bool = False
    condition_postprocessing: bool = False
    fail_on_inference_error: bool = False

    def prepare_for_dataset(self, dataset_name: str) -> None:
        """Activate dataset-specific state before evaluating a dataset.

        Zero-shot models normally keep one checkpoint for every target
        dataset, so the default is intentionally a no-op.  Few-shot wrappers
        can override this hook to switch to the matching official checkpoint
        without duplicating the evaluation runner.
        """

    def prepare_metric_mask(self, mask: np.ndarray) -> np.ndarray:
        """Transform a binary mask into the model's metric coordinate space.

        The historical zero-shot protocol evaluates at 518 x 518.  Models
        whose official test code uses a different resize/crop pipeline can
        override this method.
        """
        mask_tensor = torch.from_numpy(np.asarray(mask)).float()[None, None]
        resized = F.interpolate(mask_tensor, size=(518, 518), mode="nearest")
        return (resized[0, 0].cpu().numpy() > 0.5).astype(np.float32)

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Transform a stored raw map into the model's metric map.

        This default exactly preserves the existing zero-shot bilinear
        interpolation behavior.  A model may override it when its paper uses
        additional smoothing or a native evaluation resolution.
        """
        map_tensor = torch.from_numpy(np.asarray(anomaly_map)).float()[None, None]
        resized = F.interpolate(
            map_tensor,
            size=(518, 518),
            mode="bilinear",
            align_corners=self.metric_map_align_corners,
        )
        return resized[0, 0].cpu().numpy().astype(np.float32)

    def prepare_artifact_maps(self, anomaly_maps: np.ndarray) -> np.ndarray:
        """Return maps to serialize after metric inputs have been retained.

        Most wrappers already return token-resolution maps, so preserving the
        input is the correct default. Models that expose only full-resolution
        predictions can override this hook to avoid oversized artifact files.
        """
        return anomaly_maps

    def inference_provenance(self) -> Dict[str, Any]:
        """Return JSON-serializable model provenance for run manifests."""
        return {"model": self.model_name}

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize inference components."""
        pass

    @abstractmethod
    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        """
        Run inference and extract raw outputs BEFORE final upsampling.

        Args:
            image: PIL RGB image (corrupted or clean).
            category: Category name for prompt-based models.

        Returns:
            anomaly_score: float - Image-level anomaly score.
            lowres_map: np.ndarray of shape (H_low, W_low) - Raw anomaly map
                        at token/patch resolution (e.g., 14x14 or 24x24).
        """
        pass

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batched inference hook. Wrappers can override this for efficient GPU
        batching; the default preserves compatibility by looping one image at
        a time.
        """
        scores, lowres_maps = [], []
        for image in images:
            score, lowres_map = self.forward_raw(image, category=category)
            scores.append(score)
            lowres_maps.append(lowres_map)
        return np.asarray(scores, dtype=np.float32), np.stack(lowres_maps)

    def postprocess_condition_outputs(
        self,
        scores: np.ndarray,
        anomaly_maps: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        """Finalize scores after all samples from one class are available.

        Most models require no condition-level processing.  AA-CLIP overrides
        this hook because its official industrial inference normalizes across
        a whole class/condition and fuses image and pixel-map predictions.
        """
        return scores, anomaly_maps

    def release(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Model Wrappers
# ─────────────────────────────────────────────────────────────────────────────


class VCPCLIPWrapper(BaseModelWrapper):
    """Wrapper for VCP-CLIP: Vision-language Contrastive Prompting."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("VCP-CLIP", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        # Import model-specific code
        # from vcp_clip import VCPCLIPModel  # noqa: E402
        # self.model = VCPCLIPModel.load(self.checkpoint_path)
        # self.model.to(self.device).eval()
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        """
        Intercept VCP-CLIP to get pre-upsampled patch similarities.
        The model internally computes token-level similarity maps at
        (H/patch_size, W/patch_size) resolution before F.interpolate.
        """
        # Placeholder implementation structure:
        # with torch.no_grad():
        #     preprocessed = self.model.preprocess(image).unsqueeze(0).to(self.device)
        #     # Hook into the model before the final upsample
        #     features = self.model.encode_image(preprocessed)
        #     text_features = self.model.encode_text(category)
        #     # Compute patch-level similarities (raw low-res map)
        #     patch_sims = self.model.compute_patch_similarity(features, text_features)
        #     # patch_sims shape: (1, H_low, W_low)
        #     lowres_map = patch_sims.squeeze(0).cpu().numpy()
        #     anomaly_score = lowres_map.max()
        #     return float(anomaly_score), lowres_map

        # Stub return for framework validation
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class CraneWrapper(BaseModelWrapper):
    """Wrapper for Crane model."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("Crane", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class FAPromptWrapper(BaseModelWrapper):
    """Wrapper for FAPrompt model."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("FAPrompt", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class AnomalyCLIPWrapper(BaseModelWrapper):
    """Wrapper for AnomalyCLIP model."""

    def __init__(
        self,
        checkpoint_path: str = "",
        anomalyclip_root: str = "",
        image_size: int = 518,
        features_list: Optional[Sequence[int]] = None,
        feature_map_layer: Optional[Sequence[int]] = None,
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        dpam_layer: int = 20,
        clip_download_root: str = "",
        **kwargs,
    ):
        super().__init__("AnomalyCLIP", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.anomalyclip_root = anomalyclip_root
        self.image_size = image_size
        self.features_list = _as_tuple(features_list, (6, 12, 18, 24))
        self.feature_map_layer = _as_tuple(feature_map_layer, (0, 1, 2, 3))
        self.depth = depth
        self.n_ctx = n_ctx
        self.t_n_ctx = t_n_ctx
        self.dpam_layer = dpam_layer
        self.clip_download_root = clip_download_root
        self.preprocess = None
        self.prompt_learner = None
        self.text_features = None
        self._anomalyclip_lib = None

    def load_model(self) -> None:
        root = self._find_anomalyclip_root()
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()

        for module_name in ("utils", "prompt_ensemble"):
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", "") if module else ""
            if module and not str(module_file).startswith(root_str):
                sys.modules.pop(module_name, None)

        try:
            AnomalyCLIP_lib = importlib.import_module("AnomalyCLIP_lib")
            prompt_module = importlib.import_module("prompt_ensemble")
            utils_module = importlib.import_module("utils")
        except ImportError as exc:
            raise ImportError(
                f"Failed to import AnomalyCLIP from {root}. The source "
                "directory exists, but one of its Python dependencies or "
                f"modules failed to import. Original error: {exc!r}"
            ) from exc
        AnomalyCLIP_PromptLearner = prompt_module.AnomalyCLIP_PromptLearner
        get_transform = utils_module.get_transform

        checkpoint_path = self._find_checkpoint(root)
        parameters = {
            "Prompt_length": self.n_ctx,
            "learnabel_text_embedding_depth": self.depth,
            "learnabel_text_embedding_length": self.t_n_ctx,
        }

        load_kwargs = {
            "device": self.device,
            "design_details": parameters,
            "download_root": str(
                Path(
                    self.clip_download_root
                    or os.environ.get("ANOMALYCLIP_CLIP_CACHE", "")
                    or Path.home() / ".cache" / "clip"
                ).expanduser()
            ),
        }

        model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", **load_kwargs)
        model.eval()

        transform_args = SimpleNamespace(image_size=self.image_size)
        self.preprocess, _ = get_transform(transform_args)

        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device if torch.cuda.is_available() else "cpu",
        )
        if "prompt_learner" not in checkpoint:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'prompt_learner'."
            )
        prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        prompt_learner.to(self.device)

        model.to(self.device)
        model.visual.DAPM_replace(DPAM_layer=self.dpam_layer)

        with torch.no_grad():
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(
                cls_id=None
            )
            text_features = model.encode_text_learn(
                prompts, tokenized_prompts, compound_prompts_text
            ).float()
            text_features = torch.stack(
                torch.chunk(text_features, dim=0, chunks=2), dim=1
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

        self.model = model
        self.prompt_learner = prompt_learner
        self.text_features = text_features
        self._anomalyclip_lib = AnomalyCLIP_lib

    def _find_anomalyclip_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.anomalyclip_root,
            os.environ.get("ANOMALYCLIP_ROOT"),
            self.kwargs.get("model_root"),
            "AnomalyCLIP",
            "../AnomalyCLIP",
            "../../AnomalyCLIP",
            str(harness_dir.parent / "AnomalyCLIP"),
            str(harness_dir.parent.parent / "AnomalyCLIP"),
            str(harness_dir.parent.parent.parent / "AnomalyCLIP"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and (path / "AnomalyCLIP_lib").exists():
                return path
        raise FileNotFoundError(
            "Could not locate the AnomalyCLIP source directory. Pass "
            "anomalyclip_root=... or set ANOMALYCLIP_ROOT to a clone of "
            "https://github.com/zqhang/AnomalyCLIP."
        )

    def _find_checkpoint(self, root: Path) -> Path:
        path = _resolve_existing_path(self.checkpoint_path)
        if path and path.is_file():
            return path
        if path and path.is_dir():
            epoch_path = path / "epoch_15.pth"
            if epoch_path.exists():
                return epoch_path
            pth_files = sorted(path.glob("*.pth"))
            if pth_files:
                return pth_files[-1]

        env_path = _resolve_existing_path(os.environ.get("ANOMALYCLIP_CHECKPOINT"))
        if env_path and env_path.is_file():
            return env_path

        defaults = [
            root / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth",
            root / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth",
        ]
        for candidate in defaults:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Could not locate AnomalyCLIP checkpoint. Pass checkpoint_path=... "
            "or set ANOMALYCLIP_CHECKPOINT."
        )

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, lowres_maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), lowres_maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.preprocess is None or self.text_features is None:
            raise RuntimeError("AnomalyCLIP model is not loaded. Call load_model() first.")
        if not images:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("AnomalyCLIPWrapper.forward_raw_batch expects PIL images.")

        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device, non_blocking=True)

        with torch.inference_mode():
            image_features, patch_features = self.model.encode_image(
                image_tensor,
                list(self.features_list),
                DPAM_layer=self.dpam_layer,
            )
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_probs = image_features @ self.text_features[0].T
            text_probs = (text_probs / 0.07).softmax(-1)
            anomaly_scores = text_probs[:, 1]

            lowres_maps = []
            selected_layers = set(self.feature_map_layer)
            for idx, patch_feature in enumerate(patch_features):
                if idx not in selected_layers:
                    continue
                patch_feature = patch_feature / patch_feature.norm(
                    dim=-1, keepdim=True
                )
                similarity, _ = self._anomalyclip_lib.compute_similarity(
                    patch_feature, self.text_features[0]
                )
                patch_similarity = similarity[:, 1:, :]
                side = int(patch_similarity.shape[1] ** 0.5)
                if side * side != patch_similarity.shape[1]:
                    raise ValueError(
                        "AnomalyCLIP patch tokens do not form a square map: "
                        f"{patch_similarity.shape[1]} tokens."
                    )
                similarity_map = patch_similarity.reshape(
                    patch_similarity.shape[0], side, side, -1
                )
                anomaly_map = (
                    similarity_map[..., 1] + 1 - similarity_map[..., 0]
                ) / 2.0
                lowres_maps.append(anomaly_map)

            if not lowres_maps:
                raise RuntimeError("No AnomalyCLIP patch features were selected.")

            lowres_map = torch.stack(lowres_maps).sum(dim=0)

        return (
            anomaly_scores.detach().cpu().numpy().astype(np.float32),
            lowres_map.detach().cpu().numpy().astype(np.float32),
        )


class AdaCLIPWrapper(BaseModelWrapper):
    """Wrapper for AdaCLIP model."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("AdaCLIP", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class AACLIPWrapper(BaseModelWrapper):
    """Wrapper for AA-CLIP model."""

    metric_map_align_corners = True
    condition_postprocessing = True

    _REAL_NAMES: Dict[str, str] = {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    }
    _PROMPT_NORMAL = ["{}", "a {}", "the {}"]
    _PROMPT_ABNORMAL = [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ]
    _PROMPT_TEMPLATES = ["{}.", "a photo of {}."]

    def __init__(
        self,
        checkpoint_path: str = "",
        aaclip_root: str = "",
        clip_weight_path: str = "",
        clip_model_name: str = "ViT-L-14-336",
        image_size: int = 518,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.1,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: Optional[Sequence[int]] = None,
        relu: bool = False,
        apply_score_blur: bool = True,
        **kwargs,
    ):
        super().__init__("AA-CLIP", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.aaclip_root = aaclip_root
        self.clip_weight_path = clip_weight_path
        self.clip_model_name = clip_model_name
        self.image_size = image_size
        self.text_adapt_weight = text_adapt_weight
        self.image_adapt_weight = image_adapt_weight
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.levels = list(_as_tuple(levels, (6, 12, 18, 24)))
        self.relu = relu
        self.apply_score_blur = apply_score_blur
        self.clip_model = None
        self.preprocess = None
        self.tokenize = None
        self.gaussian_blur2d = None
        self.text_encoder = None
        self.adapt_text = False
        self._text_feature_cache: Dict[str, torch.Tensor] = {}

    def load_model(self) -> None:
        root = self._find_aaclip_root()
        self._prepare_imports(root)

        try:
            clip_module = importlib.import_module("model.clip")
            adapter_module = importlib.import_module("model.adapter")
            tokenizer_module = importlib.import_module("model.tokenizer")
            from torchvision import transforms
            from kornia.filters import gaussian_blur2d
        except ImportError as exc:
            raise ImportError(
                f"Failed to import AA-CLIP from {root}. Install the official "
                "requirements and ensure the cloned repository is complete. "
                f"Original error: {exc!r}"
            ) from exc

        clip_weight = self._find_clip_weight(root)
        clip_module._MODEL_CKPT_PATHS[self.clip_model_name] = clip_weight

        self.clip_model = clip_module.create_model(
            model_name=self.clip_model_name,
            img_size=self.image_size,
            device=self.device,
            pretrained="openai",
            require_pretrained=True,
        )
        self.clip_model.eval()

        model = adapter_module.AdaptedCLIP(
            clip_model=self.clip_model,
            text_adapt_weight=self.text_adapt_weight,
            image_adapt_weight=self.image_adapt_weight,
            text_adapt_until=self.text_adapt_until,
            image_adapt_until=self.image_adapt_until,
            levels=self.levels,
            relu=self.relu,
        ).to(self.device)
        model.eval()

        image_checkpoint, text_checkpoint = self._find_adapter_checkpoints(root)
        image_state = self._load_checkpoint_section(
            image_checkpoint, "image_adapter"
        )
        model.image_adapter.load_state_dict(image_state)

        if text_checkpoint:
            text_state = self._load_checkpoint_section(
                text_checkpoint, "text_adapter"
            )
            model.text_adapter.load_state_dict(text_state)
            self.adapt_text = True
            self.text_encoder = model
        else:
            self.adapt_text = False
            self.text_encoder = self.clip_model

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.tokenize = tokenizer_module.tokenize
        self.gaussian_blur2d = gaussian_blur2d
        self.model = model

    def _find_aaclip_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.aaclip_root,
            os.environ.get("AACLIP_ROOT"),
            self.kwargs.get("model_root"),
            "AA-CLIP",
            "../AA-CLIP",
            "../../AA-CLIP",
            str(harness_dir.parent / "AA-CLIP"),
            str(harness_dir.parent.parent / "AA-CLIP"),
            str(harness_dir.parent.parent.parent / "AA-CLIP"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if (
                path
                and (path / "model" / "clip.py").exists()
                and (path / "model" / "adapter.py").exists()
            ):
                return path
        raise FileNotFoundError(
            "Could not locate the AA-CLIP source directory. Pass "
            "aaclip_root=... or set AACLIP_ROOT to a clone of "
            "https://github.com/Mwxinnn/AA-CLIP."
        )

    def _prepare_imports(self, root: Path) -> None:
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()

        prefixes = ("model", "dataset")
        exact_names = {"utils", "forward_utils"}
        for module_name in list(sys.modules):
            if (
                module_name in exact_names
                or any(
                    module_name == prefix or module_name.startswith(f"{prefix}.")
                    for prefix in prefixes
                )
            ):
                sys.modules.pop(module_name, None)

    def _find_clip_weight(self, root: Path) -> Path:
        candidates = [
            self.clip_weight_path,
            os.environ.get("AACLIP_CLIP_WEIGHT"),
            str(root / "model" / "ViT-L-14-336px.pt"),
            str(root / "ViT-L-14-336px.pt"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_file():
                return path
        raise FileNotFoundError(
            "Could not locate AA-CLIP's OpenAI CLIP weight "
            "ViT-L-14-336px.pt. Pass clip_weight_path=..., set "
            "AACLIP_CLIP_WEIGHT, or place the file under AA-CLIP/model/."
        )

    def _checkpoint_roots(self, root: Path) -> Sequence[Path]:
        candidates = [
            self.checkpoint_path,
            os.environ.get("AACLIP_CHECKPOINT"),
            os.environ.get("AACLIP_SAVE_PATH"),
            str(root / "ckpt" / "baseline"),
            str(root / "checkpoints"),
        ]
        roots = []
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path:
                roots.append(path)
        return roots

    def _find_adapter_checkpoints(self, root: Path) -> Tuple[Path, Optional[Path]]:
        image_candidates = []
        text_candidates = []

        for path in self._checkpoint_roots(root):
            if path.is_file():
                if path.name == "text_adapter.pth":
                    text_candidates.append(path)
                    image_candidates.extend(sorted(path.parent.glob("image_adapter_*.pth")))
                else:
                    image_candidates.append(path)
                    text_path = path.parent / "text_adapter.pth"
                    if text_path.exists():
                        text_candidates.append(text_path)
            elif path.is_dir():
                image_candidates.extend(sorted(path.glob("image_adapter_*.pth")))
                image_candidates.extend(sorted(path.glob("*image_adapter*.pth")))
                image_candidates.extend(sorted(path.rglob("image_adapter_*.pth")))
                text_path = path / "text_adapter.pth"
                if text_path.exists():
                    text_candidates.append(text_path)

        image_candidates = list(dict.fromkeys(image_candidates))
        text_candidates = list(dict.fromkeys(text_candidates))
        for image_path in reversed(image_candidates):
            if self._checkpoint_has_key(image_path, "image_adapter"):
                same_dir_text = image_path.parent / "text_adapter.pth"
                if same_dir_text.exists():
                    text_path = same_dir_text
                else:
                    text_path = text_candidates[-1] if text_candidates else None
                return image_path, text_path

        searched = "\n".join(f"  - {path}" for path in self._checkpoint_roots(root))
        raise FileNotFoundError(
            "Could not locate an AA-CLIP image adapter checkpoint. Pass "
            "checkpoint_path=... pointing to an image_adapter_*.pth file or "
            f"a save directory. Checked:\n{searched}"
        )

    def _checkpoint_has_key(self, checkpoint_path: Path, key: str) -> bool:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            return False
        return isinstance(checkpoint, dict) and key in checkpoint

    def _load_checkpoint_section(self, checkpoint_path: Path, key: str) -> Dict[str, Any]:
        map_location = self.device if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(checkpoint, dict) or key not in checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain '{key}'.")
        return checkpoint[key]

    def _get_text_features(self, category: str) -> torch.Tensor:
        category = category or "object"
        if category in self._text_feature_cache:
            return self._text_feature_cache[category]

        real_name = self._REAL_NAMES.get(category, category.replace("_", " "))
        text_features = []
        prompt_states = (self._PROMPT_NORMAL, self._PROMPT_ABNORMAL)

        with torch.no_grad():
            for states in prompt_states:
                prompted_sentence = []
                for state in states:
                    prompted_state = state.format(real_name)
                    for template in self._PROMPT_TEMPLATES:
                        prompted_sentence.append(template.format(prompted_state))
                tokens = self.tokenize(prompted_sentence).to(self.device)
                class_embeddings = self.text_encoder.encode_text(tokens)
                class_embeddings = F.normalize(class_embeddings, dim=-1)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = F.normalize(class_embedding, dim=0)
                text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(self.device)
        self._text_feature_cache[category] = text_features
        return text_features

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, lowres_maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), lowres_maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("AA-CLIP model is not loaded. Call load_model() first.")
        if not images:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("AACLIPWrapper.forward_raw_batch expects PIL images.")

        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device, non_blocking=True)

        with torch.inference_mode():
            text_features = self._get_text_features(category)
            patch_features, det_feature = self.model(image_tensor)

            image_logits = det_feature @ text_features
            anomaly_scores = (image_logits[:, 1] + 1.0) / 2.0

            lowres_maps = []
            for patch_feature in patch_features:
                patch_scores = 100.0 * torch.matmul(patch_feature, text_features)
                batch_size, token_count, channel_count = patch_scores.shape
                if channel_count != 2:
                    raise ValueError(
                        "AA-CLIP patch scores should have normal/abnormal "
                        f"channels, got {channel_count}."
                    )
                side = int(token_count ** 0.5)
                if side * side != token_count:
                    raise ValueError(
                        "AA-CLIP patch tokens do not form a square map: "
                        f"{token_count} tokens."
                    )
                patch_scores = patch_scores.permute(0, 2, 1).view(
                    batch_size, channel_count, side, side
                )
                lowres_map = (
                    patch_scores[:, 1] + 1.0 - patch_scores[:, 0]
                ) / 2.0
                if self.apply_score_blur:
                    lowres_map = self.gaussian_blur2d(
                        lowres_map.unsqueeze(1), (7, 7), (1, 1)
                    ).squeeze(1)
                lowres_maps.append(lowres_map)

            if not lowres_maps:
                raise RuntimeError("AA-CLIP returned no patch features.")

            lowres_map = torch.stack(lowres_maps).sum(dim=0)

        return (
            anomaly_scores.detach().cpu().numpy().astype(np.float32),
            lowres_map.detach().cpu().numpy().astype(np.float32),
        )

    def postprocess_condition_outputs(
        self,
        scores: np.ndarray,
        anomaly_maps: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
        """Apply the official AA-CLIP industrial score normalization/fusion."""
        return postprocess_aaclip_industrial_condition(scores, anomaly_maps)

    def release(self) -> None:
        self._text_feature_cache.clear()
        self.clip_model = None
        self.preprocess = None
        self.tokenize = None
        self.gaussian_blur2d = None
        self.text_encoder = None
        super().release()


class BayesPFLWrapper(BaseModelWrapper):
    """Wrapper for Bayes-PFL model."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("Bayes-PFL", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class TipsomalyWrapper(BaseModelWrapper):
    """Paper-faithful wrapper for the official Tipsomaly implementation."""

    fail_on_inference_error = True

    def __init__(
        self,
        checkpoint_path: str = "",
        tipsomaly_root: str = "",
        models_dir: str = "",
        dataset_name: str = "mvtec",
        weight_dataset: str = "",
        model_version: str = "l14h",
        image_size: int = 518,
        sigma: float = 4.0,
        epoch: int = 2,
        fixed_prompt_type: str = "industrial",
        prompt_learn_method: str = "concat",
        n_prompt: int = 8,
        decoupled_prompt: bool = True,
        aggregate_local2global: bool = True,
        **kwargs,
    ):
        super().__init__("Tipsomaly", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.tipsomaly_root = tipsomaly_root
        self.models_dir = models_dir
        self.dataset_name = dataset_name.lower().strip()
        self.weight_dataset = weight_dataset.lower().strip()
        self.model_version = model_version
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        self.epoch = int(epoch)
        self.fixed_prompt_type = fixed_prompt_type
        self.prompt_learn_method = prompt_learn_method
        self.n_prompt = int(n_prompt)
        self.decoupled_prompt = bool(decoupled_prompt)
        self.aggregate_local2global = bool(aggregate_local2global)

        self.preprocess = None
        self.text_encoder = None
        self.temperature = None
        self._fixed_text_features: Dict[str, torch.Tensor] = {}
        self._learnable_text_features = None
        self._resolved_checkpoint = None
        self._resolved_models_dir = None
        self._weight_dataset = ""

    def load_model(self) -> None:
        if self.dataset_name not in {"mvtec", "visa"}:
            raise ValueError(
                "Tipsomaly dataset_name must be 'mvtec' or 'visa', got "
                f"{self.dataset_name!r}."
            )
        if self.image_size != 518:
            raise ValueError(
                "The released Tipsomaly checkpoints use image_size=518."
            )
        if not self.decoupled_prompt:
            raise ValueError(
                "The published Tipsomaly zero-shot method requires "
                "decoupled_prompt=True."
            )

        root = self._find_tipsomaly_root()
        self._prepare_imports(root)
        try:
            tips_module = importlib.import_module("model.tips")
            omaly_module = importlib.import_module("model.omaly")
            transforms_module = importlib.import_module("datasets.input_transforms")
        except ImportError as exc:
            raise ImportError(
                f"Failed to import Tipsomaly from {root}. Install sentencepiece, "
                "scipy, and the official repository requirements needed by the "
                f"TIPS backbone. Original error: {exc!r}"
            ) from exc

        weight_dataset = self._cross_dataset_weight_name()
        checkpoint = self._find_checkpoint(root, weight_dataset)
        models_dir = self._find_models_dir(root)

        backbone_vision, backbone_text, tokenizer, temperature = (
            tips_module.load_model.get_model(str(models_dir), self.model_version)
        )
        preprocess, _ = transforms_module.create_transforms_tips(self.image_size)

        for backbone in (backbone_vision, backbone_text):
            backbone.to(self.device).eval()
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)

        text_encoder = omaly_module.text_encoder(
            tokenizer,
            backbone_text,
            "tips",
            backbone_text.transformer.width,
            64,
            self.prompt_learn_method,
            self.fixed_prompt_type,
            self.n_prompt,
            0,
            0,
        )
        checkpoint_object = self._torch_load(checkpoint, map_location="cpu")
        learnable_prompts = (
            checkpoint_object.get("learnable_prompts")
            if isinstance(checkpoint_object, dict)
            else checkpoint_object
        )
        if not isinstance(learnable_prompts, nn.ParameterList):
            raise TypeError(
                f"Tipsomaly checkpoint {checkpoint} must contain a ParameterList "
                "under 'learnable_prompts'."
            )
        if len(learnable_prompts) != 2:
            raise ValueError(
                f"Tipsomaly checkpoint {checkpoint} contains "
                f"{len(learnable_prompts)} prompt tensors; expected 2."
            )
        text_encoder.learnable_prompts = learnable_prompts
        text_encoder.to(self.device).eval()
        for parameter in text_encoder.parameters():
            parameter.requires_grad_(False)

        vision_encoder = omaly_module.vision_encoder(
            backbone_vision, "tips"
        ).to(self.device).eval()
        for parameter in vision_encoder.parameters():
            parameter.requires_grad_(False)

        with torch.inference_mode():
            learned_features = text_encoder(["object"], self.device, learned=True)
            learned_features = learned_features / learned_features.norm(
                dim=-1, keepdim=True
            )

        self.model = vision_encoder
        self.text_encoder = text_encoder
        self.preprocess = preprocess
        self.temperature = torch.as_tensor(temperature, device=self.device)
        self._learnable_text_features = learned_features
        self._fixed_text_features.clear()
        self._resolved_checkpoint = checkpoint
        self._resolved_models_dir = models_dir
        self._weight_dataset = weight_dataset

    def _find_tipsomaly_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.tipsomaly_root,
            os.environ.get("TIPSOMALY_ROOT"),
            self.kwargs.get("model_root"),
            "Tipsomaly",
            "../Tipsomaly",
            "../../Tipsomaly",
            str(harness_dir.parent / "Tipsomaly"),
            str(harness_dir.parent.parent / "Tipsomaly"),
            str(harness_dir.parent.parent.parent / "Tipsomaly"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if (
                path
                and (path / "model" / "tips" / "load_model.py").is_file()
                and (path / "model" / "omaly" / "text_encoder.py").is_file()
            ):
                return path
        raise FileNotFoundError(
            "Could not locate the Tipsomaly source directory. Pass "
            "tipsomaly_root=... or set TIPSOMALY_ROOT to a clone of "
            "https://github.com/Alireza99Salehi/Tipsomaly."
        )

    @staticmethod
    def _prepare_imports(root: Path) -> None:
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()

        # Tipsomaly owns top-level packages named ``model`` and ``datasets``.
        # Clear identically named packages left by another model implementation.
        for module_name in list(sys.modules):
            if (
                module_name == "model"
                or module_name.startswith("model.")
                or module_name == "datasets"
                or module_name.startswith("datasets.")
            ):
                sys.modules.pop(module_name, None)

    def _cross_dataset_weight_name(self) -> str:
        expected = "visa" if self.dataset_name == "mvtec" else "mvtec"
        selected = self.weight_dataset or expected
        if selected not in {"mvtec", "visa"}:
            raise ValueError(
                "Tipsomaly weight_dataset must be 'mvtec' or 'visa', got "
                f"{selected!r}."
            )
        if selected == self.dataset_name:
            raise ValueError(
                "Tipsomaly zero-shot evaluation requires the checkpoint trained "
                f"on the other dataset; target={self.dataset_name}, weights={selected}."
            )
        return selected

    def _find_checkpoint(self, root: Path, weight_dataset: str) -> Path:
        path = _resolve_existing_path(self.checkpoint_path)
        if path and path.is_file():
            return path
        filename = f"learnable_params_{self.epoch}.pth"
        if path and path.is_dir():
            candidate = path / filename
            if candidate.is_file():
                return candidate

        env_path = _resolve_existing_path(os.environ.get("TIPSOMALY_CHECKPOINT"))
        if env_path and env_path.is_file():
            return env_path
        default = (
            root
            / "workspaces"
            / f"trained_on_{weight_dataset}_default"
            / "vegan-arkansas"
            / "checkpoints"
            / filename
        )
        if default.is_file():
            return default
        raise FileNotFoundError(
            "Could not locate Tipsomaly's released learnable prompt checkpoint. "
            f"Expected {default}, or pass checkpoint_path explicitly."
        )

    def _find_models_dir(self, root: Path) -> Path:
        candidates = [
            self.models_dir,
            os.environ.get("TIPS_MODELS_DIR"),
            str(root / "tips"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_dir():
                return path
        # The official loader creates this directory and downloads missing TIPS
        # base components when Internet access is available.
        path = Path(self.models_dir or root / "tips").expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _torch_load(path: Path, map_location: str) -> Any:
        try:
            return torch.load(
                path,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _fixed_features_for_category(self, category: str) -> torch.Tensor:
        category = category.replace("_", " ").strip()
        if not category:
            raise ValueError("Tipsomaly requires a non-empty category name.")
        if category not in self._fixed_text_features:
            with torch.inference_mode():
                features = self.text_encoder([category], self.device, learned=False)
                features = features / features.norm(dim=-1, keepdim=True)
            self._fixed_text_features[category] = features
        return self._fixed_text_features[category]

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, lowres_maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), lowres_maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (
            self.model is None
            or self.preprocess is None
            or self.text_encoder is None
            or self.temperature is None
            or self._learnable_text_features is None
        ):
            raise RuntimeError("Tipsomaly model is not loaded. Call load_model() first.")
        if not images:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("TipsomalyWrapper.forward_raw_batch expects PIL images.")

        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device, non_blocking=True)
        fixed_features = self._fixed_features_for_category(category)

        with torch.inference_mode():
            vision_features = self.model(image_tensor)
            vision_features = [
                feature / feature.norm(dim=-1, keepdim=True)
                for feature in vision_features
            ]
            spatial_probs = F.softmax(
                (
                    vision_features[1]
                    @ fixed_features.permute(0, 2, 1)
                )
                / self.temperature,
                dim=-1,
            ).squeeze(1)
            patch_probs = F.softmax(
                (
                    vision_features[2]
                    @ self._learnable_text_features.permute(0, 2, 1)
                )
                / self.temperature,
                dim=-1,
            )

            # Paper inference: the spatial global token's anomaly probability
            # plus the strongest abnormal patch evidence.
            anomaly_scores = spatial_probs[:, 1]
            if self.aggregate_local2global:
                anomaly_scores = (
                    anomaly_scores + patch_probs[:, :, 1].max(dim=1).values
                )

            token_count = patch_probs.shape[1]
            side = int(token_count ** 0.5)
            if side * side != token_count:
                raise ValueError(
                    "Tipsomaly patch tokens do not form a square map: "
                    f"{token_count} tokens."
                )
            lowres_maps = (
                1 - patch_probs[..., 0] + patch_probs[..., 1]
            ) / 2.0
            lowres_maps = lowres_maps.reshape(-1, side, side)

        return (
            anomaly_scores.detach().cpu().numpy().astype(np.float32),
            lowres_maps.detach().cpu().numpy().astype(np.float32),
        )

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Apply Tipsomaly's official bilinear upsampling and Gaussian filter."""
        from scipy.ndimage import gaussian_filter

        map_tensor = torch.from_numpy(np.asarray(anomaly_map)).float()[None, None]
        upsampled = F.interpolate(
            map_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        return gaussian_filter(upsampled, sigma=self.sigma).astype(np.float32)

    def prepare_metric_mask(self, mask: np.ndarray) -> np.ndarray:
        """Match the upstream PIL bilinear mask resize followed by thresholding."""
        binary_mask = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        pil_mask = Image.fromarray(binary_mask)
        resized = pil_mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        return (np.asarray(resized) > 127.5).astype(np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "official_repository": (
                "https://github.com/Alireza99Salehi/Tipsomaly"
            ),
            "evaluation_dataset": self.dataset_name,
            "weight_dataset": self._weight_dataset,
            "checkpoint": str(self._resolved_checkpoint or ""),
            "tips_models_dir": str(self._resolved_models_dir or ""),
            "model_version": self.model_version,
            "image_size": self.image_size,
            "sigma": self.sigma,
            "epoch": self.epoch,
            "fixed_prompt_type": self.fixed_prompt_type,
            "prompt_learn_method": self.prompt_learn_method,
            "n_prompt": self.n_prompt,
            "decoupled_prompt": self.decoupled_prompt,
            "aggregate_local2global": self.aggregate_local2global,
            "artifact_map_resolution": self.image_size // 14,
        }

    def release(self) -> None:
        self.preprocess = None
        self.text_encoder = None
        self.temperature = None
        self._fixed_text_features.clear()
        self._learnable_text_features = None
        super().release()


class FiLoWrapper(BaseModelWrapper):
    """Paper-faithful wrapper for the official FiLo implementation."""

    metric_map_align_corners = True
    fail_on_inference_error = True

    def __init__(
        self,
        checkpoint_path: str = "",
        grounding_checkpoint_path: str = "",
        filo_root: str = "",
        groundingdino_config_path: str = "",
        dataset_name: str = "mvtec",
        clip_model: str = "ViT-L-14-336",
        clip_pretrained: str = "openai",
        image_size: int = 518,
        features_list: Optional[Sequence[int]] = None,
        n_ctx: int = 12,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        area_threshold: float = 0.7,
        outside_box_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__("FiLo", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.grounding_checkpoint_path = grounding_checkpoint_path
        self.filo_root = filo_root
        self.groundingdino_config_path = groundingdino_config_path
        self.dataset_name = dataset_name.lower().strip()
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.image_size = int(image_size)
        self.features_list = list(_as_tuple(features_list, (6, 12, 18, 24)))
        self.n_ctx = int(n_ctx)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.area_threshold = float(area_threshold)
        self.outside_box_weight = float(outside_box_weight)

        self.preprocess = None
        self.grounding_model = None
        self._dino_transform = None
        self._gaussian_blur = None
        self._filo_module = None
        self._get_phrases_from_posmap = None
        self._weight_dataset = ""
        self._grounding_attention_backend = ""
        self._resolved_checkpoint = None
        self._resolved_grounding_checkpoint = None

    def load_model(self) -> None:
        if self.dataset_name not in {"mvtec", "visa"}:
            raise ValueError(
                "FiLo dataset_name must be 'mvtec' or 'visa', got "
                f"{self.dataset_name!r}."
            )
        if self.image_size != 518:
            raise ValueError(
                "The released FiLo localization grid and position prompts require "
                "image_size=518."
            )

        root = self._find_filo_root()
        self._prepare_imports(root)
        try:
            filo_module = importlib.import_module("models.FiLo")
            dino_transforms = importlib.import_module(
                "groundingdino.datasets.transforms"
            )
            dino_models = importlib.import_module("groundingdino.models")
            dino_msda = importlib.import_module(
                "groundingdino.models.GroundingDINO.ms_deform_attn"
            )
            slconfig_module = importlib.import_module("groundingdino.util.slconfig")
            dino_utils = importlib.import_module("groundingdino.util.utils")
            from torchvision.transforms import GaussianBlur
        except ImportError as exc:
            raise ImportError(
                f"Failed to import FiLo from {root}. Install the official "
                "requirements and GroundingDINO package before loading the model. "
                f"Original error: {exc!r}"
            ) from exc

        try:
            importlib.import_module("groundingdino._C")
            self._grounding_attention_backend = "compiled_cuda_extension"
        except (ImportError, OSError):
            # The vendored Grounding DINO source includes this numerically
            # equivalent torch/grid_sample implementation for CPU inference.
            # It also operates on CUDA tensors and avoids the legacy extension
            # build, which is incompatible with some managed notebook images.
            def _pytorch_deformable_attention(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            ):
                del level_start_index, im2col_step
                return dino_msda.multi_scale_deformable_attn_pytorch(
                    value,
                    spatial_shapes,
                    sampling_locations,
                    attention_weights,
                )

            dino_msda.MultiScaleDeformableAttnFunction.apply = staticmethod(
                _pytorch_deformable_attention
            )
            self._grounding_attention_backend = "torch_grid_sample"

        filo_checkpoint = self._require_checkpoint(
            self.checkpoint_path,
            "FILO_CHECKPOINT",
            root,
            f"filo_train_on_{self._cross_dataset_weight_name()}.pth",
            "FiLo",
        )
        grounding_checkpoint = self._require_checkpoint(
            self.grounding_checkpoint_path,
            "FILO_GROUNDING_CHECKPOINT",
            root,
            f"grounding_train_on_{self._cross_dataset_weight_name()}.pth",
            "Grounding DINO",
        )
        config_path = _resolve_existing_path(self.groundingdino_config_path)
        if config_path is None:
            config_path = (
                root
                / "models"
                / "GroundingDINO"
                / "groundingdino"
                / "config"
                / "GroundingDINO_SwinT_OGC.py"
            )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"FiLo Grounding DINO config not found: {config_path}"
            )

        dino_args = slconfig_module.SLConfig.fromfile(str(config_path))
        dino_args.device = self.device
        grounding_model = dino_models.build_model(dino_args)
        grounding_state = self._torch_load(grounding_checkpoint, map_location="cpu")
        grounding_model.load_state_dict(
            dino_utils.clean_state_dict(grounding_state), strict=False
        )
        del grounding_state
        grounding_model.to(self.device).eval()
        for parameter in grounding_model.parameters():
            parameter.requires_grad_(False)

        args = SimpleNamespace(
            clip_model=self.clip_model,
            clip_pretrained=self.clip_pretrained,
            image_size=self.image_size,
            features_list=self.features_list,
            n_ctx=self.n_ctx,
            device=self.device,
        )
        obj_list = (
            [
                "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                "leather", "metal nut", "pill", "screw", "tile", "toothbrush",
                "transistor", "wood", "zipper",
            ]
            if self.dataset_name == "mvtec"
            else [
                "candle", "cashew", "chewinggum", "fryum", "pipe fryum",
                "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
                "capsules",
            ]
        )
        model = filo_module.FiLo(obj_list, args, self.device)
        filo_state = self._torch_load(filo_checkpoint, map_location="cpu")
        if not isinstance(filo_state, dict) or not isinstance(
            filo_state.get("filo"), dict
        ):
            raise ValueError(
                f"FiLo checkpoint {filo_checkpoint} does not contain a 'filo' "
                "state_dict."
            )
        model.load_state_dict(filo_state["filo"], strict=False)
        del filo_state
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        self.preprocess = model.preprocess
        self.grounding_model = grounding_model
        self._dino_transform = dino_transforms.Compose(
            [
                dino_transforms.RandomResize(
                    [self.image_size, self.image_size], max_size=1333
                ),
                dino_transforms.ToTensor(),
                dino_transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        self._gaussian_blur = GaussianBlur(3, 4.0)
        self._filo_module = filo_module
        self._get_phrases_from_posmap = dino_utils.get_phrases_from_posmap
        self._weight_dataset = self._cross_dataset_weight_name()
        self._resolved_checkpoint = filo_checkpoint
        self._resolved_grounding_checkpoint = grounding_checkpoint

    def _find_filo_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.filo_root,
            os.environ.get("FILO_ROOT"),
            self.kwargs.get("model_root"),
            "FiLo",
            "../FiLo",
            "../../FiLo",
            str(harness_dir.parent / "FiLo"),
            str(harness_dir.parent.parent / "FiLo"),
            str(harness_dir.parent.parent.parent / "FiLo"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if (
                path
                and (path / "models" / "FiLo.py").is_file()
                and (path / "models" / "GroundingDINO").is_dir()
            ):
                return path
        raise FileNotFoundError(
            "Could not locate the FiLo source directory. Pass filo_root=... or "
            "set FILO_ROOT to a clone of "
            "https://github.com/CASIA-LMC-Lab/FiLo."
        )

    def _prepare_imports(self, root: Path) -> None:
        for path in (root, root / "models" / "GroundingDINO"):
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)
        importlib.invalidate_caches()

        # FiLo owns a top-level package called ``models``. Avoid accidentally
        # reusing a package supplied by another external anomaly model.
        expected_models_root = (root / "models").resolve()
        loaded_models = sys.modules.get("models")
        loaded_file = getattr(loaded_models, "__file__", None)
        if loaded_file:
            try:
                is_filo_models = Path(loaded_file).resolve().is_relative_to(
                    expected_models_root
                )
            except AttributeError:
                is_filo_models = str(Path(loaded_file).resolve()).startswith(
                    str(expected_models_root)
                )
            if not is_filo_models:
                for module_name in list(sys.modules):
                    if module_name == "models" or module_name.startswith("models."):
                        sys.modules.pop(module_name, None)

    def _cross_dataset_weight_name(self) -> str:
        return "visa" if self.dataset_name == "mvtec" else "mvtec"

    @staticmethod
    def _torch_load(path: Path, map_location: str) -> Any:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _require_checkpoint(
        explicit_path: str,
        environment_name: str,
        root: Path,
        expected_name: str,
        label: str,
    ) -> Path:
        candidates = [
            explicit_path,
            os.environ.get(environment_name),
            str(root / expected_name),
            str(root / "ckpt" / expected_name),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_file():
                return path
        raise FileNotFoundError(
            f"Could not locate the {label} checkpoint {expected_name}. Pass the "
            f"path explicitly or set {environment_name}."
        )

    def _grounding_output(
        self, image_tensor: torch.Tensor, caption: str
    ) -> Tuple[torch.Tensor, list]:
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        with torch.inference_mode():
            outputs = self.grounding_model(
                image_tensor[None].to(self.device), captions=[caption]
            )
        logits = outputs["pred_logits"].detach().cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].detach().cpu()[0]
        box_areas = boxes[:, 2] * boxes[:, 3]
        keep = (logits.max(dim=1).values > self.box_threshold) & (
            box_areas < self.area_threshold
        )
        if keep.any():
            filtered_logits = logits[keep]
            filtered_boxes = boxes[keep]
        else:
            best = torch.argmax(logits.max(dim=1).values)
            filtered_logits = logits[best].unsqueeze(0)
            filtered_boxes = boxes[best].unsqueeze(0)

        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        phrases = []
        for logit in filtered_logits:
            phrase = self._get_phrases_from_posmap(
                logit > self.text_threshold, tokenized, tokenizer
            )
            phrases.append(phrase + f"({str(logit.max().item())[:4]})")
        return filtered_boxes, phrases

    def _localize(
        self, image: Image.Image, category: str
    ) -> Tuple[torch.Tensor, list]:
        details_by_dataset = (
            self._filo_module.mvtec_anomaly_detail_gpt
            if self.dataset_name == "mvtec"
            else self._filo_module.visa_anomaly_detail_gpt
        )
        if category not in details_by_dataset:
            raise KeyError(
                f"FiLo has no fine-grained anomaly descriptions for {category!r}."
            )
        details = details_by_dataset[category]
        caption = " . ".join(
            ["anomaly", "damage", "broken", "defect", "contamination"] + details
        )
        dino_image, _ = self._dino_transform(image, None)
        boxes, phrases = self._grounding_output(dino_image, caption)

        boxes_pixels = boxes.clone()
        valid = []
        scale = torch.tensor([self.image_size] * 4, dtype=boxes_pixels.dtype)
        keywords = details + [
            "anomaly", "damage", "broken", "defect", "contamination"
        ]
        for index in range(boxes_pixels.shape[0]):
            is_valid = any(keyword in phrases[index] for keyword in keywords)
            valid.append(is_valid)
            if not is_valid:
                continue
            boxes_pixels[index] *= scale
            boxes_pixels[index, :2] -= boxes_pixels[index, 2:] / 2
            boxes_pixels[index, 2:] += boxes_pixels[index, :2]

        max_box = None
        max_probability = 0.0
        for index, is_valid in enumerate(valid):
            if not is_valid:
                continue
            probability = float(phrases[index].rsplit("(", 1)[1].rstrip(")"))
            if probability >= max_probability:
                max_box = boxes_pixels[index]
                max_probability = probability
        if max_box is None:
            center = (259.0, 259.0)
        else:
            center = (
                float((max_box[0] + max_box[2]) / 2),
                float((max_box[1] + max_box[3]) / 2),
            )

        positions = []
        for region, ((x1, y1), (x2, y2)) in self._filo_module.location_map.items():
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                positions.append(region)
                break
        return boxes_pixels, positions

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        if (
            self.model is None
            or self.preprocess is None
            or self.grounding_model is None
            or self._gaussian_blur is None
        ):
            raise RuntimeError("FiLo model is not loaded. Call load_model() first.")
        if not isinstance(image, Image.Image):
            raise TypeError("FiLoWrapper.forward_raw expects a PIL image.")

        category = category.replace("_", " ").strip()
        rgb_image = image.convert("RGB")
        boxes, positions = self._localize(rgb_image, category)
        image_tensor = self.preprocess(rgb_image).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        items = {"img": image_tensor, "cls_name": [category]}
        with torch.inference_mode():
            text_probs, anomaly_maps = self.model(
                items, with_adapter=True, positions=positions
            )
            smoothed_maps = [
                self._gaussian_blur(
                    (anomaly_map[:, 1] - anomaly_map[:, 0] + 1) / 2
                )
                for anomaly_map in anomaly_maps
            ]
            anomaly_map = torch.mean(torch.stack(smoothed_maps), dim=0).unsqueeze(1)
            score = (
                text_probs.reshape(-1)[1].item() + anomaly_map.max().item()
            ) / 2

            # Match test.py: retain the original score inside Grounding DINO
            # rectangles and down-weight all pixels outside them.
            box_mask = anomaly_map.clone()
            for rectangle in boxes:
                left, top, right, bottom = [int(value.item()) for value in rectangle]
                box_mask[:, :, top:bottom, left:right] = 1
            anomaly_map = torch.where(
                box_mask == 1,
                anomaly_map,
                anomaly_map * self.outside_box_weight,
            )

        return (
            float(score),
            anomaly_map[0, 0].detach().cpu().numpy().astype(np.float32),
        )

    def prepare_artifact_maps(self, anomaly_maps: np.ndarray) -> np.ndarray:
        """Store FiLo maps on its native 37x37 ViT-L/14 token grid.

        FiLo performs its official softmax, smoothing, and Grounding-DINO box
        fusion at 518x518. The runner keeps those exact maps for every metric;
        only the archived copy is reduced to the native token-grid resolution.
        """
        maps = np.asarray(anomaly_maps)
        if maps.ndim != 3:
            raise ValueError(
                f"FiLo artifact maps must have shape [N, H, W], got {maps.shape}."
            )
        token_resolution = self.image_size // 14
        map_tensor = torch.from_numpy(maps).float().unsqueeze(1)
        compact = F.interpolate(
            map_tensor,
            size=(token_resolution, token_resolution),
            mode="bilinear",
            align_corners=True,
        )
        return compact[:, 0].cpu().numpy().astype(np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "official_repository": "https://github.com/CASIA-LMC-Lab/FiLo",
            "evaluation_dataset": self.dataset_name,
            "weight_dataset": self._weight_dataset,
            "filo_checkpoint": str(self._resolved_checkpoint or ""),
            "grounding_checkpoint": str(
                self._resolved_grounding_checkpoint or ""
            ),
            "image_size": self.image_size,
            "features_list": self.features_list,
            "n_ctx": self.n_ctx,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "area_threshold": self.area_threshold,
            "outside_box_weight": self.outside_box_weight,
            "grounding_attention_backend": self._grounding_attention_backend,
            "artifact_map_resolution": self.image_size // 14,
        }

    def release(self) -> None:
        self.preprocess = None
        self._dino_transform = None
        self._gaussian_blur = None
        self._filo_module = None
        self._get_phrases_from_posmap = None
        if self.grounding_model is not None:
            del self.grounding_model
            self.grounding_model = None
        super().release()


class AFCLIPWrapper(BaseModelWrapper):
    """Wrapper for the official AF-CLIP zero-shot inference implementation."""

    def __init__(
        self,
        checkpoint_path: str = "",
        afclip_root: str = "",
        prompt_checkpoint_path: str = "",
        adaptor_checkpoint_path: str = "",
        weight_dataset: str = "mvtec",
        clip_weight_path: str = "",
        clip_model_name: str = "ViT-L/14@336px",
        clip_download_dir: str = "",
        image_size: int = 518,
        prompt_len: int = 12,
        feature_layers: Optional[Sequence[int]] = None,
        memory_layers: Optional[Sequence[int]] = None,
        alpha: float = 0.1,
        **kwargs,
    ):
        super().__init__("AF-CLIP", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.afclip_root = afclip_root
        self.prompt_checkpoint_path = prompt_checkpoint_path
        self.adaptor_checkpoint_path = adaptor_checkpoint_path
        self.weight_dataset = weight_dataset.lower().strip()
        self.clip_weight_path = clip_weight_path
        self.clip_model_name = clip_model_name
        self.clip_download_dir = clip_download_dir
        self.image_size = image_size
        self.prompt_len = prompt_len
        self.feature_layers = list(_as_tuple(feature_layers, (6, 12, 18, 24)))
        self.memory_layers = list(_as_tuple(memory_layers, (6, 12, 18, 24)))
        self.alpha = alpha
        self.preprocess = None
        self._args = None

    def load_model(self) -> None:
        if self.weight_dataset not in {"mvtec", "visa"}:
            raise ValueError(
                "AF-CLIP weight_dataset must be 'mvtec' or 'visa', got "
                f"{self.weight_dataset!r}."
            )

        root = self._find_afclip_root()
        self._prepare_imports(root)

        try:
            clip_module = importlib.import_module("clip.clip")
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                f"Failed to import AF-CLIP from {root}. Install the official "
                "requirements and ensure the cloned repository is complete. "
                f"Original error: {exc!r}"
            ) from exc

        clip_source = self._find_clip_weight(root)
        download_root = Path(
            self.clip_download_dir
            or os.environ.get("AFCLIP_CLIP_DOWNLOAD_DIR", "")
            or root / "download" / "clip"
        )
        model, preprocess = clip_module.load(
            name=str(clip_source) if clip_source else self.clip_model_name,
            jit=False,
            device=self.device,
            download_root=str(download_root),
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._args = SimpleNamespace(
            prompt_len=self.prompt_len,
            img_size=self.image_size,
            feature_layers=self.feature_layers,
            memory_layers=self.memory_layers,
            alpha=self.alpha,
        )
        model.insert(args=self._args, tokenizer=clip_module.tokenize, device=self.device)

        prompt_checkpoint, adaptor_checkpoint = self._find_checkpoints(root)
        prompt_object = self._load_full_checkpoint(prompt_checkpoint)
        adaptor_object = self._load_full_checkpoint(adaptor_checkpoint)
        self._install_prompt(model, prompt_object, prompt_checkpoint)
        self._install_adaptor(model, adaptor_object, adaptor_checkpoint)

        # Match AF-CLIP main.py exactly: force square 518x518 preprocessing.
        preprocess.transforms[0] = transforms.Resize(
            size=(self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        preprocess.transforms[1] = transforms.CenterCrop(
            size=(self.image_size, self.image_size)
        )

        model.to(self.device).eval()
        self.preprocess = preprocess
        self.model = model

    def _find_afclip_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.afclip_root,
            os.environ.get("AFCLIP_ROOT"),
            self.kwargs.get("model_root"),
            "AF-CLIP",
            "../AF-CLIP",
            "../../AF-CLIP",
            str(harness_dir.parent / "AF-CLIP"),
            str(harness_dir.parent.parent / "AF-CLIP"),
            str(harness_dir.parent.parent.parent / "AF-CLIP"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if (
                path
                and (path / "clip" / "clip.py").exists()
                and (path / "clip" / "model.py").exists()
                and (path / "clip" / "adaptor.py").exists()
            ):
                return path
        raise FileNotFoundError(
            "Could not locate the AF-CLIP source directory. Pass "
            "afclip_root=... or set AFCLIP_ROOT to a clone of "
            "https://github.com/Faustinaqq/AF-CLIP."
        )

    def _prepare_imports(self, root: Path) -> None:
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()

        # AF-CLIP ships its own modified package named ``clip``. Remove any
        # previously imported OpenAI/third-party clip package before loading it.
        for module_name in list(sys.modules):
            if module_name == "clip" or module_name.startswith("clip."):
                sys.modules.pop(module_name, None)

    def _find_clip_weight(self, root: Path) -> Optional[Path]:
        candidates = [
            self.clip_weight_path,
            os.environ.get("AFCLIP_CLIP_WEIGHT"),
            str(root / "download" / "clip" / "ViT-L-14-336px.pt"),
            str(root / "ViT-L-14-336px.pt"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_file():
                return path
        return None

    def _find_checkpoints(self, root: Path) -> Tuple[Path, Path]:
        prompt_name = f"{self.weight_dataset}_prompt.pt"
        adaptor_name = f"{self.weight_dataset}_adaptor.pt"

        prompt_candidates = [self.prompt_checkpoint_path]
        adaptor_candidates = [self.adaptor_checkpoint_path]
        checkpoint_root = _resolve_existing_path(self.checkpoint_path)
        if checkpoint_root:
            if checkpoint_root.is_dir():
                prompt_candidates.append(str(checkpoint_root / prompt_name))
                adaptor_candidates.append(str(checkpoint_root / adaptor_name))
            elif checkpoint_root.name.endswith("_prompt.pt"):
                prompt_candidates.append(str(checkpoint_root))
                adaptor_candidates.append(str(checkpoint_root.with_name(adaptor_name)))
            elif checkpoint_root.name.endswith("_adaptor.pt"):
                adaptor_candidates.append(str(checkpoint_root))
                prompt_candidates.append(str(checkpoint_root.with_name(prompt_name)))

        prompt_candidates.extend(
            [
                os.environ.get("AFCLIP_PROMPT_CHECKPOINT"),
                str(root / "weight" / prompt_name),
            ]
        )
        adaptor_candidates.extend(
            [
                os.environ.get("AFCLIP_ADAPTOR_CHECKPOINT"),
                str(root / "weight" / adaptor_name),
            ]
        )

        prompt_path = next(
            (
                path
                for candidate in prompt_candidates
                if (path := _resolve_existing_path(candidate)) and path.is_file()
            ),
            None,
        )
        adaptor_path = next(
            (
                path
                for candidate in adaptor_candidates
                if (path := _resolve_existing_path(candidate)) and path.is_file()
            ),
            None,
        )
        if prompt_path is None or adaptor_path is None:
            raise FileNotFoundError(
                "Could not locate AF-CLIP's dataset-specific checkpoints. "
                f"Expected {prompt_name} and {adaptor_name} under {root / 'weight'}, "
                "or pass checkpoint_path/prompt_checkpoint_path/"
                "adaptor_checkpoint_path explicitly."
            )
        return prompt_path, adaptor_path

    def _load_full_checkpoint(self, checkpoint_path: Path) -> Any:
        # The official repository stores complete Parameter/Module objects, not
        # plain state_dicts. PyTorch 2.6 therefore requires weights_only=False.
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)

    def _install_prompt(
        self, model: nn.Module, checkpoint: Any, checkpoint_path: Path
    ) -> None:
        prompt = checkpoint
        if isinstance(prompt, dict):
            for key in ("state_prompt_embedding", "prompt", "weight"):
                if key in prompt:
                    prompt = prompt[key]
                    break
        if not isinstance(prompt, (torch.Tensor, nn.Parameter)):
            raise TypeError(
                f"AF-CLIP prompt checkpoint {checkpoint_path} must contain a tensor, "
                f"got {type(prompt).__name__}."
            )
        expected_shape = tuple(model.state_prompt_embedding.shape)
        if tuple(prompt.shape) != expected_shape:
            raise ValueError(
                f"AF-CLIP prompt shape mismatch in {checkpoint_path}: expected "
                f"{expected_shape}, got {tuple(prompt.shape)}."
            )
        prompt = prompt.detach().to(
            device=self.device,
            dtype=model.token_embedding.weight.dtype,
        )
        model.state_prompt_embedding = nn.Parameter(prompt, requires_grad=False)

    def _install_adaptor(
        self, model: nn.Module, checkpoint: Any, checkpoint_path: Path
    ) -> None:
        if isinstance(checkpoint, nn.Module):
            model.adaptor = checkpoint.to(self.device).eval()
            for parameter in model.adaptor.parameters():
                parameter.requires_grad_(False)
            return

        state_dict = checkpoint
        if isinstance(state_dict, dict):
            for key in ("adaptor", "state_dict"):
                value = state_dict.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"AF-CLIP adaptor checkpoint {checkpoint_path} must contain an "
                f"nn.Module or state_dict, got {type(checkpoint).__name__}."
            )
        model.adaptor.load_state_dict(state_dict)
        model.adaptor.to(self.device).eval()
        for parameter in model.adaptor.parameters():
            parameter.requires_grad_(False)

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, lowres_maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), lowres_maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.preprocess is None or self._args is None:
            raise RuntimeError("AF-CLIP model is not loaded. Call load_model() first.")
        if not images:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("AFCLIPWrapper.forward_raw_batch expects PIL images.")

        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device, non_blocking=True)

        with torch.inference_mode():
            anomaly_scores, anomaly_maps, _ = self.model.detect_forward_seg(
                image_tensor, self._args
            )

        if anomaly_maps.ndim != 4 or anomaly_maps.shape[1] != 1:
            raise ValueError(
                "AF-CLIP anomaly maps should have shape [B, 1, H, W], got "
                f"{tuple(anomaly_maps.shape)}."
            )
        return (
            anomaly_scores.detach().cpu().numpy().astype(np.float32),
            anomaly_maps[:, 0].detach().cpu().numpy().astype(np.float32),
        )

    def release(self) -> None:
        self.preprocess = None
        self._args = None
        super().release()


class FBCLIPWrapper(BaseModelWrapper):
    """Wrapper for the official FB-CLIP cross-dataset inference path."""

    fail_on_inference_error = True

    def __init__(
        self,
        checkpoint_path: str = "",
        fbclip_root: str = "",
        dataset_name: str = "",
        weight_dataset: str = "",
        clip_weight_path: str = "",
        clip_download_dir: str = "",
        clip_model_name: str = "ViT-L/14@336px",
        image_size: int = 518,
        depth: int = 9,
        n_ctx: int = 12,
        t_n_ctx: int = 4,
        feature_map_layer: Optional[Sequence[int]] = None,
        features_list: Optional[Sequence[int]] = None,
        feature_layers: Optional[Sequence[int]] = None,
        sigma: float = 4.0,
        use_gaussian_filter: bool = True,
        **kwargs,
    ):
        super().__init__("FB-CLIP", **kwargs)
        self.checkpoint_path = checkpoint_path
        self.fbclip_root = fbclip_root
        self.dataset_name = dataset_name.lower().strip()
        self.weight_dataset = weight_dataset.lower().strip()
        self.clip_weight_path = clip_weight_path
        self.clip_download_dir = clip_download_dir
        self.clip_model_name = clip_model_name
        self.image_size = int(image_size)
        self.depth = int(depth)
        self.n_ctx = int(n_ctx)
        self.t_n_ctx = int(t_n_ctx)
        self.feature_map_layer = list(
            _as_tuple(feature_map_layer, (5, 11, 17, 24))
        )
        self.features_list = list(_as_tuple(features_list, (5, 11, 17, 24)))
        self.feature_layers = list(
            _as_tuple(feature_layers, (1, 6, 12, 18, 24))
        )
        self.sigma = float(sigma)
        self.use_gaussian_filter = bool(use_gaussian_filter)
        self.preprocess = None
        self.prompt_learner = None
        self._args = None
        self._resolved_checkpoint: Optional[Path] = None
        self._resolved_clip_weight: Optional[Path] = None

    def load_model(self) -> None:
        if self.weight_dataset not in {"mvtec", "visa"}:
            raise ValueError(
                "FB-CLIP weight_dataset must identify the checkpoint's training "
                f"domain ('mvtec' or 'visa'), got {self.weight_dataset!r}."
            )
        if self.dataset_name:
            self._cross_dataset_weight_name()

        root = self._find_fbclip_root()
        checkpoint = self._find_checkpoint(root)
        self._prepare_imports(root)

        try:
            fbclip_module = importlib.import_module("FBCLIP_lib")
            model_load_module = importlib.import_module("FBCLIP_lib.model_load")
            prompt_module = importlib.import_module("prompt_ensemble")
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                f"Failed to import FB-CLIP from {root}. Install the official "
                "requirements and ensure the clone is complete. "
                f"Original error: {exc!r}"
            ) from exc

        # The released model loader references hashlib during its verified
        # OpenAI CLIP download but does not import it in the current upstream
        # revision. Supplying the standard-library module keeps that official
        # download path usable without modifying the clone.
        if not hasattr(model_load_module, "hashlib"):
            import hashlib

            model_load_module.hashlib = hashlib

        design_details = {
            "Prompt_length": self.n_ctx,
            "learnabel_text_embedding_depth": self.depth,
            "learnabel_text_embedding_length": self.t_n_ctx,
        }
        clip_source = self._find_clip_weight(root)
        download_root = Path(
            self.clip_download_dir
            or os.environ.get("FBCLIP_CLIP_DOWNLOAD_DIR", "")
            or root / "clip"
        )
        model, _ = fbclip_module.load(
            str(clip_source) if clip_source else self.clip_model_name,
            device=self.device,
            design_details=design_details,
            download_root=str(download_root),
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        # Match test_with_trained_model_pic.py: construct the class-agnostic
        # prompt learner while CLIP is on CPU, then install FB modules on the
        # target device before loading their released parameters.
        prompt_learner = prompt_module.FBCLIP_PromptLearner(
            model.to("cpu"), design_details
        )
        prompt_learner.to(self.device)
        model.to(self.device)
        self._args = SimpleNamespace(
            depth=self.depth,
            n_ctx=self.n_ctx,
            t_n_ctx=self.t_n_ctx,
            feature_map_layer=self.feature_map_layer,
            features_list=self.features_list,
            feature_layers=self.feature_layers,
            image_size=self.image_size,
        )
        model.FB_params(args=self._args, device=self.device)

        checkpoint_object = self._load_checkpoint(checkpoint)
        self._install_checkpoint(model, prompt_learner, checkpoint_object, checkpoint)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop((self.image_size, self.image_size)),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        model.eval()
        prompt_learner.eval()
        self.model = model
        self.prompt_learner = prompt_learner
        self._resolved_checkpoint = checkpoint
        self._resolved_clip_weight = clip_source

    def _cross_dataset_weight_name(self) -> str:
        if self.dataset_name not in {"mvtec", "visa"}:
            raise ValueError(
                "FB-CLIP dataset_name must be 'mvtec' or 'visa', got "
                f"{self.dataset_name!r}."
            )
        expected = "visa" if self.dataset_name == "mvtec" else "mvtec"
        if self.weight_dataset and self.weight_dataset != expected:
            raise ValueError(
                "FB-CLIP zero-shot evaluation requires weights trained on the "
                f"other dataset: target={self.dataset_name!r}, expected "
                f"weight_dataset={expected!r}, got {self.weight_dataset!r}."
            )
        return expected

    def _find_fbclip_root(self) -> Path:
        harness_dir = Path(__file__).resolve().parent
        candidates = [
            self.fbclip_root,
            os.environ.get("FBCLIP_ROOT"),
            self.kwargs.get("model_root"),
            "FB-CLIP",
            "../FB-CLIP",
            "../../FB-CLIP",
            str(harness_dir.parent / "FB-CLIP"),
            str(harness_dir.parent.parent / "FB-CLIP"),
            str(harness_dir.parent.parent.parent / "FB-CLIP"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if (
                path
                and (path / "FBCLIP_lib" / "FBCLIP.py").is_file()
                and (path / "prompt_ensemble.py").is_file()
            ):
                return path
        raise FileNotFoundError(
            "Could not locate the FB-CLIP source directory. Pass "
            "fbclip_root=... or set FBCLIP_ROOT to a clone of "
            "https://github.com/Xi-Mu-Yu/FB-CLIP."
        )

    def _prepare_imports(self, root: Path) -> None:
        root_str = str(root)
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
        importlib.invalidate_caches()
        for module_name in list(sys.modules):
            if (
                module_name == "FBCLIP_lib"
                or module_name.startswith("FBCLIP_lib.")
                or module_name == "prompt_ensemble"
            ):
                sys.modules.pop(module_name, None)

    def _find_checkpoint(self, root: Path) -> Path:
        expected_name = f"{self.weight_dataset}_epoch_"
        candidates = [
            self.checkpoint_path,
            os.environ.get("FBCLIP_CHECKPOINT"),
        ]
        checkpoint_root = _resolve_existing_path(self.checkpoint_path)
        if checkpoint_root and checkpoint_root.is_dir():
            candidates.extend(
                str(path)
                for path in sorted(checkpoint_root.glob(f"{expected_name}*_model.pth"))
            )
        candidates.extend(
            str(path)
            for path in sorted(root.rglob(f"{expected_name}*_model.pth"))
        )
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_file():
                if not path.name.startswith(expected_name):
                    raise ValueError(
                        "FB-CLIP checkpoint/domain mismatch: weight_dataset="
                        f"{self.weight_dataset!r}, checkpoint={path.name!r}."
                    )
                return path
        raise FileNotFoundError(
            "Could not locate the released FB-CLIP checkpoint. Expected a file "
            f"named {expected_name}*_model.pth; pass checkpoint_path=... or set "
            "FBCLIP_CHECKPOINT."
        )

    def _find_clip_weight(self, root: Path) -> Optional[Path]:
        candidates = [
            self.clip_weight_path,
            os.environ.get("FBCLIP_CLIP_WEIGHT"),
            str(root / "clip" / "ViT-L-14-336px.pt"),
            str(root / "ViT-L-14-336px.pt"),
        ]
        for candidate in candidates:
            path = _resolve_existing_path(candidate)
            if path and path.is_file():
                return path
        return None

    def _load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        load_kwargs = {"map_location": self.device}
        try:
            # The official files contain only tensors and scalar metadata. The
            # loss is a NumPy scalar, so allowlist NumPy's scalar/dtype types
            # while retaining PyTorch's restricted weights-only unpickler.
            numpy_scalar = np.core.multiarray.scalar
            numpy_dtype_type = type(np.dtype(np.float64))
            with torch.serialization.safe_globals(
                [numpy_scalar, np.dtype, numpy_dtype_type]
            ):
                checkpoint = torch.load(
                    checkpoint_path,
                    weights_only=True,
                    **load_kwargs,
                )
        except (AttributeError, TypeError):
            # PyTorch < 2.6 does not provide weights_only/safe_globals.
            checkpoint = torch.load(checkpoint_path, **load_kwargs)
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"FB-CLIP checkpoint {checkpoint_path} must contain a dict."
            )
        required = {"prompt_learner", "model_trainable_params"}
        missing = required.difference(checkpoint)
        if missing:
            raise KeyError(
                f"FB-CLIP checkpoint {checkpoint_path} is missing keys: "
                f"{sorted(missing)}."
            )
        source_domain = str(checkpoint.get("source_domain", "")).lower().strip()
        if source_domain and source_domain != self.weight_dataset:
            raise ValueError(
                "FB-CLIP checkpoint metadata/domain mismatch: expected "
                f"{self.weight_dataset!r}, found {source_domain!r}."
            )
        return checkpoint

    def _install_checkpoint(
        self,
        model: nn.Module,
        prompt_learner: nn.Module,
        checkpoint: Dict[str, Any],
        checkpoint_path: Path,
    ) -> None:
        prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
        trainable_parameters = checkpoint["model_trainable_params"]
        if not isinstance(trainable_parameters, dict):
            raise TypeError(
                "FB-CLIP model_trainable_params must be a state dictionary, got "
                f"{type(trainable_parameters).__name__}."
            )
        named_parameters = dict(model.named_parameters())
        missing = sorted(set(trainable_parameters).difference(named_parameters))
        if missing:
            raise KeyError(
                f"FB-CLIP checkpoint {checkpoint_path} contains parameters not "
                f"present in the model: {missing}."
            )
        with torch.no_grad():
            for name, value in trainable_parameters.items():
                parameter = named_parameters[name]
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        scores, lowres_maps = self.forward_raw_batch([image], category=category)
        return float(scores[0]), lowres_maps[0]

    def forward_raw_batch(
        self, images: Sequence[Image.Image], category: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        del category  # FB-CLIP's released prompt learner is class agnostic.
        if (
            self.model is None
            or self.prompt_learner is None
            or self.preprocess is None
            or self._args is None
        ):
            raise RuntimeError("FB-CLIP model is not loaded. Call load_model() first.")
        if not images:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        if not all(isinstance(image, Image.Image) for image in images):
            raise TypeError("FBCLIPWrapper.forward_raw_batch expects PIL images.")

        image_tensor = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device, non_blocking=True)
        with torch.inference_mode():
            prompts, tokenized_prompts, compound_prompts_text = (
                self.prompt_learner(cls_id=None)
            )
            scores, anomaly_maps, _ = self.model.FB_encode(
                image_tensor,
                args=self._args,
                prompts=prompts,
                tokenized_prompts=tokenized_prompts,
                compound_prompts_text=compound_prompts_text,
            )
        if anomaly_maps.ndim != 4 or anomaly_maps.shape[1] != 1:
            raise ValueError(
                "FB-CLIP anomaly maps should have shape [B, 1, H, W], got "
                f"{tuple(anomaly_maps.shape)}."
            )
        if scores.ndim != 1 or scores.shape[0] != len(images):
            raise ValueError(
                "FB-CLIP image scores should have shape [B], got "
                f"{tuple(scores.shape)}."
            )
        return (
            scores.detach().cpu().numpy().astype(np.float32),
            anomaly_maps[:, 0].detach().cpu().numpy().astype(np.float32),
        )

    def prepare_metric_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        metric_map = super().prepare_metric_map(anomaly_map)
        if self.use_gaussian_filter and self.sigma > 0:
            from scipy.ndimage import gaussian_filter

            metric_map = gaussian_filter(metric_map, sigma=self.sigma)
        return np.asarray(metric_map, dtype=np.float32)

    def inference_provenance(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "implementation": "Xi-Mu-Yu/FB-CLIP",
            "checkpoint": str(self._resolved_checkpoint or self.checkpoint_path),
            "dataset": self.dataset_name,
            "weight_dataset": self.weight_dataset,
            "clip_weight": str(
                self._resolved_clip_weight or self.clip_weight_path or "automatic"
            ),
            "clip_model": self.clip_model_name,
            "image_size": self.image_size,
            "depth": self.depth,
            "n_ctx": self.n_ctx,
            "t_n_ctx": self.t_n_ctx,
            "feature_map_layer": self.feature_map_layer,
            "features_list": self.features_list,
            "feature_layers": self.feature_layers,
            "gaussian_sigma": self.sigma if self.use_gaussian_filter else None,
        }

    def release(self) -> None:
        if self.prompt_learner is not None:
            del self.prompt_learner
            self.prompt_learner = None
        self.preprocess = None
        self._args = None
        super().release()


class CoPSWrapper(BaseModelWrapper):
    """Wrapper for CoPS model."""

    def __init__(self, checkpoint_path: str = "", **kwargs):
        super().__init__("CoPS", **kwargs)
        self.checkpoint_path = checkpoint_path

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class WinCLIPWrapper(BaseModelWrapper):
    """Wrapper for WinCLIP (training-free) model."""

    def __init__(self, **kwargs):
        super().__init__("WinCLIP", **kwargs)

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class AnoVLWrapper(BaseModelWrapper):
    """Wrapper for AnoVL (training-free) model."""

    def __init__(self, **kwargs):
        super().__init__("AnoVL", **kwargs)

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((24, 24), dtype=np.float32)
        return 0.0, lowres_map


class MRADWrapper(BaseModelWrapper):
    """Wrapper for MRAD (training-free) model."""

    def __init__(self, **kwargs):
        super().__init__("MRAD", **kwargs)

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


class AnomalyAgentWrapper(BaseModelWrapper):
    """Wrapper for AnomalyAgent (training-free) model."""

    def __init__(self, **kwargs):
        super().__init__("AnomalyAgent", **kwargs)

    def load_model(self) -> None:
        pass

    def forward_raw(
        self, image: Image.Image, category: str = ""
    ) -> Tuple[float, np.ndarray]:
        lowres_map = np.zeros((14, 14), dtype=np.float32)
        return 0.0, lowres_map


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {
    "VCP-CLIP": VCPCLIPWrapper,
    "Crane": CraneWrapper,
    "FAPrompt": FAPromptWrapper,
    "AnomalyCLIP": AnomalyCLIPWrapper,
    "AdaCLIP": AdaCLIPWrapper,
    "AA-CLIP": AACLIPWrapper,
    "Bayes-PFL": BayesPFLWrapper,
    "FiLo": FiLoWrapper,
    "Tipsomaly": TipsomalyWrapper,
    "AF-CLIP": AFCLIPWrapper,
    "FB-CLIP": FBCLIPWrapper,
    "CoPS": CoPSWrapper,
    "WinCLIP": WinCLIPWrapper,
    "AnoVL": AnoVLWrapper,
    "MRAD": MRADWrapper,
    "AnomalyAgent": AnomalyAgentWrapper,
}


def get_model(model_name: str, device: str = "cuda", **kwargs) -> BaseModelWrapper:
    """Instantiate a model wrapper by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](device=device, **kwargs)
