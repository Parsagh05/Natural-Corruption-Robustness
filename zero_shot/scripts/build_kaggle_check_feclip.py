"""Build the self-contained Kaggle FE-CLIP checkpoint verification notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_check_feclip.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# FE-CLIP checkpoint verification against ICCV 2025

This notebook downloads the independent checkpoints from
[`Parsagh1383/FE-CLIP_Reproduction_Checkpoint`](https://huggingface.co/datasets/Parsagh1383/FE-CLIP_Reproduction_Checkpoint),
reconstructs FE-CLIP, evaluates the clean MVTec AD and VisA test sets using the
paper's cross-dataset protocol, and compares the reproduced metrics with Tables
1--3 of the [official FE-CLIP paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FE-CLIP_Frequency_Enhanced_CLIP_Model_for_Zero-Shot_Anomaly_Detection_and_ICCV_2025_paper.pdf).

The verification has two separate meanings:

1. **Structural integrity:** archive checksum, checkpoint keys, epoch, backbone
   checksum, source/target mapping, configuration, finite tensors, and strict
   state-dict restoration.
2. **Numerical reproduction:** per-category image AUROC/AP and pixel AUROC/PRO,
   followed by the paper's macro average across categories.

Passing structural checks proves that the uploaded files are intact and
loadable. Numerical closeness is stronger evidence, but exact equality is not
guaranteed: these are one-seed independent reconstructions and the authors did
not publish their implementation, checkpoints, five seeds, or every evaluation
detail.
"""
    ),
    code(
        r"""
from pathlib import Path

# Kaggle paths used by the training notebook.
MVTEC_PATH = Path('/kaggle/input/datasets/alirezasalehy/mvtec-ad/mvtec_anomaly_detection')
VISA_PATH = Path('/kaggle/input/datasets/alirezasalehy/visa-ad/VisA_20220922')
WORKING_DIR = Path('/kaggle/working/feclip_verification')
OUTPUT_DIR = WORKING_DIR / 'results'

HF_REPO_ID = 'Parsagh1383/FE-CLIP_Reproduction_Checkpoint'
HF_ARCHIVE_NAME = 'FE-CLIP.zip'
HF_REPO_TYPE = 'dataset'
ARCHIVE_SHA256 = '7821723a70ad54720f78d46f46d0e812f26b0d394280c8eaa5d8a76a19751499'

OPENAI_CLIP_URL = 'https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt'
OPENAI_CLIP_SHA256 = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
CLIP_COMMIT = 'dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'

RUN_DATASETS = ('MVTec', 'VisA')
IMAGE_SIZE = 336
BATCH_SIZE = 2
NUM_WORKERS = 2
GAUSSIAN_SIGMA = 4.0
PRO_MAX_STEPS = 200
PRO_FPR_LIMIT = 0.3
SEED = 111

# Paper values from Tables 1--3, in percentage points.
PAPER_TARGETS = {
    'MVTec': {'image_auroc': 91.9, 'image_ap': 96.5, 'pixel_auroc': 92.6, 'pixel_pro': 88.3},
    'VisA':  {'image_auroc': 84.6, 'image_ap': 86.6, 'pixel_auroc': 95.9, 'pixel_pro': 92.8},
}

# Exact means agreement at the paper's one-decimal reporting precision is very
# strict. CLOSE_TOLERANCE_PP is the practical one-seed reproduction threshold.
EXACT_TOLERANCE_PP = 0.15
CLOSE_TOLERANCE_PP = 2.0

assert set(RUN_DATASETS) <= set(PAPER_TARGETS)
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('Datasets:', RUN_DATASETS)
print('Output:', OUTPUT_DIR)
"""
    ),
    code(
        r"""
# Internet must be enabled for Hugging Face and the official OpenAI CLIP files.
%pip install -q "huggingface_hub[hf_xet]>=0.30" "ftfy==6.2.3" "regex==2024.11.6" "tqdm==4.67.1" "scikit-image>=0.24" "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
"""
    ),
    code(
        r"""
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

import clip
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, average_precision_score, roc_auc_score
from skimage import measure
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.JPG'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download_verified(url, target, expected_sha256):
    target = Path(target)
    if target.exists() and sha256(target) == expected_sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    actual = sha256(target)
    if actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(f'SHA-256 mismatch for {target.name}: {actual}')
    return target

set_seed(SEED)
print('Torch:', torch.__version__)
print('Device:', DEVICE)
if DEVICE.type != 'cuda':
    print('WARNING: CPU evaluation is valid but will be very slow.')
"""
    ),
    code(
        r"""
# Download, verify, and extract the published checkpoint archive.
archive_path = Path(hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=HF_ARCHIVE_NAME,
    repo_type=HF_REPO_TYPE,
    local_dir=WORKING_DIR / 'huggingface',
))
archive_hash = sha256(archive_path)
assert archive_hash == ARCHIVE_SHA256, (
    f'Archive checksum changed: expected {ARCHIVE_SHA256}, got {archive_hash}. '
    'Confirm that the Hugging Face artifact is the intended version.'
)

extract_dir = WORKING_DIR / 'extracted'
checkpoint_root = extract_dir / 'FE-CLIP'
if not checkpoint_root.exists():
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

CHECKPOINTS = {
    # Cross-dataset protocol: VisA-trained -> MVTec; MVTec-trained -> VisA.
    'MVTec': checkpoint_root / 'train_on_visa_seed_111' / 'feclip_train_on_visa_epoch_09.pth',
    'VisA': checkpoint_root / 'train_on_mvtec_seed_111' / 'feclip_train_on_mvtec_epoch_09.pth',
}
for target_dataset, checkpoint_path in CHECKPOINTS.items():
    assert checkpoint_path.is_file(), f'Missing checkpoint: {checkpoint_path}'
    print(f'{target_dataset:>6} <- {checkpoint_path.relative_to(checkpoint_root)}')

CLIP_WEIGHT_PATH = download_verified(
    OPENAI_CLIP_URL,
    WORKING_DIR / 'ViT-L-14-336px.pt',
    OPENAI_CLIP_SHA256,
)
print('Archive SHA-256:', archive_hash)
print('CLIP SHA-256:', sha256(CLIP_WEIGHT_PATH))
"""
    ),
    markdown(
        r"""
## Evaluation protocol

The paper reports **dataset-level macro averages**: metrics are computed for
each object category and then averaged across the 15 MVTec or 12 VisA
categories. Image metrics are AUROC and average precision; pixel metrics are
AUROC and per-region overlap (PRO). PRO below follows the 200-threshold
Gudovskiy/AnomalyCLIP implementation and integrates only below FPR 0.3.

The paper does not state every preprocessing/postprocessing detail. This
notebook uses the reconstruction's 336×336 direct resize and applies Gaussian
smoothing with σ=4, matching the public AnomalyCLIP evaluation protocol that
FE-CLIP follows. The chosen values are recorded in the exported report.
"""
    ),
    code(
        r"""
@dataclass(frozen=True)
class EvaluationSample:
    image_path: Path
    mask_path: Optional[Path]
    label: int
    category: str

def image_files(folder):
    return sorted(path for path in Path(folder).rglob('*') if path.is_file() and path.suffix in IMAGE_EXTENSIONS)

def mvtec_samples(root):
    root = Path(root)
    samples = []
    for category_dir in sorted(path for path in root.iterdir() if (path / 'test').is_dir()):
        for defect_dir in sorted(path for path in (category_dir / 'test').iterdir() if path.is_dir()):
            anomalous = defect_dir.name.lower() != 'good'
            for image_path in image_files(defect_dir):
                mask_path = None
                if anomalous:
                    relative = image_path.relative_to(defect_dir)
                    candidates = [
                        category_dir / 'ground_truth' / defect_dir.name / relative.with_name(relative.stem + '_mask.png'),
                        category_dir / 'ground_truth' / defect_dir.name / relative.with_suffix('.png'),
                    ]
                    mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
                    if mask_path is None:
                        raise FileNotFoundError(f'Missing MVTec mask for {image_path}')
                samples.append(EvaluationSample(image_path, mask_path, int(anomalous), category_dir.name))
    if not samples:
        raise RuntimeError(f'No MVTec test samples found under {root}')
    return samples

def resolve_visa_path(root, value):
    value = str(value or '').strip()
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [Path(root) / path]
    return next((candidate for candidate in candidates if candidate.exists()), None)

def visa_samples(root):
    root = Path(root)
    split_path = root / 'split_csv' / '1cls.csv'
    if not split_path.exists():
        raise FileNotFoundError(f'Official VisA split not found: {split_path}')
    samples = []
    with split_path.open('r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if str(row.get('split', '')).strip().lower() != 'test':
                continue
            category = str(row.get('object', '')).strip()
            image_path = resolve_visa_path(root, row.get('image'))
            if image_path is None:
                raise FileNotFoundError(f"Missing VisA image: {row.get('image')}")
            label_text = str(row.get('label', '')).strip().lower()
            anomalous = label_text not in {'normal', 'good', '0', 'false'}
            mask_path = resolve_visa_path(root, row.get('mask')) if anomalous else None
            if anomalous and mask_path is None:
                raise FileNotFoundError(f"Missing VisA mask: {row.get('mask')}")
            samples.append(EvaluationSample(image_path, mask_path, int(anomalous), category))
    if not samples:
        raise RuntimeError(f'No VisA test samples found through {split_path}')
    return samples

class CategoryDataset(Dataset):
    def __init__(self, samples, image_size=336):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert('RGB')
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        image = TF.normalize(TF.to_tensor(image), CLIP_MEAN, CLIP_STD)
        if sample.mask_path is None:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            mask_image = Image.open(sample.mask_path).convert('L')
            mask_image = TF.resize(mask_image, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
            mask = (TF.pil_to_tensor(mask_image).float() > 0).float()
        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(sample.label, dtype=torch.long),
            'image_path': str(sample.image_path),
        }

DATASET_SAMPLES = {
    'MVTec': mvtec_samples(MVTEC_PATH),
    'VisA': visa_samples(VISA_PATH),
}
expected_category_counts = {'MVTec': 15, 'VisA': 12}
for dataset_name, samples in DATASET_SAMPLES.items():
    categories = sorted({sample.category for sample in samples})
    assert len(categories) == expected_category_counts[dataset_name]
    print(dataset_name, 'samples:', len(samples), 'categories:', len(categories),
          'normal/anomalous:', sum(s.label == 0 for s in samples), sum(s.label == 1 for s in samples))
"""
    ),
    code(
        r"""
def orthonormal_dct_matrix(size, dtype=torch.float32):
    matrix = torch.empty(size, size, dtype=dtype)
    for frequency in range(size):
        alpha = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
        for position in range(size):
            matrix[frequency, position] = alpha * math.cos(
                math.pi * (2 * position + 1) * frequency / (2 * size)
            )
    return matrix

class FFEAdapter(nn.Module):
    def __init__(self, channels, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('dct', orthonormal_dct_matrix(window_size), persistent=True)
        self.linear = nn.Linear(channels, channels)
        self.activation = nn.GELU()

    def forward(self, features):
        batch, channels, height, width = features.shape
        p = self.window_size
        if height % p or width % p:
            raise ValueError(f'FFE requires a grid divisible by {p}, got {height}x{width}')
        blocks = features.unfold(2, p, p).unfold(3, p, p)
        dct = self.dct.to(device=features.device, dtype=features.dtype)
        frequency = torch.einsum('ip,bcxypq,jq->bcxyij', dct, blocks, dct)
        frequency = self.activation(self.linear(frequency.permute(0, 2, 3, 4, 5, 1)))
        frequency = frequency.permute(0, 5, 1, 2, 3, 4)
        spatial = torch.einsum('ip,bcxyij,jq->bcxypq', dct, frequency, dct)
        return spatial.permute(0, 1, 2, 4, 3, 5).reshape(batch, channels, height, width)

class LFSAdapter(nn.Module):
    def __init__(self, channels, window_size=3, conv_kernel=1):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('dct', orthonormal_dct_matrix(window_size), persistent=True)
        self.conv = nn.Conv2d(channels, channels, conv_kernel, padding=conv_kernel // 2)
        self.activation = nn.GELU()

    def forward(self, features):
        batch, channels, height, width = features.shape
        q = self.window_size
        patches = F.unfold(features, kernel_size=q, padding=q // 2, stride=1)
        patches = patches.view(batch, channels, q, q, height, width)
        dct = self.dct.to(device=features.device, dtype=features.dtype)
        frequency = torch.einsum('ip,bcpqhw,jq->bcijhw', dct, patches, dct)
        return self.activation(self.conv(frequency.mean(dim=(2, 3))))

class FECLIP(nn.Module):
    def __init__(self, clip_model, stage_endpoints=(6, 12, 18, 24), p=3, q=3,
                 frequency_lambda=0.1, lfs_kernel=1):
        super().__init__()
        self.clip = clip_model
        self.stage_endpoints = tuple(stage_endpoints)
        self.frequency_lambda = frequency_lambda
        visual = self.clip.visual
        self.width = visual.conv1.out_channels
        self.embed_dim = visual.proj.shape[1]
        self.grid_size = visual.input_resolution // visual.conv1.kernel_size[0]
        self.ffe_adapters = nn.ModuleList([FFEAdapter(self.width, p) for _ in self.stage_endpoints])
        self.lfs_adapters = nn.ModuleList([LFSAdapter(self.width, q, lfs_kernel) for _ in self.stage_endpoints])
        self.patch_projection = nn.Linear(self.width, self.embed_dim, bias=True)
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        tokens = clip.tokenize(['A photo of a normal object', 'A photo of a damaged object'])
        with torch.no_grad():
            text = self.clip.encode_text(tokens.to(next(self.clip.parameters()).device)).float()
        self.register_buffer('text_features', F.normalize(text, dim=-1), persistent=True)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        return self

    def _probabilities(self, features):
        features = F.normalize(features.float(), dim=-1)
        scale = self.clip.logit_scale.exp().detach().float()
        return (scale * features @ self.text_features.float().T).softmax(dim=-1)

    def forward(self, images):
        visual = self.clip.visual
        x = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
        batch, channels, grid_h, grid_w = x.shape
        if (grid_h, grid_w) != (self.grid_size, self.grid_size):
            raise ValueError(f'Expected {self.grid_size}x{self.grid_size}, got {grid_h}x{grid_w}')
        x = x.reshape(batch, channels, grid_h * grid_w).permute(0, 2, 1)
        class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            batch, 1, channels, device=x.device, dtype=x.dtype
        )
        x = torch.cat([class_token, x], dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype)).permute(1, 0, 2)

        class_probabilities, segmentation_probabilities = [], []
        block_start = 0
        for stage_index, block_end in enumerate(self.stage_endpoints):
            for block in visual.transformer.resblocks[block_start:block_end]:
                x = block(x)
            block_start = block_end
            cls = x[0]
            patches = x[1:].permute(1, 2, 0).reshape(batch, channels, grid_h, grid_w)
            frequency = self.ffe_adapters[stage_index](patches) + self.lfs_adapters[stage_index](patches)
            enhanced = self.frequency_lambda * frequency + (1.0 - self.frequency_lambda) * patches
            class_probabilities.append(self._probabilities(visual.ln_post(cls) @ visual.proj))
            projected_patches = self.patch_projection(enhanced.permute(0, 2, 3, 1))
            segmentation_probabilities.append(self._probabilities(projected_patches))
            x = torch.cat([cls.unsqueeze(0), enhanced.flatten(2).permute(2, 0, 1)], dim=0)
        return class_probabilities, segmentation_probabilities
"""
    ),
    code(
        r"""
# Exact metric definitions used by the public AnomalyCLIP evaluation family.
def cal_pro_score(masks, anomaly_maps, max_step=200, expected_fpr=0.3):
    masks = (np.asarray(masks) > 0).astype(np.uint8)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    binary_maps = np.zeros_like(anomaly_maps, dtype=bool)
    min_threshold, max_threshold = anomaly_maps.min(), anomaly_maps.max()
    delta = (max_threshold - min_threshold) / max_step
    if delta <= 0:
        return 0.0
    pros, fprs = [], []
    inverse_masks = 1 - masks
    normal_pixel_count = inverse_masks.sum()
    for threshold in np.arange(min_threshold, max_threshold, delta):
        binary_maps[anomaly_maps <= threshold] = False
        binary_maps[anomaly_maps > threshold] = True
        overlaps = []
        for binary_map, mask in zip(binary_maps, masks):
            for region in measure.regionprops(measure.label(mask)):
                overlap = binary_map[region.coords[:, 0], region.coords[:, 1]].sum()
                overlaps.append(overlap / region.area)
        false_positives = np.logical_and(inverse_masks, binary_maps).sum()
        pros.append(float(np.mean(overlaps)))
        fprs.append(false_positives / normal_pixel_count)
    pros, fprs = np.asarray(pros), np.asarray(fprs)
    valid = fprs < expected_fpr
    selected_fprs, selected_pros = fprs[valid], pros[valid]
    if selected_fprs.size < 2 or np.ptp(selected_fprs) == 0:
        return 0.0
    selected_fprs = (selected_fprs - selected_fprs.min()) / (selected_fprs.max() - selected_fprs.min())
    return float(auc(selected_fprs, selected_pros))

def category_metrics(labels, scores, masks, anomaly_maps):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.uint8)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    return {
        'image_auroc': 100.0 * roc_auc_score(labels, scores),
        'image_ap': 100.0 * average_precision_score(labels, scores),
        'pixel_auroc': 100.0 * roc_auc_score(masks.reshape(-1), anomaly_maps.reshape(-1)),
        'pixel_pro': 100.0 * cal_pro_score(masks, anomaly_maps, PRO_MAX_STEPS, PRO_FPR_LIMIT),
    }
"""
    ),
    code(
        r"""
EXPECTED_CONFIG = {
    'backbone': 'OpenAI CLIP ViT-L/14@336px',
    'image_size': 336,
    'epochs': 9,
    'learning_rate': 5e-4,
    'total_batch_size': 16,
    'stage_endpoints': (6, 12, 18, 24),
    'ffe_window_p': 3,
    'lfs_window_q': 3,
    'frequency_lambda': 0.1,
    'use_amp': False,
    'seed': 111,
}
EXPECTED_SOURCE = {'MVTec': 'VisA', 'VisA': 'MVTec'}

def validate_checkpoint_file(target_dataset, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = {'epoch', 'config', 'history', 'ffe_adapters', 'lfs_adapters', 'patch_projection', 'optimizer'}
    assert required <= set(payload), f'Missing keys: {required - set(payload)}'
    assert payload['epoch'] == 9
    config = payload['config']
    assert config['train_dataset'] == EXPECTED_SOURCE[target_dataset]
    assert config['clip_sha256'] == OPENAI_CLIP_SHA256
    for key, expected in EXPECTED_CONFIG.items():
        actual = config[key]
        if isinstance(expected, tuple):
            actual = tuple(actual)
        assert actual == expected, f'{target_dataset} config mismatch for {key}: {actual!r} != {expected!r}'
    for section in ('ffe_adapters', 'lfs_adapters', 'patch_projection'):
        for name, tensor in payload[section].items():
            assert torch.isfinite(tensor).all(), f'Non-finite tensor: {section}.{name}'
    return payload

def build_and_restore(target_dataset, checkpoint_path, device):
    payload = validate_checkpoint_file(target_dataset, checkpoint_path)
    backbone, _ = clip.load(str(CLIP_WEIGHT_PATH), device='cpu', jit=False)
    backbone = backbone.float().to(device).eval()
    config = payload['config']
    model = FECLIP(
        backbone,
        stage_endpoints=tuple(config['stage_endpoints']),
        p=config['ffe_window_p'],
        q=config['lfs_window_q'],
        frequency_lambda=config['frequency_lambda'],
        lfs_kernel=config['lfs_conv_kernel'],
    ).to(device)
    model.ffe_adapters.load_state_dict(payload['ffe_adapters'], strict=True)
    model.lfs_adapters.load_state_dict(payload['lfs_adapters'], strict=True)
    model.patch_projection.load_state_dict(payload['patch_projection'], strict=True)
    assert all(not parameter.requires_grad for parameter in model.clip.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 9_184_000, trainable_count
    return model.eval(), payload

for target_dataset, checkpoint_path in CHECKPOINTS.items():
    payload = validate_checkpoint_file(target_dataset, checkpoint_path)
    print(target_dataset, 'structural metadata: PASS;',
          'trained on', payload['config']['train_dataset'],
          'history epochs', len(payload['history']))
"""
    ),
    code(
        r"""
def evaluate_target(dataset_name, model, samples):
    category_rows = []
    examples = []
    categories = sorted({sample.category for sample in samples})
    started = time.perf_counter()

    for category in tqdm(categories, desc=f'{dataset_name} categories'):
        category_samples = [sample for sample in samples if sample.category == category]
        loader = DataLoader(
            CategoryDataset(category_samples, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=DEVICE.type == 'cuda',
        )
        all_labels, all_scores, all_masks, all_maps = [], [], [], []

        with torch.inference_mode():
            for batch in loader:
                images = batch['image'].to(DEVICE, non_blocking=True)
                class_probabilities, segmentation_probabilities = model(images)
                image_scores = torch.stack([prob[:, 1] for prob in class_probabilities]).mean(dim=0)
                anomaly_maps = torch.stack([
                    probability[..., 1] for probability in segmentation_probabilities
                ]).mean(dim=0).unsqueeze(1)
                anomaly_maps = F.interpolate(
                    anomaly_maps, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
                ).squeeze(1).cpu().numpy()
                if GAUSSIAN_SIGMA > 0:
                    anomaly_maps = np.stack([
                        gaussian_filter(anomaly_map, sigma=GAUSSIAN_SIGMA)
                        for anomaly_map in anomaly_maps
                    ])
                labels = batch['label'].numpy()
                masks = batch['mask'].squeeze(1).numpy().astype(np.uint8)
                all_labels.extend(labels.tolist())
                all_scores.extend(image_scores.cpu().numpy().tolist())
                all_masks.extend(masks)
                all_maps.extend(anomaly_maps)

                if len(examples) < 4:
                    for index, label in enumerate(labels):
                        if label == 1 and len(examples) < 4:
                            examples.append({
                                'category': category,
                                'image_path': batch['image_path'][index],
                                'mask': masks[index],
                                'map': anomaly_maps[index],
                                'score': float(image_scores[index].cpu()),
                            })

        row = {'dataset': dataset_name, 'category': category, 'samples': len(category_samples)}
        row.update(category_metrics(all_labels, all_scores, all_masks, all_maps))
        category_rows.append(row)
        del loader, all_masks, all_maps
        gc.collect()

    category_frame = pd.DataFrame(category_rows)
    means = {
        metric: float(category_frame[metric].mean())
        for metric in ('image_auroc', 'image_ap', 'pixel_auroc', 'pixel_pro')
    }
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    category_frame.to_csv(OUTPUT_DIR / f'{dataset_name.lower()}_per_category.csv', index=False)
    return category_frame, means, examples, elapsed_minutes

all_category_frames = {}
observed_means = {}
qualitative_examples = {}
runtime_minutes = {}

for dataset_name in RUN_DATASETS:
    print(f'\n===== Evaluating {dataset_name} with {CHECKPOINTS[dataset_name].name} =====')
    model, payload = build_and_restore(dataset_name, CHECKPOINTS[dataset_name], DEVICE)
    frame, means, examples, minutes = evaluate_target(dataset_name, model, DATASET_SAMPLES[dataset_name])
    all_category_frames[dataset_name] = frame
    observed_means[dataset_name] = means
    qualitative_examples[dataset_name] = examples
    runtime_minutes[dataset_name] = minutes
    print(frame.round(2).to_string(index=False))
    print('Macro mean:', {key: round(value, 2) for key, value in means.items()})
    print('Minutes:', round(minutes, 2))
    del model, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
"""
    ),
    code(
        r"""
comparison_rows = []
for dataset_name in RUN_DATASETS:
    for metric, target in PAPER_TARGETS[dataset_name].items():
        observed = observed_means[dataset_name][metric]
        gap = observed - target
        absolute_gap = abs(gap)
        if absolute_gap <= EXACT_TOLERANCE_PP:
            status = 'EXACT_AT_PAPER_PRECISION'
        elif absolute_gap <= CLOSE_TOLERANCE_PP:
            status = 'CLOSE'
        else:
            status = 'OUTSIDE_TOLERANCE'
        comparison_rows.append({
            'dataset': dataset_name,
            'metric': metric,
            'paper': target,
            'observed': observed,
            'observed_1dp': round(observed, 1),
            'gap_pp': gap,
            'absolute_gap_pp': absolute_gap,
            'status': status,
        })

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(OUTPUT_DIR / 'paper_comparison.csv', index=False)
display(comparison.round({'observed': 3, 'gap_pp': 3, 'absolute_gap_pp': 3}))

close_reproduction = bool((comparison['absolute_gap_pp'] <= CLOSE_TOLERANCE_PP).all())
exact_reproduction = bool((comparison['absolute_gap_pp'] <= EXACT_TOLERANCE_PP).all())
verdict = (
    'EXACT_AT_PAPER_PRECISION' if exact_reproduction else
    'CLOSE_ONE_SEED_REPRODUCTION' if close_reproduction else
    'NOT_NUMERICALLY_REPRODUCED_WITHIN_TOLERANCE'
)

report = {
    'verdict': verdict,
    'structural_integrity': 'PASS',
    'exact_tolerance_pp': EXACT_TOLERANCE_PP,
    'close_tolerance_pp': CLOSE_TOLERANCE_PP,
    'archive_sha256': archive_hash,
    'clip_sha256': sha256(CLIP_WEIGHT_PATH),
    'checkpoint_mapping': {name: str(path) for name, path in CHECKPOINTS.items()},
    'paper_targets': PAPER_TARGETS,
    'observed_macro_means': observed_means,
    'runtime_minutes': runtime_minutes,
    'protocol': {
        'image_size': IMAGE_SIZE,
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'pro_max_steps': PRO_MAX_STEPS,
        'pro_fpr_limit': PRO_FPR_LIMIT,
        'aggregation': 'metric_per_category_then_unweighted_category_mean',
        'checkpoint_seed': SEED,
    },
    'limitations': [
        'independent reconstruction, not official author code',
        'one checkpoint seed versus paper average of five unpublished seeds',
        'paper does not publish every implementation and evaluation detail',
        'different PyTorch version and training GPUs from the paper',
    ],
}
(OUTPUT_DIR / 'verification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('\nFINAL VERDICT:', verdict)
print('Structural integrity: PASS')
print('Results saved to:', OUTPUT_DIR)
"""
    ),
    code(
        r"""
# Optional qualitative sanity check: input, ground truth, and predicted map.
for dataset_name, examples in qualitative_examples.items():
    if not examples:
        continue
    figure, axes = plt.subplots(len(examples), 3, figsize=(12, 3.5 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, example in enumerate(examples):
        image = Image.open(example['image_path']).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(f"{example['category']} | score={example['score']:.3f}")
        axes[row_index, 1].imshow(example['mask'], cmap='gray')
        axes[row_index, 1].set_title('Ground truth')
        axes[row_index, 2].imshow(image)
        axes[row_index, 2].imshow(example['map'], cmap='jet', alpha=0.55)
        axes[row_index, 2].set_title('FE-CLIP anomaly map')
        for axis in axes[row_index]:
            axis.axis('off')
    figure.suptitle(dataset_name, fontsize=15)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f'{dataset_name.lower()}_qualitative.png', dpi=160, bbox_inches='tight')
    plt.show()

archive_output = shutil.make_archive(
    str(WORKING_DIR / 'feclip_verification_results'), 'zip', OUTPUT_DIR
)
print('Downloadable verification archive:', archive_output)
"""
    ),
    markdown(
        r"""
## Interpreting the verdict

- `structural_integrity: PASS` means the Hugging Face ZIP and both checkpoint
  files are intact, consistent with their metadata, finite, and strictly
  loadable into this reconstruction.
- `EXACT_AT_PAPER_PRECISION` means every metric is within 0.15 percentage
  points of the paper.
- `CLOSE_ONE_SEED_REPRODUCTION` means every metric is within the configurable
  2.0-point tolerance. This is reasonable evidence for a one-seed independent
  reproduction, but it is not proof of exact author-code equivalence.
- `NOT_NUMERICALLY_REPRODUCED_WITHIN_TOLERANCE` means at least one metric is
  farther away. It does **not** by itself mean the files are corrupted; likely
  causes include architecture assumptions, patch projection, stage behavior,
  preprocessing, postprocessing, or the authors' unpublished five-run setup.

The authoritative paper targets used here are:

| Dataset | Image AUROC | Image AP | Pixel AUROC | PRO |
|---|---:|---:|---:|---:|
| MVTec AD | 91.9 | 96.5 | 92.6 | 88.3 |
| VisA | 84.6 | 86.6 | 95.9 | 92.8 |
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes, {len(cells)} cells)")
