"""Build the zero- and few-shot APRIL-GAN Kaggle notebooks."""

from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
APRILGAN_COMMIT = "f13b8a634e04f9fde8fa03db125b25af5695d8e1"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_TEMPLATE = '''import os
import subprocess
import sys
from pathlib import Path

print("====== PREPARING NATURAL-CORRUPTION BENCHMARK AND APRIL-GAN ======")
BENCHMARK_REPOSITORY = "Parsagh05/Natural-Corruption-Robustness"
BENCHMARK_ROOT = Path("/kaggle/working/Natural-Corruption-Robustness")
APRILGAN_ROOT = Path("/kaggle/working/VAND-APRIL-GAN")
APRILGAN_COMMIT = "{commit}"

if not BENCHMARK_ROOT.exists():
    subprocess.run(
        ["git", "clone", f"https://github.com/{{BENCHMARK_REPOSITORY}}.git", str(BENCHMARK_ROOT)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(BENCHMARK_ROOT), "pull", "--ff-only"], check=True
    )

required_wrapper = BENCHMARK_ROOT / "{wrapper_file}"
if not required_wrapper.is_file() or "{wrapper_name}" not in required_wrapper.read_text(encoding="utf-8"):
    raise RuntimeError(
        "The cloned benchmark revision does not yet contain APRIL-GAN support. "
        "Commit and push these local changes before running on Kaggle."
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "-r",
        str(BENCHMARK_ROOT / "{requirements}"),
    ],
    check=True,
)

if not APRILGAN_ROOT.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/ByChelsea/VAND-APRIL-GAN.git", str(APRILGAN_ROOT)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(APRILGAN_ROOT), "fetch", "origin"], check=True
    )
subprocess.run(
    ["git", "-C", str(APRILGAN_ROOT), "checkout", "--detach", APRILGAN_COMMIT],
    check=True,
)
resolved_commit = subprocess.run(
    ["git", "-C", str(APRILGAN_ROOT), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if resolved_commit != APRILGAN_COMMIT:
    raise RuntimeError(f"Wrong APRIL-GAN source commit: {{resolved_commit}}")

required_upstream = [
    APRILGAN_ROOT / "open_clip" / "factory.py",
    APRILGAN_ROOT / "model.py",
    APRILGAN_ROOT / "prompt_ensemble.py",
    APRILGAN_ROOT / "exps" / "pretrained" / "mvtec_pretrained.pth",
    APRILGAN_ROOT / "exps" / "pretrained" / "visa_pretrained.pth",
]
missing_upstream = [str(path) for path in required_upstream if not path.is_file()]
if missing_upstream:
    raise FileNotFoundError(
        "APRIL-GAN clone/checkpoints are incomplete:\\n  - " + "\\n  - ".join(missing_upstream)
    )

for import_path in (BENCHMARK_ROOT, APRILGAN_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
os.environ["APRILGAN_ROOT"] = str(APRILGAN_ROOT)

import torch

if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running APRIL-GAN.")
print(f"GPU: {{torch.cuda.get_device_name(0)}}")
print(f"Official APRIL-GAN commit: {{resolved_commit}}")
print("Released cross-dataset checkpoints found in exps/pretrained.")
'''


ZERO_EVALUATION = '''import gc

from shared import corruption_plan_path
from zero_shot.harness.runner import run_evaluation

# Choose exactly one target. APRIL-GAN automatically enforces the official
# opposite-dataset checkpoint: VisA weights -> MVTec, MVTec weights -> VisA.
DATASET_NAME = "visa"  # "mvtec" or "visa"
DATASET_NAME = DATASET_NAME.lower().strip()
if DATASET_NAME not in {"mvtec", "visa"}:
    raise ValueError("DATASET_NAME must be 'mvtec' or 'visa'.")
IS_MVTEC = DATASET_NAME == "mvtec"
WEIGHT_DATASET = "visa" if IS_MVTEC else "mvtec"

MVTEC_ROOT = "/kaggle/input/datasets/alirezasalehy/mvtec-ad/mvtec_anomaly_detection"
VISA_ROOT = "/kaggle/input/datasets/alirezasalehy/visa-ad/VisA_20220922"
OUTPUT_ROOT = "/kaggle/working/outputs"
CLIP_DOWNLOAD_DIR = "/kaggle/working/aprilgan-clip"

# Optional offline backbone. If it is not attached, Internet must be enabled
# and the official OpenAI ViT-L/14@336px file is downloaded once.
clip_candidates = []
kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    clip_candidates.extend(kaggle_input.rglob("ViT-L-14-336px.pt"))
CLIP_WEIGHT_PATH = str(next((path for path in clip_candidates if path.is_file()), ""))

USE_CATEGORIZED_CORRUPTIONS = True
CORRUPTION_SEED = 123
UNCATEGORIZED_CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "motion_blur", "zoom_blur", "brightness", "contrast",
]
CATEGORIZED_CORRUPTION_TYPES = ["noise", "blur", "photometric", "geometric"]
CORRUPTION_TYPES = (
    CATEGORIZED_CORRUPTION_TYPES
    if USE_CATEGORIZED_CORRUPTIONS
    else UNCATEGORIZED_CORRUPTION_TYPES
)
INCLUDE_CLEAN_BASELINE = True
SEVERITY_LEVELS = [1, 2, 3, 4]
BATCH_SIZE = 1
CORRUPTION_CACHE_ROOT = None
CORRUPTION_CACHE_FORMAT = "png"
DEVICE = "cuda"
CORRUPTION_PLAN = corruption_plan_path(DATASET_NAME)

checkpoint_paths = {
    "mvtec": str(APRILGAN_ROOT / "exps" / "pretrained" / "mvtec_pretrained.pth"),
    "visa": str(APRILGAN_ROOT / "exps" / "pretrained" / "visa_pretrained.pth"),
}
print("LAUNCHING APRIL-GAN ZERO-SHOT ROBUSTNESS BENCHMARK")
print(f"Target / weights: {DATASET_NAME} / {WEIGHT_DATASET} (cross-dataset)")
print(f"CLIP backbone: {CLIP_WEIGHT_PATH or 'official download'}")
print(f"Corruptions: {CORRUPTION_TYPES} @ {SEVERITY_LEVELS}; clean={INCLUDE_CLEAN_BASELINE}")
print(f"Outputs: {OUTPUT_ROOT}")

run_evaluation(
    mvtec_root=MVTEC_ROOT if IS_MVTEC else None,
    visa_root=None if IS_MVTEC else VISA_ROOT,
    output_root=OUTPUT_ROOT,
    models=["APRIL-GAN"],
    model_kwargs={
        "APRIL-GAN": {
            "aprilgan_root": str(APRILGAN_ROOT),
            "checkpoint_paths": checkpoint_paths,
            "weight_dataset": WEIGHT_DATASET,
            "clip_weight_path": CLIP_WEIGHT_PATH,
            "clip_download_dir": CLIP_DOWNLOAD_DIR,
            "strict_source_commit": True,
        }
    },
    device=DEVICE,
    dataset=DATASET_NAME,
    corruption_types=CORRUPTION_TYPES,
    severity_levels=SEVERITY_LEVELS,
    include_clean=INCLUDE_CLEAN_BASELINE,
    batch_size=BATCH_SIZE,
    corruption_cache_root=CORRUPTION_CACHE_ROOT,
    corruption_cache_format=CORRUPTION_CACHE_FORMAT,
    categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS,
    categorized_corruption_plans={DATASET_NAME: str(CORRUPTION_PLAN)},
    corruption_seed=CORRUPTION_SEED if USE_CATEGORIZED_CORRUPTIONS else None,
)

gc.collect()
torch.cuda.empty_cache()
print(f"Finished. Collect {OUTPUT_ROOT}/APRIL-GAN_artifacts.zip")
'''


FEW_EVALUATION = '''import gc

from few_shot.harness.dataset import build_dataset_configs
from few_shot.harness.models import APRILGANFewShotWrapper, discover_aprilgan_checkpoints
from few_shot.harness.runner import run_aprilgan_evaluations

DATASET_NAME = "visa"  # "mvtec", "visa", or "both"
DATASET_NAME = DATASET_NAME.lower().strip()
if DATASET_NAME not in {"mvtec", "visa", "both"}:
    raise ValueError("DATASET_NAME must be 'mvtec', 'visa', or 'both'.")
DATASETS_TO_RUN = ("mvtec", "visa") if DATASET_NAME == "both" else (DATASET_NAME,)

MVTEC_ROOT = "/kaggle/input/datasets/alirezasalehy/mvtec-ad/mvtec_anomaly_detection"
VISA_ROOT = "/kaggle/input/datasets/alirezasalehy/visa-ad/VisA_20220922"
OUTPUT_ROOT = "/kaggle/working/outputs"
CLIP_DOWNLOAD_DIR = "/kaggle/working/aprilgan-clip"
clip_candidates = []
kaggle_input = Path("/kaggle/input")
if kaggle_input.exists():
    clip_candidates.extend(kaggle_input.rglob("ViT-L-14-336px.pt"))
CLIP_WEIGHT_PATH = str(next((path for path in clip_candidates if path.is_file()), ""))

USE_CATEGORIZED_CORRUPTIONS = True
CORRUPTION_SEED = 123
UNCATEGORIZED_CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "motion_blur", "zoom_blur", "brightness", "contrast",
]
CATEGORIZED_CORRUPTION_TYPES = ["noise", "blur", "photometric", "geometric"]
CORRUPTION_TYPES = (
    CATEGORIZED_CORRUPTION_TYPES
    if USE_CATEGORIZED_CORRUPTIONS
    else UNCATEGORIZED_CORRUPTION_TYPES
)
INCLUDE_CLEAN_BASELINE = True
SEVERITY_LEVELS = [1, 2, 3, 4]
SHOTS_TO_RUN = [1, 2, 4]
REFERENCE_SEEDS = [42]
BATCH_SIZE = 1
DEVICE = "cuda"
CORRUPTION_CACHE_ROOT = None
CORRUPTION_CACHE_FORMAT = "png"

CHECKPOINT_PATHS = discover_aprilgan_checkpoints(
    str(APRILGAN_ROOT / "exps" / "pretrained")
)
dataset_configs = build_dataset_configs(
    mvtec_root=MVTEC_ROOT if "mvtec" in DATASETS_TO_RUN else None,
    visa_root=VISA_ROOT if "visa" in DATASETS_TO_RUN else None,
)
config_by_dataset = {
    ("mvtec" if config.name.lower().startswith("mvtec") else "visa"): config
    for config in dataset_configs
}
missing = [name for name in DATASETS_TO_RUN if name not in config_by_dataset]
if missing:
    raise FileNotFoundError(f"Could not resolve selected Kaggle dataset roots: {missing}")
resolved_roots = {
    name: str(config_by_dataset[name].root_path) for name in DATASETS_TO_RUN
}

# Preflight the largest requested support selection before loading the CLIP model.
support_probe = APRILGANFewShotWrapper(
    checkpoint_paths=CHECKPOINT_PATHS,
    dataset_roots=resolved_roots,
    shot=max(SHOTS_TO_RUN),
    reference_seed=REFERENCE_SEEDS[0],
    device=DEVICE,
)
for dataset_name in DATASETS_TO_RUN:
    selections = support_probe._select_support_paths(dataset_name)
    print(
        f"Support preflight: {dataset_name}, {len(selections)} categories, "
        f"{max(SHOTS_TO_RUN)} draws/category."
    )

print("LAUNCHING APRIL-GAN FEW-SHOT ROBUSTNESS BENCHMARK")
print(f"Datasets: {DATASETS_TO_RUN}; shots={SHOTS_TO_RUN}; support seeds={REFERENCE_SEEDS}")
print("Checkpoints: released cross-dataset exps/pretrained files")
print(f"CLIP backbone: {CLIP_WEIGHT_PATH or 'official download'}")
print(f"Corruptions: {CORRUPTION_TYPES} @ {SEVERITY_LEVELS}; clean={INCLUDE_CLEAN_BASELINE}")

run_aprilgan_evaluations(
    mvtec_root=MVTEC_ROOT,
    visa_root=VISA_ROOT,
    output_root=OUTPUT_ROOT,
    aprilgan_root=str(APRILGAN_ROOT),
    checkpoint_paths=CHECKPOINT_PATHS,
    shots=SHOTS_TO_RUN,
    datasets=DATASETS_TO_RUN,
    reference_seeds=REFERENCE_SEEDS,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    corruption_types=CORRUPTION_TYPES,
    severity_levels=SEVERITY_LEVELS,
    categorized_corruptions=USE_CATEGORIZED_CORRUPTIONS,
    corruption_cache_root=CORRUPTION_CACHE_ROOT,
    corruption_cache_format=CORRUPTION_CACHE_FORMAT,
    corruption_seed=CORRUPTION_SEED,
    include_clean=INCLUDE_CLEAN_BASELINE,
    strict_source_commit=True,
    clip_weight_path=CLIP_WEIGHT_PATH,
    clip_download_dir=CLIP_DOWNLOAD_DIR,
)

gc.collect()
torch.cuda.empty_cache()
archives = []
for shot in SHOTS_TO_RUN:
    for seed in REFERENCE_SEEDS:
        suffix = f"-seed-{seed}" if len(REFERENCE_SEEDS) > 1 else ""
        archives.append(f"APRIL-GAN-{shot}-shot{suffix}_artifacts.zip")
print(f"Finished. Collect from {OUTPUT_ROOT}: {archives}")
'''


def main() -> None:
    zero_cells = [
        code(
            SETUP_TEMPLATE.format(
                commit=APRILGAN_COMMIT,
                wrapper_file="zero_shot/harness/models.py",
                wrapper_name="APRILGANWrapper",
                requirements="zero_shot/requirements.txt",
            )
        ),
        markdown(
            """# APRIL-GAN zero-shot natural-corruption benchmark

This notebook pins the official [ByChelsea/VAND-APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN) implementation and uses the checkpoints committed in `exps/pretrained`. It follows the released cross-dataset protocol: VisA-trained projections evaluate MVTec AD and MVTec-trained projections evaluate VisA. The OpenAI ViT-L/14@336px backbone is downloaded once unless attached as a Kaggle input.
"""
        ),
        code(ZERO_EVALUATION),
    ]
    few_cells = [
        code(
            SETUP_TEMPLATE.format(
                commit=APRILGAN_COMMIT,
                wrapper_file="few_shot/harness/models.py",
                wrapper_name="APRILGANFewShotWrapper",
                requirements="few_shot/requirements.txt",
            )
        ),
        markdown(
            """# APRIL-GAN few-shot natural-corruption benchmark

APRIL-GAN does not publish separate few-shot checkpoints. This notebook uses the same released cross-dataset projection checkpoint as zero-shot inference and builds a clean target-normal memory bank for each category. It reproduces the official `torch.randint` support sampling (including replacement), four-layer nearest-neighbour distance maps, and class/condition image-score fusion. Support images stay clean while only test images receive the configured corruptions.
"""
        ),
        code(FEW_EVALUATION),
    ]
    outputs = {
        REPOSITORY_ROOT / "zero_shot" / "kaggle_final_aprilgan.ipynb": notebook(zero_cells),
        REPOSITORY_ROOT / "few_shot" / "kaggle_aprilgan.ipynb": notebook(few_cells),
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
