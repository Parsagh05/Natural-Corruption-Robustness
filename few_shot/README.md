# Few-Shot Natural Corruption Robustness

Few-shot evaluation pipeline for MVTec AD and VisA. It shares the exact
corruption implementation and fixed categorized assignment plans used by the
zero-shot pipeline, while allowing every few-shot wrapper to keep its official
preprocessing, checkpoint, map postprocessing, and image-scoring rules.

## Implemented models

- INP-Former: official 1-shot, 2-shot, and 4-shot checkpoints for MVTec AD and
  VisA (six dataset/checkpoint evaluations).
- PromptAD: retrained class-specific CLS/SEG checkpoint pairs for 1-shot,
  2-shot, and 4-shot MVTec AD and VisA evaluation.

## Retraining PromptAD checkpoints

[`kaggle_train_promptad.ipynb`](kaggle_train_promptad.ipynb) retrains the
official CVPR 2024 PromptAD source at pinned commit
`0f86ce0dc1ed59007d51348d8d566aed31360cf9`. It reproduces the official
class-specific 1/2/4-shot training setup on MVTec AD and VisA, including the
checked-in normal-reference index files, LAION-400M ViT-B/16+ backbone,
240-pixel preprocessing, fp16 prompt tuning, seed 111, SGD and cosine schedule,
100 epochs, and the separate image-level (`CLS`) and pixel-level (`SEG`)
training objectives.

PromptAD does not use one dataset-wide checkpoint. The complete two-dataset
suite contains 162 files: 27 classes x 3 shot settings x 2 tasks. The notebook
can shard the suite across Kaggle sessions, validate resumed checkpoints, and
exports an indexed ZIP with SHA-256 hashes, selected normal filenames, exact
commands, runtime versions, logs, CSVs, and the official result directory
layout. It displays an overall checkpoint progress bar and an instrumented
100-epoch `tqdm` bar for every class/task job. The upstream training scripts
choose the best epoch using test AUROC;
the notebook retains that behavior for paper/code fidelity and calls it out in
the artifact manifest.

On Kaggle's **T4 x2** accelerator, the notebook uses task-level parallelism:
`CLS` for a class runs on GPU 0 while that same class's `SEG` job runs on GPU
1. Set `GPU_IDS = [0, 1]` and keep `TASKS_TO_TRAIN = ["cls", "seg"]`; use
`MAX_JOBS_THIS_SESSION = 2` for one complete class pair per session or `None`
for every selected pair. Each subprocess writes to an isolated result tree,
then the parent validates and promotes its checkpoint and merges only its CSV
metric under a lock. This avoids the official scripts' shared-CSV race. A
single-GPU session remains supported with `GPU_IDS = [0]`, where the two tasks
run sequentially.

The resulting checkpoint packages are directly consumable by the registered
PromptAD few-shot wrapper.

`harness/models.py` contains a registry and `register_model()` extension point
for subsequent few-shot models. A new wrapper only needs to implement model
loading and raw inference; it can optionally override dataset checkpoint
selection and native metric map/mask preparation.

## Paper-faithful PromptAD inference

[`kaggle_promptad.ipynb`](kaggle_promptad.ipynb) evaluates the published
[`parsagholami/promptad-few-shot-checkpoints-mvtec-ad-and-visa`](https://www.kaggle.com/datasets/parsagholami/promptad-few-shot-checkpoints-mvtec-ad-and-visa)
checkpoint dataset with the same clean/corruption protocol and artifact schema
as INP-Former. It pins the official PromptAD source commit, verifies the
attached checkpoint indexes and optional SHA-256 hashes, downloads the frozen
LAION-400M OpenCLIP backbone, and runs the selected 1/2/4-shot suites.

PromptAD requires a class-matched pair for every evaluation: the CLS buffer
provides the released harmonic fusion of its textual score and maximum visual
map score, while the SEG buffer provides the harmonic text/visual anomaly map.
The wrapper keeps one frozen backbone in GPU memory and swaps the three small
inference buffers between the
two passes. It preserves the official cv2 1024-square pre-resize followed by
240-pixel model preprocessing, 400-pixel metric space, segmentation Gaussian
smoothing with sigma 4, and the released
test scripts' cv2 BGR channel path. The official implementation is fp16-only,
so CUDA is required. Artifacts store the native 15 x 15 fused SEG maps; the
official 400 x 400 interpolation and smoothing are applied for metrics instead
of inflating every raw-map artifact.

From the repository root, the equivalent CLI is:

```bash
python -m few_shot.scripts.run_promptad \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --visa-root /path/to/VisA_20220922 \
  --promptad-root ../PromptAD \
  --checkpoint-root /path/to/promptad-kaggle-dataset \
  --output-root outputs \
  --verify-checkpoint-hashes
```

## Paper-faithful INP-Former inference

The wrapper constructs the architecture from the official
[`luow23/INP-Former`](https://github.com/luow23/INP-Former) source and loads each
official `model.pth` with `strict=True`. The official checkpoint is a full state
dict, including the frozen DINOv2-register encoder, so no separate DINO
backbone download is required.

The fixed official checkpoint configuration is enforced:

- encoder: `dinov2reg_vit_base_14`
- resize: 448 x 448
- center crop: 392 x 392
- intrinsic normal prototypes: 6
- encoder targets: blocks 2-9, fused as 2 groups of 4
- decoder: 8 INP-guided prototype blocks, fused as 2 groups of 4
- raw anomaly map: mean of the two encoder/decoder cosine-distance maps at
  28 x 28
- test map: bilinear 28 -> 392 with `align_corners=True`, bilinear 392 -> 256
  with `align_corners=False`, then the official 5 x 5 Gaussian filter with
  sigma 4
- image score: mean of the largest 1% of values in the final 256 x 256 map

Ground-truth masks follow the official resize-448, center-crop-392, and
nearest-resize-256 path. Categorized geometric corruptions first transform the
image and mask with identical deterministic parameters.

## Official six-checkpoint suite

Keep every downloaded `model.pth` in its official directory because all six
files have the same basename:

```text
checkpoints/
|-- INP-Former-Few-Shot-1_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
|-- INP-Former-Few-Shot-1_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
|-- INP-Former-Few-Shot-2_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
|-- INP-Former-Few-Shot-2_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
|-- INP-Former-Few-Shot-4_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
`-- INP-Former-Few-Shot-4_dataset=VisA_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth
```

Official downloads:

| Shot | MVTec AD | VisA |
|---|---|---|
| 1 | [model.pth](https://drive.google.com/file/d/1ymAywov3JFFVzwDpcdt9Tj_iFv-mk32c/view?usp=sharing) | [model.pth](https://drive.google.com/file/d/1mwpzXjLmjYLWFDx4dUF1yuErzL37K21p/view?usp=sharing) |
| 2 | [model.pth](https://drive.google.com/file/d/1K9X8-v1bSy_mgrbVSK0w6Fx525clSTtz/view?usp=sharing) | [model.pth](https://drive.google.com/file/d/1_vlO4OSQSze095ddhkkyRWCOA2IRVLia/view?usp=sharing) |
| 4 | [model.pth](https://drive.google.com/file/d/15UtpeFveG2azUQmhogoET2HifEyIKSvX/view?usp=sharing) | [model.pth](https://drive.google.com/file/d/1MFZcRNwALdPPv1Wemk5_1WLq76BINdky/view?usp=sharing) |

The included `kaggle_inpformer.ipynb` downloads these public Google Drive files
directly by default. Enable **Internet** in the Kaggle notebook settings and
leave `DOWNLOAD_FROM_GOOGLE_DRIVE = True`. Downloads use resumable temporary
files and are checked for a valid PyTorch archive and plausible size before
being accepted. If Google Drive temporarily rate-limits the files, upload or
attach them as a Kaggle dataset, set `DOWNLOAD_FROM_GOOGLE_DRIVE = False`, and
set `ATTACHED_CHECKPOINT_ROOT` instead.

## Run all six evaluations

From the repository root:

```bash
python -m pip install -r few_shot/requirements.txt
git clone https://github.com/luow23/INP-Former.git ../INP-Former
git -C ../INP-Former checkout --detach 17d265381d9b323a2ef6e05aab0665a85edebe84
python -m few_shot.scripts.run_inpformer \
  --mvtec-root /path/to/mvtec_anomaly_detection \
  --visa-root /path/to/VisA_pytorch/1cls \
  --inpformer-root ../INP-Former \
  --checkpoint-root /path/to/checkpoints \
  --output-root outputs
```

By default the command runs 1-, 2-, and 4-shot evaluation on both datasets.
Pass `--shots 1`, `--shots 2`, or `--shots 4` to run only one shot setting.
By default every dataset/shot pair evaluates `noise`, `blur`, `photometric`, and
`geometric` at severity levels 1-4, plus one clean baseline by default. Fixed
plans from `shared/corruption_plans/` use corruption seed 123.

The Kaggle notebook exposes the evaluation protocol in one editable control
block. `DATASET_NAME` accepts `"mvtec"`, `"visa"`, or `"both"` and controls
the mounted dataset, downloaded checkpoints, and evaluation target.
INP-Former few-shot always pairs the selected target with its same-dataset
official checkpoint (MVTec with MVTec; VisA with VisA); cross-dataset weights
are specific to zero-shot protocols. `USE_CATEGORIZED_CORRUPTIONS` switches between the categorized and
uncategorized lists; either list can be reduced to a subset.
`INCLUDE_CLEAN_BASELINE`, `SEVERITY_LEVELS`, `SHOTS_TO_RUN`, `DEVICE`,
`BATCH_SIZE`, and the cache settings are independent controls. Set the active
corruption list to `[]` with `INCLUDE_CLEAN_BASELINE = True` for a clean-only
run. Keep severity 0 out of `SEVERITY_LEVELS`; it is reserved for clean data.

The Kaggle notebook intentionally leaves `CORRUPTION_CACHE_ROOT = None`. The
complete categorized plans contain 62,192 dataset-image/condition pairs; a
lossless full-resolution PNG cache can exceed the notebook's working storage.
Corruptions remain identical across shots because they are regenerated from
the same deterministic per-image seed. Enable a cache only when the available
disk capacity has been checked.

The default `SHOTS_TO_RUN = [1, 2, 4]` launches three evaluations when one
dataset is selected and six when `DATASET_NAME = "both"`. The fixed plans
contain 198,237 image-condition inferences across the complete three-shot,
both-dataset suite, so one free Kaggle session may be insufficient depending
on its GPU. Set `SHOTS_TO_RUN` to `[1]`, `[2]`, or `[4]` to run one shot per
session without changing the other evaluation settings.

## Outputs and logging

Each shot produces `INP-Former-<shot>-shot_artifacts.zip`. The archive contains
the selected dataset(s) and follows the zero-shot artifact/CSV contract:

```text
INP-Former-1-shot/
|-- evaluation.log
|-- run_manifest.json
|-- INP-Former-1-shot_MVTec_SP.csv
|-- INP-Former-1-shot_MVTec_PX.csv
|-- INP-Former-1-shot_MVTec_FINE_GRAINED_SP.csv
|-- INP-Former-1-shot_MVTec_FINE_GRAINED_PX.csv
|-- INP-Former-1-shot_MVTec_FINE_GRAINED_PER_IMAGE.json
|-- INP-Former-1-shot_VisA_*.csv/json
|-- MVTec/<class>/<corruption-category>/level_<severity>/
`-- VisA/<class>/<corruption-category>/level_<severity>/
```

Condition directories store `raw_scores.npy`, raw 28 x 28 `lowres_maps.npy`,
and `metadata.json`. The manifest records every condition, dataset root,
checkpoint path and file metadata, source commit, paper configuration, runtime,
cache configuration, and generated summary paths. The text log records every
condition and per-class image/pixel metrics. Fine-grained files split each
categorized group back into the concrete corruption subsets assigned by the
fixed plan.
