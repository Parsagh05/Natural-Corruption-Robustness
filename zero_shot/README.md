# Zero-Shot Natural Corruption Robustness

Zero-shot evaluation pipeline for measuring how vision-language anomaly-detection models behave under natural image corruptions. The harness supports MVTec AD and VisA, saves per-condition artifacts, and exports image-level and pixel-level metrics.

Implemented model wrappers:

- AnomalyCLIP
- AA-CLIP
- AF-CLIP
- FiLo

The registry also names planned models, but only the four wrappers above have executable inference implementations.

## Repository layout

```text
zero_shot/
|-- harness/                         Core datasets, model wrappers, metrics, runner, and storage
|-- kaggle_final_*.ipynb             Kaggle evaluation launchers
|-- kaggle_train_aaclip.ipynb        Kaggle AA-CLIP training launcher
|-- kaggle_train_feclip.ipynb        Independent paper-faithful FE-CLIP training reconstruction
|-- AA-CLIP/                         Checked-in AA-CLIP results
`-- AF-CLIP/                         Checked-in AF-CLIP results
```

`kaggle_train_feclip.ipynb` is self-contained because the FE-CLIP authors have
not released the promised source code or checkpoints. It implements the
paper-stated ViT-L/14@336px, FFE/LFS, cross-dataset training, losses, and
9-epoch Adam protocol. Its default Kaggle configuration uses Accelerate/DDP on
the T4 x2 accelerator with one persistent worker per GPU and gradient
accumulation that preserves the paper's total batch size of 16. The notebook
explicitly records every detail omitted or
internally inconsistent in the paper in its checkpoint metadata; consequently,
it is a transparent reconstruction rather than an official implementation.
FE-CLIP inference is not yet registered as an executable harness wrapper.

Reusable corruption assets live at the repository root: the vendored
implementation and frost assets are in `shared/imagenet_c/`, while fixed
MVTec and VisA categorized assignments are in `shared/corruption_plans/`.
Deterministic image/mask transforms and per-sample seeding live in
`shared/corruption.py`. This layout lets zero-shot and few-shot evaluations
consume exactly the same corruption definitions and plans.

## Evaluation protocol

By default, every model/dataset evaluation starts with a zero-corruption baseline using the original images. It is reported as `clean_level 0`; pass `include_clean=False` to omit it. The default concrete corruptions are Gaussian, shot, and impulse noise; defocus, motion, and zoom blur; brightness; and contrast. Corruption severity levels default to 1-4.

The optional categorized protocol groups operations into `noise`, `blur`, `photometric`, and `geometric`. It requires the matching dataset corruption-plan CSV so each image receives a deterministic, balanced assignment. Geometric operations transform the image and its mask with the same parameters.

Metrics include image-level AUROC, average precision, and F1-max, plus pixel-level AUROC, F1-max, and AUPRO. Pixel metrics are evaluated at 518 x 518; AUPRO uses an FPR limit of 0.3.

## Kaggle quick start

1. Clone or use [`Parsagh05/Natural-Corruption-Robustness`](https://github.com/Parsagh05/Natural-Corruption-Robustness). The evaluation notebooks load the harness from `zero_shot/` and corruption assets from `shared/`. If the GitHub owner or name changes, update `BENCHMARK_REPOSITORY` in the first cell of each evaluation notebook.
2. Attach the MVTec AD or VisA dataset and the model checkpoint input required by the selected notebook.
3. Open one of the evaluation notebooks, choose `DATASET_NAME`, corruption mode, severities, and batch size, then run all cells. No GitHub token or Kaggle secret is required for the public benchmark repository.

Notebook-specific external assets:

| Notebook | External model source | Required weights |
|---|---|---|
| `kaggle_final_anomalyclip.ipynb` | `zqhang/AnomalyCLIP` | AnomalyCLIP `epoch_15.pth` prompt checkpoint |
| `kaggle_final_aaclip.ipynb` | `Mwxinnn/AA-CLIP` | image/text adapter checkpoint directory and `ViT-L-14-336px.pt` |
| `kaggle_final_afclip.ipynb` | `Faustinaqq/AF-CLIP` | Prompt/adaptor weights included upstream; CLIP backbone downloaded or supplied as input |
| `kaggle_final_filo.ipynb` | `CASIA-LMC-Lab/FiLo` | Cross-dataset FiLo and fine-tuned Grounding DINO checkpoints downloaded from `FantasticGNU/FiLo`; OpenAI CLIP backbone downloaded automatically |

FiLo follows the official cross-dataset zero-shot protocol: VisA-trained FiLo
and Grounding DINO weights evaluate MVTec, while MVTec-trained weights evaluate
VisA. Its Kaggle launcher uses Grounding DINO's bundled PyTorch deformable-
attention fallback when its optional compiled extension is unavailable, and
intentionally uses batch size 1 because the released FiLo forward path is
single-image. Enable a GPU and Internet. An offline rerun must attach the two
matching checkpoints and pre-populate the OpenAI CLIP and BERT tokenizer/model
caches.

The notebooks use Kaggle paths by design. Dataset mount variables are near the top of each evaluation cell and must match the attached Kaggle inputs.

## Local setup

From the repository root, create an environment and install the shared harness dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA version when using a GPU. ImageMagick must also be available on the system because the vendored ImageNet-C motion-blur implementation uses its Wand bindings.

Clone the official repository for the model you want to run outside `zero_shot/` (or pass its path through `model_kwargs` / the corresponding environment variable). Model checkpoints and datasets are intentionally not committed.

The programmatic entry point is:

```python
from zero_shot.harness.runner import run_evaluation

run_evaluation(
    mvtec_root="/path/to/mvtec_anomaly_detection",
    visa_root=None,
    output_root="outputs",
    models=["AnomalyCLIP"],
    model_kwargs={
        "AnomalyCLIP": {
            "anomalyclip_root": "../AnomalyCLIP",
            "checkpoint_path": "/path/to/epoch_15.pth",
        }
    },
    dataset="mvtec",
    device="cuda",
)
```

For a short smoke run, pass one corruption and one severity, for example `corruption_types=["brightness"]` and `severity_levels=[1]`. A full run is GPU- and time-intensive.

## Outputs

During evaluation, artifacts are written under:

```text
outputs/<model>/<dataset>/<category>/<corruption>/level_<severity>/
```

The original-image baseline is stored under `clean/level_0/`; corrupted conditions retain their configured severity directories.

Each condition stores `raw_scores.npy`, `lowres_maps.npy`, and `metadata.json`. Summary files are exported as `<model>_<dataset>_SP.csv` and `<model>_<dataset>_PX.csv`. At the end of a model run, condition artifacts are archived into a ZIP file and the uncompressed model artifact directory is removed.

Categorized runs additionally split the in-memory predictions and correctly
transformed masks by their assigned concrete corruption. They automatically
export `<model>_<dataset>_FINE_GRAINED_SP.csv`,
`<model>_<dataset>_FINE_GRAINED_PX.csv`, and
`<model>_<dataset>_FINE_GRAINED_PER_IMAGE.json`. The JSON records each image's
scalar anomaly score and references its saved raw-score and low-resolution-map
array positions. Fine-grained categorized metrics describe the assigned image
subsets; they are not the full-image-set uncategorized protocol.

Result rows use one canonical order: `clean_level 0`, concrete corruptions in
the order configured in `harness/config.py`, categorized corruptions in their
configured order, and ascending severity within each corruption. Normalize all
checked-in result CSVs after collecting or concatenating runs with:

```bash
python zero_shot/scripts/normalize_result_csvs.py
```

Use `python zero_shot/scripts/normalize_result_csvs.py --check` in validation workflows
to detect ordering drift without modifying files. The normalizer preserves CSV
cell values, removes wholly empty records, and rejects malformed or duplicate
conditions instead of silently guessing. Before atomically replacing a file,
it reads its temporary output back and verifies every decoded cell string and
the complete row order.

Generated outputs, caches, datasets, model weights, external model clones, and notebook checkpoints are ignored by Git. Checked-in result CSV and JSON files under `AA-CLIP/` and `AF-CLIP/` remain tracked.

## Reproducibility notes

- The global default seed is `111`; categorized plans currently use seed `123` in the notebooks.
- Corruptions are seeded per sample path, corruption, and severity.
- VisA uses an official split file when one is found; otherwise the loader falls back to directory discovery.
- Do not select robustness thresholds from labeled test anomalies. F1-max is reported as a descriptive per-condition metric, not a fixed operating threshold.
