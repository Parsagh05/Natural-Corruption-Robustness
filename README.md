# Natural Corruption Robustness

Evaluation pipeline for measuring how vision-language anomaly-detection models behave under natural image corruptions. The harness supports MVTec AD and VisA, saves per-condition artifacts, and exports image-level and pixel-level metrics.

Implemented model wrappers:

- AnomalyCLIP
- AA-CLIP
- AF-CLIP

The registry also names planned models, but only the three wrappers above have executable inference implementations.

## Repository layout

```text
.
|-- harness/                         Core datasets, model wrappers, metrics, runner, and storage
|-- imagenet_c/                      Vendored corruption implementations and frost assets
|-- visa_corruption_plan.csv         Fixed VisA categorized-corruption assignments
|-- mvtec_corruption_plan.csv        Fixed MVTec categorized-corruption assignments
|-- kaggle_final_*.ipynb             Kaggle evaluation launchers
|-- kaggle_train_aaclip.ipynb        Kaggle AA-CLIP training launcher
`-- AA-CLIP/                         Checked-in summary CSV results
```

## Evaluation protocol

The default concrete corruptions are Gaussian, shot, and impulse noise; defocus, motion, and zoom blur; brightness; and contrast. Severity levels default to 1-4.

The optional categorized protocol groups operations into `noise`, `blur`, `photometric`, and `geometric`. It requires the matching dataset corruption-plan CSV so each image receives a deterministic, balanced assignment. Geometric operations transform the image and its mask with the same parameters.

Metrics include image-level AUROC, average precision, and F1-max, plus pixel-level AUROC, F1-max, and AUPRO. Pixel metrics are evaluated at 518 x 518; AUPRO uses an FPR limit of 0.3.

## Kaggle quick start

1. Clone or use [`Parsagh05/Natural-Corruption-Robustness`](https://github.com/Parsagh05/Natural-Corruption-Robustness). If the GitHub owner or name changes, update `BENCHMARK_REPOSITORY` in the first cell of each evaluation notebook.
2. Attach the MVTec AD or VisA dataset and the model checkpoint input required by the selected notebook.
3. Open one of the evaluation notebooks, choose `DATASET_NAME`, corruption mode, severities, and batch size, then run all cells. No GitHub token or Kaggle secret is required for the public benchmark repository.

Notebook-specific external assets:

| Notebook | External model source | Required weights |
|---|---|---|
| `kaggle_final_anomalyclip.ipynb` | `zqhang/AnomalyCLIP` | AnomalyCLIP `epoch_15.pth` prompt checkpoint |
| `kaggle_final_aaclip.ipynb` | `Mwxinnn/AA-CLIP` | image/text adapter checkpoint directory and `ViT-L-14-336px.pt` |
| `kaggle_final_afclip.ipynb` | `Faustinaqq/AF-CLIP` | Prompt/adaptor weights included upstream; CLIP backbone downloaded or supplied as input |

The notebooks use Kaggle paths by design. Dataset mount variables are near the top of each evaluation cell and must match the attached Kaggle inputs.

## Local setup

Create an environment and install the shared harness dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA version when using a GPU. ImageMagick must also be available on the system because the vendored ImageNet-C motion-blur implementation uses its Wand bindings.

Clone the official repository for the model you want to run next to this repository (or pass its path through `model_kwargs` / the corresponding environment variable). Model checkpoints and datasets are intentionally not committed.

The programmatic entry point is:

```python
from harness.runner import run_evaluation

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

Each condition stores `raw_scores.npy`, `lowres_maps.npy`, and `metadata.json`. Summary files are exported as `<model>_<dataset>_SP.csv` and `<model>_<dataset>_PX.csv`. At the end of a model run, condition artifacts are archived into a ZIP file and the uncompressed model artifact directory is removed.

Generated outputs, caches, datasets, model weights, external model clones, and notebook checkpoints are ignored by Git. The existing `AA-CLIP/*.csv` files remain tracked as reference results.

## Reproducibility notes

- The global default seed is `111`; categorized plans currently use seed `123` in the notebooks.
- Corruptions are seeded per sample path, corruption, and severity.
- VisA uses an official split file when one is found; otherwise the loader falls back to directory discovery.
- Do not select robustness thresholds from labeled test anomalies. F1-max is reported as a descriptive per-condition metric, not a fixed operating threshold.
