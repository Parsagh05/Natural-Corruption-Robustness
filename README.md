# Natural Corruption Robustness

Evaluation code and results for studying vision-language anomaly-detection models under natural image corruptions.

## Repository structure

```text
.
|-- zero_shot/    Complete zero-shot evaluation pipeline, notebooks, tests, and results
`-- few_shot/     Reserved workspace for the future few-shot pipeline
```

The few-shot implementation has not been added yet. See [`zero_shot/README.md`](zero_shot/README.md) for the implemented protocol, model setup, Kaggle instructions, output format, and reproducibility notes.

## Setup and validation

From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
cd zero_shot
python -m unittest discover -s tests -v
python scripts/finalize_result_files.py --check .
```

Datasets, external model repositories, checkpoints, generated outputs, and caches are intentionally excluded from Git.
