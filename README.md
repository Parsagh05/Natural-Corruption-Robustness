# Natural Corruption Robustness

Evaluation code and results for studying vision-language anomaly-detection models under natural image corruptions.

## Repository structure

```text
.
|-- shared/       Corruption implementation and fixed dataset assignment plans
|-- zero_shot/    Complete zero-shot evaluation pipeline, notebooks, tests, and results
`-- few_shot/     Reserved workspace for the future few-shot pipeline
```

Both evaluation modes use `shared/imagenet_c/` and the CSV files in
`shared/corruption_plans/`, so corruption behavior and categorized assignments
stay identical across shot modes. The few-shot implementation has not been
added yet. See [`zero_shot/README.md`](zero_shot/README.md) for the implemented
protocol, model setup, Kaggle instructions, output format, and reproducibility
notes.

## Setup and validation

From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s zero_shot/tests -v
python zero_shot/scripts/finalize_result_files.py --check zero_shot
```

Datasets, external model repositories, checkpoints, generated outputs, and caches are intentionally excluded from Git.
