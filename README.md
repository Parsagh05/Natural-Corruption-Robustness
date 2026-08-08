# Natural Corruption Robustness

Evaluation code and results for studying vision-language anomaly-detection models under natural image corruptions.

## Repository structure

```text
.
|-- shared/       Corruption implementation and fixed dataset assignment plans
|-- zero_shot/    Complete zero-shot evaluation pipeline, notebooks, tests, and results
`-- few_shot/     Few-shot harness, model implementations, notebooks, and tests
```

Both evaluation modes use `shared/imagenet_c/` and the CSV files in
`shared/corruption_plans/`, so corruption behavior and categorized assignments
stay identical across shot modes. See [`zero_shot/README.md`](zero_shot/README.md)
for zero-shot models and [`few_shot/README.md`](few_shot/README.md) for the
paper-faithful INP-Former, PromptAD, AF-CLIP+, and APRIL-GAN 1/2/4-shot suites
on MVTec AD and VisA.

## Setup and validation

From the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s zero_shot/tests -v
python -m unittest discover -s few_shot/tests -v
python zero_shot/scripts/finalize_result_files.py --check zero_shot
```

Datasets, external model repositories, checkpoints, generated outputs, and caches are intentionally excluded from Git.
