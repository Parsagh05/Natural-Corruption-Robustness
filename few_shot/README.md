# Few-Shot Evaluation

This directory is reserved for the future few-shot evaluation pipeline. No
few-shot implementation is included yet.

Few-shot code should reuse the repository-level corruption assets rather than
copying zero-shot files:

- `shared/imagenet_c/` contains the vendored corruption implementation.
- `shared/corruption_plans/` contains fixed MVTec and VisA assignments.
- `shared.corruption` exposes deterministic image/mask corruption utilities.
- `shared.corruption_plan_path(dataset)` resolves the matching plan path.
