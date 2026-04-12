# Repository Guidelines

## Project Structure & Module Organization
This repository contains the `ace_dinov2_lmc` training and evaluation layer for ACE + DINOv2 + GeoLMC. Root-level scripts are the main entry points: `train_ace_dinov2_lmc.py`, `test_ace_dinov2_lmc.py`, `trainer_dinov2_lmc.py`, and `options_dinov2_lmc.py`. The [`memory_extraction/`](/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction) package holds the pooled-memory pipeline, including `run_memory_extraction.py`, `validate_memory.py`, and `extract_memory.sh`. Research notes are organized under `dino_lmc_base/`, `memory_extraction/`, and `integrate_bse_memory/` with `00_ideas/` through `04_evaluation/`.

## Build, Test, and Development Commands
Run commands from the parent project root: `/home/xwh/project/ace_depth`.

```bash
conda activate ace
python ace_dinov2_lmc/train_ace_dinov2_lmc.py <scene> <output.pt> --use_lmc False --device cuda:0
python ace_dinov2_lmc/train_ace_dinov2_lmc.py <scene> <output.pt> --use_lmc True --memory_path /path/to/memory.pt
python ace_dinov2_lmc/test_ace_dinov2_lmc.py <scene> output/.../best.pt --device cuda:0
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction <dataset_path> output/memory.pt --dataset_loader ace --device cuda:0
python -m ace_dinov2_lmc.memory_extraction.validate_memory /path/to/memory.pt --device cuda:0
```

If evaluation fails due to missing RANSAC bindings, build DSAC* once in `../dsacstar` with `python setup.py install`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and short module-level comments only where logic is non-obvious. Keep CLI flags and parser names descriptive and aligned with existing `--lmc_*`, `--s1_*`, and `--ace_g_*` patterns. This codebase relies on direct script execution and `sys.path.insert(...)`; preserve that convention unless you are refactoring the whole import path.

## Testing Guidelines
There is no dedicated `pytest` suite in this directory. Validation is command-driven: use `test_ace_dinov2_lmc.py` for checkpoint evaluation and `memory_extraction/validate_memory.py` for pooled-memory files. For training changes, include at least one reproducible smoke command and note the dataset backend (`ace` or `wai`) you exercised.

## Commit & Pull Request Guidelines
Recent history mixes concise version tags (`v1.6`), scoped research commits (`research(...): stage 1 complete`), and short Chinese progress summaries. Prefer a scoped, imperative subject that names the subsystem, for example `memory_extraction: tighten ACE loader path checks`. PRs should state the motivation, affected flow (`vanilla`, `iterative`, or `ace_g`), exact commands run, GPU/device assumptions, and key metrics or output paths. Include logs or screenshots only when they clarify regressions or evaluation results.
