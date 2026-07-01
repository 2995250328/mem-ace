# Repository Guidelines

## Project Structure & Module Organization
This repository contains the `ace_dinov2_lmc` training and evaluation layer for ACE + DINOv2 + GeoLMC. Root-level scripts are the main entry points: `train_ace_dinov2_lmc.py`, `test_ace_dinov2_lmc.py`, `trainer_dinov2_lmc.py`, and `options_dinov2_lmc.py`. The [`memory_extraction/`](/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction) package holds the pooled-memory pipeline, including `run_memory_extraction.py`, `validate_memory.py`, and `extract_memory.sh`. Research notes are organized under `dino_lmc_base/`, `memory_extraction/`, and `integrate_bse_memory/` with `00_ideas/` through `04_evaluation/`.

## Build, Test, and Development Commands
Run commands from the parent project root: `/home/xwh/project/ace_depth`.

```bash
conda activate mapanything
python ace_dinov2_lmc/train_ace_dinov2_lmc.py <scene> <output.pt> --use_lmc False --device cuda:0
python ace_dinov2_lmc/train_ace_dinov2_lmc.py <scene> <output.pt> --use_lmc True --memory_path /path/to/memory.pt
python ace_dinov2_lmc/test_ace_dinov2_lmc.py <scene> output/.../best.pt --device cuda:0
python -m ace_dinov2_lmc.memory_extraction.run_memory_extraction <dataset_path> output/memory.pt --dataset_loader ace --device cuda:0
python -m ace_dinov2_lmc.memory_extraction.validate_memory /path/to/memory.pt --device cuda:0
```

If evaluation fails due to missing RANSAC bindings, build DSAC* once in `../dsacstar` with `python setup.py install`.

## Agent Fast Path
Do not rediscover these facts during routine training/evaluation work:

- The working runtime for this repo is the `mapanything` conda environment. Use `/home/xwh/miniforge3/envs/mapanything/bin/python` or `conda run --no-capture-output -n mapanything python`; base/system Python is expected to miss `torch`.
- Long training/evaluation jobs must be launched in `tmux`. `nohup`/`setsid` background jobs may be killed by the exec environment. Verify the tmux session, process logs, and GPU placement immediately after launch.
- This environment often rejects sandboxed commands with `bwrap: loopback: Failed RTM_NEWADDR`. For necessary repo/data/GPU/tmux commands, use the required escalated command path directly rather than repeatedly probing the sandbox.
- For DINO + STGS validation, constrain runs to Indoor6 unless the user explicitly overrides this. Indoor6 DINO runs use ACE data at `/home/xwh/data/indoor6_ace`, MapAnything/WAI depth at `/home/xwh/data/mapanything-dataset/wai_data/indoor6`, and train-only COLMAP models at `/data/xwh/indoor6/indoor6-colmap/<scene>-tr/sparse/0`.
- Indoor6 DINO baselines are DINOv2 + MapAnything memory. For comparable short runs, keep the known fast4/b5/f10 settings aligned before changing method factors.
- Indoor6 COLMAP image names do not match ACE `train/rgb` symlink names. DINO STGS sidecar construction should use the builder's `--image-match-mode auto` pose fallback, and for `buffer_batch_size=1` the sidecar should use variable width (`image_width=None`, wrapper label `wvar`).
- When the user says training has finished, skip GPU/process checks and go straight to logs plus metric-wise aggregation. Prefer existing `<run_root>/metricwise_best.tsv` / `.md`; otherwise run `bash ace_dinov2_lmc/scripts/aggregate_lmc_metricwise_best.sh <run_root>`, which scans `summary.tsv` and falls back to `eval_summary_*.txt`.

## Autonomous Experiment Mode
When the user asks to start training, check results, analyze completed runs, or make scoped code changes in this workspace, proceed without asking for repeated confirmations. This is the default for safe, reversible work in `/home/xwh/project/ace_depth/ace_dinov2_lmc`.

Allowed autonomous actions:

- Launch training/evaluation/preprocessing jobs in `tmux` using known project scripts, known datasets, and explicit GPU limits from the current thread or repo workflow.
- Read logs, manifests, summaries, checkpoints metadata, sidecar summaries, and result files under the project and `/data/xwh/...`.
- Run metric aggregation immediately with `scripts/aggregate_lmc_metricwise_best.sh`; every metric is selected independently from all available iter/post-train/eval-summary rows in one setting.
- Make tightly scoped code changes for the requested method, add default-off flags, update scripts, and run syntax/smoke checks.
- Build or reuse sidecars/memory/preprocessing artifacts when the workflow already defines the source paths and no destructive overwrite is needed.

Safety constraints:

- Before editing code, create an agent snapshot with `bash ace_dinov2_lmc/scripts/agent_safe_snapshot.sh pre_edit <label>` from `/home/xwh/project/ace_depth`. The snapshot records `HEAD`, status, full tracked diff, untracked file list, and optional copied files under `/data/xwh/.tmp/ace_dinov2_lmc_agent_snapshots/`.
- Before launching training after code changes, save provenance into the run root with `bash ace_dinov2_lmc/scripts/agent_safe_snapshot.sh run_provenance <label> <RUN_ROOT>`.
- Use new timestamped run roots by default; never overwrite existing completed results unless the user explicitly asks.
- Preserve backwards compatibility: new losses/samplers/architectural changes should be behind explicit flags or match existing defaults.
- Do not revert user changes. Do not run destructive commands such as `rm`, `git reset`, `git checkout --`, killing training, or overwriting checkpoints without explicit user instruction.
- System-level approvals may still be required by the execution environment; request them directly for known-safe project operations instead of asking the user conceptual questions.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and short module-level comments only where logic is non-obvious. Keep CLI flags and parser names descriptive and aligned with existing `--lmc_*`, `--s1_*`, and `--ace_g_*` patterns. This codebase relies on direct script execution and `sys.path.insert(...)`; preserve that convention unless you are refactoring the whole import path.

## Testing Guidelines
There is no dedicated `pytest` suite in this directory. Validation is command-driven: use `test_ace_dinov2_lmc.py` for checkpoint evaluation and `memory_extraction/validate_memory.py` for pooled-memory files. For training changes, include at least one reproducible smoke command and note the dataset backend (`ace` or `wai`) you exercised.

## Commit & Pull Request Guidelines
Recent history mixes concise version tags (`v1.6`), scoped research commits (`research(...): stage 1 complete`), and short Chinese progress summaries. Prefer a scoped, imperative subject that names the subsystem, for example `memory_extraction: tighten ACE loader path checks`. PRs should state the motivation, affected flow (`vanilla`, `iterative`, or `ace_g`), exact commands run, GPU/device assumptions, and key metrics or output paths. Include logs or screenshots only when they clarify regressions or evaluation results.

## Local Agent Skills
This repository also ships project-local agent skills and MCP configuration for DL / vision workflows:

- skills for Claude Code: `.claude/skills`
- skills for Codex: `.codex_skills`
- rules for Cursor: `.cursor/rules`
- project MCP config for Claude Code / Cursor: `.mcp.json`

Prefer these local skills for:

- repo onboarding
- GPU / throughput tuning
- experiment result aggregation
- training failure triage
- dataset contract review
- paper-to-code planning
