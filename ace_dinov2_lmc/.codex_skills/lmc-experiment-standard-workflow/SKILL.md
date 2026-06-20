---
name: lmc-experiment-standard-workflow
description: Use when launching, monitoring, debugging, or summarizing ACE-DINOv2-LMC/GLACE/ACE-G experiment matrices across scenes, methods, GPUs, stages, ablations, or post-train evaluation. Enforce reproducible commands, tmux-based execution, baseline provenance checks, and metric-wise best aggregation where each metric may come from a different iter/seed/post-train eval.
---

# LMC Experiment Standard Workflow

Use this skill for the whole LMC project, not only one method or scene. It standardizes how to decide what to run, launch training, monitor progress, and report results.

## Hard Rules

- Work from project root `/home/xwh/project/ace_depth` when running training/eval scripts.
- Do not launch a new matrix until the baseline command lineage is understood from existing scripts, logs, and `summary.tsv`.
- Prefer tmux for long jobs. Use one window per config with stable names such as `baseline`, `alpha010`, `stage2_global`, or `scene2a_p4`.
- Use explicit `RUN_ROOT` under `/data/xwh/ace_dinov2_lmc/04_evaluation/<topic>_<date>/train/<timestamp>` unless the user gives another root.
- Preserve comparability: same scene list, memory source, stage schedule, seeds, eval hypotheses, image resolution, buffer settings, and GPU allocation unless the ablation explicitly changes them.
- After training, report metric-wise best values: each metric is selected independently from all iter, cross-iter, post-s2, seed, and post-train sources available in `summary.tsv`. Do not require all metrics to come from the same checkpoint or iter.
- Always include source/provenance paths for surprising or decision-critical results.

## Before Launch

1. Inventory the requested scope:
   - dataset and scenes
   - methods/stages
   - changed code or flags
   - available GPUs and expected runtime
   - baseline/reference run to compare against

2. Recover the true baseline:
   - read relevant `summary.tsv`, markdown reports, and log-linked command lines
   - inspect the actual command used for good prior results
   - verify whether the baseline is single-stage, two-stage, global/no-global, GLACE/LMC, ACE-G flow, full-S1, sparse-depth, or post-train only
   - do not infer baseline settings from method names alone

3. Validate code changes before long runs:
   - run `python -m py_compile` on touched Python files
   - run a checkpoint/load smoke test when state_dict or config serialization changes
   - dry-run shell wrappers when they support `DRY_RUN=true`

## Matrix Design

Keep matrix changes small enough to interpret:

- For two GPUs, run at most two primary configs concurrently unless the user asks otherwise.
- Change one main factor at a time when restoring or validating a baseline.
- Name configs by the changed factor, not by vague labels.
- Put common flags in the launch command and config-specific flags in each tmux window command.
- If a run depends on memory files, create or verify stable memory paths before launch and log the source memory path.

Recommended matrix record:

```text
RUN_ROOT=<absolute path>
session=<tmux session>
config=<name> gpu=<id> scene=<scene> methods=<methods> changed_flags=<flags>
baseline=<absolute path or report>
```

## Launch Pattern

Use the project script that already matches the dataset and method. Examples should be adapted to the requested run, not copied blindly.

```bash
cd /home/xwh/project/ace_depth
tmux new-session -d -s <session> -n <config0> "<env flags> bash ace_dinov2_lmc/scripts/<run_script>.sh 2>&1 | tee <run_root>/logs/<config0>.tmux.log"
tmux new-window -t <session> -n <config1> "<env flags> bash ace_dinov2_lmc/scripts/<run_script>.sh 2>&1 | tee <run_root>/logs/<config1>.tmux.log"
```

After launch, immediately verify:

```bash
tmux list-windows -t <session>
ps -eo pid,ppid,stat,etime,cmd | rg "<session>|train_ace_dinov2_lmc|test_ace_dinov2_lmc|run_.*lmc|conda run"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
tail -n 80 <run_root>/logs/<config>.tmux.log
```

## Monitoring

Check both process state and training health:

- active train/eval process is on the intended GPU
- logs show the intended scene, method, memory path, and changed flags
- stage progress is moving
- `valid` is not collapsing
- `nonFinite`, `nan`, `inf`, `Traceback`, and `RuntimeError` are absent
- LMC diagnostics such as memory contract, `gGain`, `dRatio`, guard losses, reread/gate stats, or attention stats are coherent for the chosen method
- post-train eval starts after training and writes summaries

When a job seems slow, distinguish:

- buffer build
- stage1 training
- stage2 training
- cross-iter eval
- post-train multi-seed eval
- actual hang or repeated error

Do not stop jobs only because wall time feels long; stop only after logs/processes show a real fault or the user asks to stop.

## Result Aggregation

Primary source is every `summary.tsv` under the run root. These files should already include iter/cross/post-s2/post-train sources when the run scripts are correct.

Use the bundled helper when possible:

```bash
python ace_dinov2_lmc/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py <run_root>
```

Aggregation rule:

- Higher is better: `50cm_5deg`, `25cm_5deg`, `10cm_5deg`, `5cm_5deg`, `2cm_2deg`, `1cm_1deg`
- Lower is better: `median_deg`, `median_cm`
- Select each metric independently.
- Include `source_<metric>` when reporting a best value if the decision depends on it.
- Include `post-train` rows because final quality often comes from post-train multi-seed evaluation.

If a result seems inconsistent with a prior report, first compare:

- exact command line
- stage schedule
- memory path and token count
- global head mode
- sparse-depth inputs
- post-train seed/hypotheses
- aggregation source file
- code commit or local diff

## Reporting Format

For status updates:

```text
session=<tmux session>
config=<name> gpu=<id> stage=<stage> progress=<step/total> health=<valid/nonfinite/errors>
run_root=<path>
```

For results:

```text
config / stage: acc50 acc25 acc10 acc5 acc2 acc1 med_deg med_cm
```

Then state the decision:

- best current config
- whether it beats restored baseline
- which metrics improved or regressed
- whether sources are post-train or iter-specific
- next smallest experiment to run

Keep final summaries concise, but preserve enough paths for audit.
