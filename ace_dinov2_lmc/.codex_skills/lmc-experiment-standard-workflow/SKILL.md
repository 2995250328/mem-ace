---
name: lmc-experiment-standard-workflow
description: Use when launching, monitoring, debugging, or summarizing ACE-DINOv2-LMC/GLACE/ACE-G experiment matrices across scenes, methods, GPUs, stages, ablations, or post-train evaluation. Enforce reproducible commands, tmux-based execution, baseline provenance checks, and metric-wise best aggregation where each metric may come from a different iter/seed/post-train eval.
---

# LMC Experiment Standard Workflow

Use this skill for the whole LMC project, not only one method or scene. It standardizes how to decide what to run, launch training, monitor progress, and report results.

## Hard Rules

- Work from project root `/home/xwh/project/ace_depth` when running training/eval scripts.
- Do not launch a new matrix until the baseline command lineage is understood from existing scripts, logs, and `summary.tsv`, unless this session already has a known comparable command family or the user explicitly asks for a fast exploratory run. In that case, record the assumption and avoid repeatedly re-recovering the same baseline.
- Prefer tmux for long jobs. Use one window per config with stable names such as `baseline`, `alpha010`, `stage2_global`, or `scene2a_p4`.
- Use explicit canonical `RUN_ROOT` under `/data/xwh/ace_dinov2_lmc/04_evaluation/<dataset>/<track>/<method>/<YYYYMMDD>_<scope>_<protocol>_<gpu_tag>/`; use the local `lmc-evaluation-directory-layout` skill before creating new run roots.
- Preserve comparability: same scene list, memory source, stage schedule, seeds, eval hypotheses, image resolution, buffer settings, and GPU allocation unless the ablation explicitly changes them.
- After training, report metric-wise best values: each metric is selected independently from all iter, cross-iter, post-s2, seed, and post-train sources available in `summary.tsv`. Do not require all metrics to come from the same checkpoint or iter.
- Always include source/provenance paths for surprising or decision-critical results.

## Autonomous Execution Contract

For this repo, treat the user request as permission to execute safe experiment workflow steps end-to-end. Do not stop for confirmations about known environment, dataset roots, `mapanything`, tmux usage, result aggregation, or baseline facts already recorded in this workflow.

Autonomously do:

- Design and launch short or full matrices when the user asks for training.
- Reuse known comparable baselines and avoid revalidating fixed facts in every turn.
- After a completed run, skip GPU/process checks and aggregate results immediately.
- Modify code only in the scoped files needed for the requested method, after recording an agent snapshot.
- Record run provenance so a later comparison can recover the code diff, command, data paths, and output root.

Pause and ask only when the next action is destructive, will overwrite existing outputs/checkpoints, exceeds the requested GPU range, changes global dependencies, or lacks enough information to avoid an invalid experiment.

## Before Launch

1. Inventory the requested scope:
   - dataset and scenes
   - methods/stages
   - changed code or flags
   - available GPUs and expected runtime
   - baseline/reference run to compare against

2. Recover the true baseline only when needed:
   - read relevant `summary.tsv`, markdown reports, and log-linked command lines
   - inspect the actual command used for good prior results
   - verify whether the baseline is single-stage, two-stage, global/no-global, GLACE/LMC, ACE-G flow, full-S1, sparse-depth, or post-train only
   - do not infer baseline settings from method names alone
   - skip this step for repeated follow-up runs in the same experiment thread when the scene, memory source, schedule, and reference command are already known

3. Validate code changes before long runs:
   - run `python -m py_compile` on touched Python files
   - run a checkpoint/load smoke test when state_dict or config serialization changes
   - dry-run shell wrappers when they support `DRY_RUN=true`

4. Use a staged budget for speculative method changes:
   - first run a short-iteration screening matrix with the same scene, memory, eval hypotheses, and seeds
   - only promote a config to full training when the short run shows a coherent signal, such as better validation metrics plus healthy diagnostics, not just one noisy checkpoint
   - include at least one sampler-only or negative-control arm when the mechanism is uncertain

## Matrix Design

Keep matrix changes small enough to interpret:

- For two GPUs, run at most two primary configs concurrently unless the user asks otherwise.
- Change one main factor at a time when restoring or validating a baseline.
- For new loss/sampler ideas, prefer a low-iteration diagnostic matrix first, then full validation only for the best one or two configs.
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

## Completion Fast Path

When the user says training has finished, do not start with GPU/process checks. Treat the user statement as authoritative unless logs contradict it. Go directly to:

1. Locate the run root from the current thread, launcher log, train manifest, or latest known `RUN_ROOT`.
2. Read the relevant train/eval logs only for terminal status, errors, and summary paths.
3. Run or read the metric-wise aggregation output. Prefer existing `<run_root>/metricwise_best.tsv` and `<run_root>/metricwise_best.md`; if missing, generate them immediately with the standard helper.
4. Report metric-wise best values and sources. Only inspect GPU/tmux state if results are missing, logs are still actively being written, or the user asks for runtime status.

Standard aggregation command:

```bash
cd /home/xwh/project/ace_depth
bash ace_dinov2_lmc/scripts/aggregate_lmc_metricwise_best.sh <run_root>
```

Launch scripts should call this helper after all training/post-train eval jobs finish, so completed runs already contain `metricwise_best.tsv` and `metricwise_best.md`.

## Result Aggregation

Primary source is every `summary.tsv` under the run root. These files should already include iter/cross/post-s2/post-train sources when the run scripts are correct. If no `summary.tsv` exists, the standard helper falls back to `eval_summary_*.txt` files and still applies metric-wise best aggregation.

Use the standard wrapper first; it writes both machine-readable TSV and Markdown:

```bash
bash ace_dinov2_lmc/scripts/aggregate_lmc_metricwise_best.sh <run_root>
```

The wrapper calls the bundled helper:

```bash
python ace_dinov2_lmc/.codex_skills/lmc-experiment-standard-workflow/scripts/best_metric_summary.py <run_root> --sources --tsv <run_root>/metricwise_best.tsv
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
