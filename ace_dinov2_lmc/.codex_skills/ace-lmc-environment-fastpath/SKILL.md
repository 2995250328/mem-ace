---
name: ace-lmc-environment-fastpath
description: Use in /home/xwh/project/ace_depth/ace_dinov2_lmc when running smoke tests, py_compile, training, evaluation, tmux, nvidia-smi, or reading/writing /data outputs. Avoid wasting attempts on system Python or sandbox-only commands; use the project conda environment, project root, and required escalations directly.
---

# ACE / LMC Environment Fastpath

Use this skill before executing commands in the ACE-DINOv2-LMC project that touch Python, PyTorch, CUDA, tmux, `nvidia-smi`, DSAC*, `/data`, or parent-root modules such as `ace_fusion.py`.

## Hard Rules

- Project execution root is `/home/xwh/project/ace_depth`, not the `ace_dinov2_lmc` subdirectory, for train/eval scripts.
- Python smoke tests that import `torch`, `dsacstar`, `ace_fusion`, trainers, or model code must use:

```bash
conda run --no-capture-output -n mapanything python ...
```

- Do not first try system `python` for torch/model smoke tests. It is expected to miss project dependencies.
- Use plain `python -m py_compile` only for syntax-only checks that do not import project runtime dependencies.
- `nvidia-smi`, `tmux`, `/data/xwh/...`, and parent-root files outside the workspace often require escalation in this environment. If the task clearly needs them, request escalation on the first attempt instead of probing the sandbox first.
- Parent project files such as `/home/xwh/project/ace_depth/ace_fusion.py` may be outside the current workspace write root. If editing them is required, state that fact and use the required approved/escalated path directly.
- Long training and evaluation jobs should run in tmux. Immediately verify tmux output and GPU placement after launch.

## Command Defaults

Use these defaults unless the user specifies otherwise:

```bash
cd /home/xwh/project/ace_depth
CONDA_ENV=mapanything
OMP_NUM_THREADS=8
OMP_DYNAMIC=FALSE
EVAL_DETERMINISTIC=True
EVAL_DSACSTAR_SEED=1305
EVAL_DSACSTAR_SEED_PER_FRAME=True
POST_TRAIN_EVAL_SEEDS="1305 2026 4242"
POST_TRAIN_HYPOTHESES=256
```

For quick Python runtime checks:

```bash
cd /home/xwh/project/ace_depth
conda run --no-capture-output -n mapanything python - <<'PY'
import torch
import dsacstar
print(torch.__version__, hasattr(dsacstar, "set_seed"))
PY
```

For syntax checks of touched files:

```bash
python -m py_compile ace_fusion.py ace_dinov2_lmc/options_dinov2_lmc.py ace_dinov2_lmc/trainer_dinov2_lmc.py
```

## Avoid

- Do not spend a failed attempt on system Python for any smoke that imports `torch`.
- Do not run training/eval scripts from the subdirectory unless the script explicitly expects it.
- Do not infer GPU state from memory usage alone; verify compute apps and map UUIDs to indices when placement matters.
- Do not leave a tmux launch unverified.

## Persistent Lessons From Repeated Runs

- Treat `mapanything` as the default training/evaluation environment for ACE-DINOv2-LMC. Do not spend attempts on base/system Python for anything importing `torch`, trainers, DINOv2, or DSAC*.
- If a sandboxed command fails with `bwrap: loopback: Failed RTM_NEWADDR`, escalate the next necessary command directly. This is an environment limitation, not a project failure.
- Persistent jobs should be launched with `tmux`; `nohup` and `setsid` have been observed to exit when the exec wrapper cleans up.
- Indoor6 DINO + STGS is a special validated path: use only Indoor6 unless explicitly asked otherwise, use DINOv2 + MapAnything memory, and keep short-run baseline comparability (`fast4_b5_f10`, `buf5M/F10M`, `K64`, `res518`, `buffer_batch_size=1`) unless the ablation changes one of these factors.
- Indoor6 ACE image names and COLMAP image names differ. Build DINO STGS sidecars from `/data/xwh/indoor6/indoor6-colmap/<scene>-tr/sparse/0` with pose fallback (`--image-match-mode auto`), variable sidecar width for `buffer_batch_size=1`, and verify `summary.json` has nonzero `image_pose_matches` and `rows` before training.

## Autonomous Safe Actions

Within this workspace, when the user requests routine experiment work, do not ask for confirmation before safe actions such as launching tmux training/eval jobs, reading logs/results, building non-destructive sidecars, running py_compile/smoke checks, or aggregating metrics. Use escalation directly when the sandbox blocks known-safe project commands.

Before code edits or training from changed code, create a snapshot/provenance record:

```bash
cd /home/xwh/project/ace_depth
bash ace_dinov2_lmc/scripts/agent_safe_snapshot.sh pre_edit <label>
bash ace_dinov2_lmc/scripts/agent_safe_snapshot.sh run_provenance <label> <RUN_ROOT>
```

Snapshots live under `/data/xwh/.tmp/ace_dinov2_lmc_agent_snapshots/` and contain `HEAD`, `git status`, tracked diff, untracked file list, and optional file copies.

Still require explicit user instruction for destructive operations: deleting files/results, killing jobs, resetting checkout state, overwriting checkpoints/run roots, changing global environment/dependencies, or using GPUs outside the requested/current limit.
