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
