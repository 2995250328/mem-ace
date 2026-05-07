---
name: ace-training-command-composer
description: Use when composing or reviewing training commands for ACE-DINOv2-LMC. Focus on correct project root, scene path, output naming, memory path, presets, GPU assignment, batch/runtime knobs, and comparability between runs.
---

# ACE Training Command Composer

Use this skill when generating or validating train commands in this repository.

## Workflow

1. Normalize working directory.
   Commands should normally run from `/home/xwh/project/ace_depth`.
2. Resolve the training mode.
   Distinguish vanilla vs `use_lmc=True`, plus `lmc_flow`, preset, and memory usage.
3. Compose required arguments.
   Scene path, output checkpoint name, backend, device, eval device, memory path, experiment subdir, and preset.
4. Compose optional runtime knobs.
   Batch size, `buffer_on_cpu`, `buffer_batch_size`, `s1_batch_size`, half precision, resume flags, and evaluation toggles.
5. Preserve comparability.
   Call out whether a command is same-recipe, throughput-only tweak, or training-dynamics change.

## Output

- final runnable command
- naming rationale
- GPU assignment note
- comparability note

## Rules

- Do not generate commands from the wrong working directory.
- If using automatic timestamps, define them explicitly.
- Distinguish throughput-only changes from changes that alter optimization behavior.
- Keep scene, memory, and output checkpoint naming aligned.
