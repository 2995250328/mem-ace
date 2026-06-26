---
name: lmc-evaluation-directory-layout
description: Standardize ACE-DINOv2-LMC / GLACE / ACE-G experiment output directories under /data/xwh/ace_dinov2_lmc/04_evaluation. Use whenever creating, launching, naming, auditing, summarizing, or migrating evaluation run roots so new experiments are grouped by dataset, track, method, date, scope, protocol, and GPU instead of flat ad-hoc folders.
---

# LMC Evaluation Directory Layout

Use this skill before creating any new directory under:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation
```

The goal is to make future experiments searchable by dataset and research track while preserving enough detail in the leaf name for quick terminal inspection.

## Canonical Layout

New runs must use:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/<dataset>/<track>/<method>/<YYYYMMDD>_<scope>_<protocol>_<gpu_tag>/
```

Allowed core values:

- `dataset`: `wayspots`, `cambridge`, `indoor6`, `shared`
- `track`: `baseline`, `stage1`, `stage2`, `fusion`, `training_efficiency`, `reproduction`, `diagnostics`, `paper_results`, `memory`, `scratch`
- `method`: short stable method slug, e.g. `single`, `pmrf_v3`, `concat_glace`, `glace_official`, `stage2_r2_fusion`
- `scope`: scene set slug, e.g. `bears_sq`, `all`, `court_stmary`, `kings`
- `protocol`: the main schedule/eval contract, e.g. `it10_buf10m_final12m_h256`, `official_30k_h64`
- `gpu_tag`: `gpu0`, `gpu01`, `gpu23`, `cpu`, or `nogpu`

Examples:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge/stage2/concat_glace/20260626_kings_it10_buf10m_final12m_h256_gpu0
/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge/baseline/glace_official/20260626_all_official_30k_h64_gpu1
/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/fusion/pmrf_v3/20260623_bears_sq_1iter_h256_gpu01
/data/xwh/ace_dinov2_lmc/04_evaluation/shared/diagnostics/dsac_determinism/20260619_seed_repro_h256_gpu01
```

## Required Run Root Contents

Every new run root should contain:

```text
manifest.json        # created before launch
logs/                # tmux, train, eval, and post-train logs
summaries/           # copied or generated aggregate summaries when available
status.tsv           # matrix status for multi-run launches when applicable
matrix.tsv           # planned config table for matrices when applicable
```

Nested trainer output directories may keep their existing project-specific structure. The canonical run root is the stable entry point.

`manifest.json` must record at least:

- dataset, track, method, scope, protocol, gpu_tag
- scenes
- command or launch script path
- code reference or local diff status when known
- metric policy, especially whether Cambridge uses median error only
- status: `planned`, `running`, `complete`, `failed`, or `archived`

## Creating a New Run Root

Prefer the bundled helper:

```bash
cd /home/xwh/project/ace_depth
python ace_dinov2_lmc/.codex_skills/lmc-evaluation-directory-layout/scripts/make_lmc_eval_run_root.py \
  --dataset cambridge \
  --track stage2 \
  --method concat_glace \
  --scope kings \
  --protocol it10_buf10m_final12m_h256 \
  --gpus 0 \
  --scenes Cambridge_KingsCollege \
  --create
```

The helper prints `RUN_ROOT=...`, creates `logs/`, `summaries/`, `artifacts/`, `scripts/`, and writes `manifest.json`.

If a launch script constructs `RUN_ROOT` manually, it must match the canonical layout exactly. Do not create new flat top-level folders under `04_evaluation`.

## Historical Directory Cleanup

Do not move old results while active tmux jobs, pending summaries, or referenced checkpoints depend on them.

For historical cleanup:

1. Generate an inventory:

   ```bash
   python ace_dinov2_lmc/.codex_skills/lmc-evaluation-directory-layout/scripts/inventory_lmc_eval_dirs.py \
     --base /data/xwh/ace_dinov2_lmc/04_evaluation \
     --output /tmp/lmc_eval_inventory.tsv
   ```

2. Review dataset/track classifications manually.
3. Migrate only after approval, preferably by copying or moving one group at a time.
4. Leave a migration map, e.g. `_migration_YYYYMMDD.tsv`, with old path, new path, and reason.
5. Preserve decision-critical files: `summary.tsv`, `eval_summary_*.txt`, `test_*.txt`, launch scripts, logs, checkpoints, and manifests.

For uncertain legacy folders, use:

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/shared/scratch/legacy_flat/<old_name>
```

only after confirming they are not active inputs.

## Naming Rules

- Include the dataset in the hierarchy, not only in the leaf name.
- Include the main protocol in the leaf name: iterations, buffer, final buffer, hypotheses, or official schedule.
- Include GPU tag in the leaf name.
- Keep leaf names concise; put detailed flags in `manifest.json`, `matrix.tsv`, and logs.
- Avoid vague names: `test`, `new`, `run`, `matrix`, `tmp`, `train`, `exp`.
- Avoid mixing unrelated datasets in one run root. Use `shared/diagnostics` only for cross-dataset tooling or infrastructure checks.

## Interaction With Result Reporting

When summarizing results, report the canonical run root first. If results still live in legacy flat folders, say `legacy-flat` explicitly so the user knows the path has not been normalized.

For Cambridge, primary aggregation is median translation/rotation. For Wayspots and Indoor6, use the project metric policy from `lmc-experiment-standard-workflow`.
