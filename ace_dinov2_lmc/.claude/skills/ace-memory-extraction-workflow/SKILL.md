---
name: ace-memory-extraction-workflow
description: Use when working on pooled memory extraction for ACE-DINOv2-LMC. Focus on extraction commands, dataset/view-selection contracts, output directory naming, clustered vs non-clustered memory, and validation of generated memory artifacts.
---

# ACE Memory Extraction Workflow

Use this skill for `memory_extraction/` work in this repository.

## Scope

- `extract_memory.sh`
- `run_memory_extraction.py`
- view-selection modes and their env vars
- output directory structure under `memory_extraction/04_evaluation/memory_extract`
- clustered and non-clustered memory outputs

## Workflow

1. Identify the extraction recipe.
   Record dataset type, scene, `N_VIEWS`, `POOL_MODE`, `CONTRACT_MODE`, view mode, gating, repair, clustering, and output root.
2. Resolve command style.
   Distinguish between shell-env-driven extraction and direct Python invocation.
3. Verify view-selection contract.
   Check whether the run is `fps_flat`, `original_multiview`, `anchor_support`, adaptive ASB, or cluster fallback.
4. Verify outputs.
   Confirm expected files such as `memory_bse.pt`, `memory_bse.clustered.pt`, `memory_bse.cluster_*.pt`, `cluster_fallback_plan.json`, `memory_policy_report.json`, `extraction_config.json`, and `extraction_log.txt`.
5. Record provenance.
   Tie every memory artifact back to exact scene, recipe, timestamp directory, and extraction flags.

## Output

- extraction command
- resolved recipe summary
- expected output files
- validation checklist

## Rules

- Treat env vars in `extract_memory.sh` as public API.
- Do not assume `memory_bse.pt` means the recipe is wrong; check `POOL_MODE` first.
- Distinguish clustered memory, per-cluster memory, and single-memory outputs explicitly.
- Prefer exact timestamp directories over “latest” wording.
