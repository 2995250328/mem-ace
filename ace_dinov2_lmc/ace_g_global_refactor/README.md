# ACE-G Planning Corpus Guide

**Absolute root:** `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor`

## 1. Purpose

This document is the navigation entry for the current ACE-G planning corpus under:

```text
ace_dinov2_lmc/ace_g_global_refactor/
```

Its purpose is simple:

1. tell you **which file to open first** for a given task,
2. tell you **where each planning document lives**,
3. give a **concise content summary** for each important file,
4. separate **high-level tracking**, **experiment reference**, **step-by-step implementation notes**, and **long-term research**.

This file is a guide to the planning corpus. It is **not** the source-of-truth experiment table, and it is **not** the per-step implementation plan.

## Quick Links

### Core entry files

- [PLAN.md](./PLAN.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN.md`
- [PLAN_CN.md](./PLAN_CN.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PLAN_CN.md`
- [EXPERIMENT_REFERENCE.md](./EXPERIMENT_REFERENCE.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
- [PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md](./PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md) <!-- updated 2026-06-13: revised fusion-effectiveness direction -->
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md`
- [SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md](./SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md`
- [LONG_TERM_RESEARCH_PLAN.md](./LONG_TERM_RESEARCH_PLAN.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`
- [COMPARE_CPE_B1_scene2a_20260518.md](./COMPARE_CPE_B1_scene2a_20260518.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/COMPARE_CPE_B1_scene2a_20260518.md`

### Current high-priority roadmap files

- [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`
- [steps/16_large_scene_multi_memory_reference_frame_plan.md](./steps/16_large_scene_multi_memory_reference_frame_plan.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan.md`
- [steps/16_large_scene_multi_memory_reference_frame_plan_zh.md](./steps/16_large_scene_multi_memory_reference_frame_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan_zh.md`
- [steps/17_view_token_memory_usage_plan.md](./steps/17_view_token_memory_usage_plan.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan.md`
- [steps/17_view_token_memory_usage_plan_zh.md](./steps/17_view_token_memory_usage_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan_zh.md`
- [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan.md`
- [steps/20_rio10_generalization_diagnostic_plan.md](./steps/20_rio10_generalization_diagnostic_plan.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan.md`
- [steps/21_glace_lmc_source_audited_architecture_plan.md](./steps/21_glace_lmc_source_audited_architecture_plan.md) <!-- updated 2026-05-26: added source-audited GLACE-LMC plugin plan -->
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/21_glace_lmc_source_audited_architecture_plan.md`

---

## 2. Recommended reading order

If you are continuing ACE-G work after a break, use this order:

### A. You want the big picture

Open:

1. `ace_g_global_refactor/PLAN.md`
2. `ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
3. `ace_g_global_refactor/PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md`
4. `ace_g_global_refactor/SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md`
5. `ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`

This gives you:

- the high-level problem map,
- the accepted baseline and rejected directions,
- the current PMRF structural direction,
- the standalone STGS supervision contract,
- the current near-term roadmap.

### B. You want to launch or judge a new experiment

Open:

1. `ace_g_global_refactor/EXPERIMENT_REFERENCE.md`
2. the relevant step document under `ace_g_global_refactor/steps/`

The first file prevents duplicate or already-negative runs. The second tells you the intended mechanism and constraints.

### C. You want to understand a specific architectural direction

Open the relevant step file directly under:

```text
ace_g_global_refactor/steps/
```

Use the index in Section 5 below.

### D. You want broader future directions, not current execution

Open:

1. `ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`
2. optionally `ace_g_global_refactor/PLAN.md`

This is where larger, multi-contract ideas live after they are deferred out of the current stabilization track.

---

## 3. Directory structure and meaning

### `ace_g_global_refactor/PLAN.md`

Role:

- the **high-level tracker**,
- the **priority map**,
- the **onboarding summary** for new agents or future sessions.

Use it when:

- you want to understand the current ACE-G problem decomposition,
- you want to know which problems are already stabilized and which remain open,
- you want the official status framing of the global refactor.

Important note:

- `PLAN.md` explicitly says that concrete implementation notes live one-per-step under `steps/`.

### `ace_g_global_refactor/PLAN_CN.md`

Role:

- synchronized Chinese translation of `PLAN.md`.

Use it when:

- you want a Chinese reading of the high-level tracker.

### `ace_g_global_refactor/EXPERIMENT_REFERENCE.md`

Role:

- the **canonical experiment reference**,
- the **baseline decision table**,
- the **do-not-rerun-unchanged** register.

Use it when:

- you are about to launch a new run,
- you want to check accepted baseline settings,
- you want to know whether a direction was already tested and rejected,
- you need current reference metrics like FGPI-4090.

This should be the first stop before any new experiment proposal.

### `ace_g_global_refactor/thinking.md` <!-- updated 2026-06-12: PMRF/G-PMRF handoff evaluation -->

Role:

- the handoff evaluation for PMRF / G-PMRF,
- the source for why PMRF-v1 is not yet an established contribution,
- the source for mandatory Single / PMRF-v1 / parameter-matched FFN controls and diagnostics before any G-PMRF implementation.

Use it when:

- judging whether PMRF/G-PMRF should continue,
- handing off the PMRF/G-PMRF design to another conversation,
- avoiding premature gates, guards, STGS, adaptive sigma/lambda, or coordinate-anchor residuals before the basic controls pass.

### `ace_g_global_refactor/PROGRESSIVE_MEMORY_REREADING_FUSION_PLAN_ZH.md`

Role:

- the older Chinese structural-method plan for Progressive Memory Re-reading Fusion (PMRF),
- useful background for the PMRF idea but no longer the decision authority,
- superseded for current handoff purposes by `thinking.md`.

Use it when:

- reading the original PMRF design context in Chinese,
- comparing the older PMRF plan against the current evaluation in `thinking.md`.

### `ace_g_global_refactor/SFM_TRACK_GUIDED_MULTI_VIEW_SUPERVISION_ZH.md`

Role:

- the standalone supervision-side STGS contract,
- the source for Xu/SeqACE comparison, sparse XYZ loss, cross-view reprojection loss, and track package schema,
- the current plan for train-only SfM-track supervision without test-time sequence input.

Use it when:

- exporting train-only SfM tracks,
- adding XYZ or cross-view supervision,
- planning PMRF x STGS complementarity experiments.

### `ace_g_global_refactor/LONG_TERM_RESEARCH_PLAN.md`

Role:

- container for broader ideas that are useful but **too broad for the current ACE-G baseline stabilization track**.

Use it when:

- the idea changes multiple contracts at once,
- the idea belongs to future multi-scene / multi-memory / joint-training work,
- the idea should not yet be merged into the near-term roadmap.

### `ace_g_global_refactor/COMPARE_CPE_B1_scene2a_20260518.md`

Role:

- one focused comparison note for a specific experiment family.

Use it when:

- you need the detailed rationale behind the CPE / B1 conclusions,
- you want supporting detail beyond the summary in `EXPERIMENT_REFERENCE.md`.

---

## 4. How to use the `steps/` directory

The `steps/` directory contains **one document per concrete direction, contract fix, or implementation track**.

Roughly speaking, the steps fall into three groups:

### Group A — foundational contracts and hygiene

These define semantics, reproducibility, and consistency. They are the foundation of all later architectural work.

### Group B — focused architectural directions

These introduce specific mechanisms such as geometry injection, multi-level compression/fusion, PointRoPE, multi-memory planning, and view-token usage.

### Group C — roadmap documents

These summarize and prioritize multiple directions rather than describing only one isolated mechanism.

---

## 5. Step-by-step index

Below is the recommended interpretation of each step file.

### Step 01 — [steps/01_deterministic_fps.md](./steps/01_deterministic_fps.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/01_deterministic_fps.md`

Topic:

- deterministic FPS behavior.

Why it matters:

- removes random variation in latent coordinate construction.

Use it when:

- checking whether memory construction is reproducible.

---

### Step 02 — [steps/02_low_risk_contract.md](./steps/02_low_risk_contract.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/02_low_risk_contract.md`

Topic:

- low-risk contract cleanup.

Why it matters:

- establishes safer semantics before larger changes.

Use it when:

- tracing early contract/hygiene fixes in the ACE-G refactor.

---

### Step 03 — [steps/03_s1_sampled_loss_step_mode.md](./steps/03_s1_sampled_loss_step_mode.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/03_s1_sampled_loss_step_mode.md`

Topic:

- S1 sampled loss step-mode semantics.

Why it matters:

- helps explain why `s1_loss_step_mode=per_iter` became important.

Use it when:

- reviewing S1 behavior or comparing old vs current S1 settings.

---

### Step 04 — [steps/04_lmc_mode_authority.md](./steps/04_lmc_mode_authority.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/04_lmc_mode_authority.md`

Topic:

- authority of requested vs effective `lmc_mode`.

Why it matters:

- prevents commands that say `global` from silently becoming effective-local.

Use it when:

- validating whether a run is truly global,
- auditing fallback behavior.

---

### Step 05 — [steps/05_geomatch_near_term.md](./steps/05_geomatch_near_term.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/05_geomatch_near_term.md`

Topic:

- the early fusion-geometry / GeoMatch / geometry-injection direction.

Why it matters:

- frames the first falsifiable geometry-side changes in fusion.

Use it when:

- understanding the origin of A0/A1/A2 and progressive geometry injection work.

---

### Step 06 — [steps/06_compressor_geometry_contract.md](./steps/06_compressor_geometry_contract.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/06_compressor_geometry_contract.md`

Topic:

- compressor geometry contract.

Why it matters:

- defines how key layer choice, coordinate scale, and multi-level memory should be interpreted.

Use it when:

- thinking about key/value hierarchy,
- planning multi-level compression,
- checking scene-scale / geometry assumptions.

---

### Step 07 — [steps/07_loss_contract.md](./steps/07_loss_contract.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/07_loss_contract.md`

Topic:

- unified loss-path semantics.

Why it matters:

- prevents S1/S2 path differences from being misdiagnosed as architecture effects.

Use it when:

- reasoning about reprojection supervision consistency.

---

### Step 08 — [steps/08_non_arch_hygiene.md](./steps/08_non_arch_hygiene.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/08_non_arch_hygiene.md`

Topic:

- non-architectural hygiene work.

Why it matters:

- keeps experiment semantics clean without changing the model itself.

Use it when:

- reviewing logging, config, bookkeeping, or reproducibility cleanup.

---

### Step 09 — [steps/09_module_mode_contract.md](./steps/09_module_mode_contract.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/09_module_mode_contract.md`

Topic:

- module mode consistency.

Why it matters:

- makes module/training behavior easier to reason about.

Use it when:

- tracing training/eval/frozen-module mode semantics.

---

### Step 10 — [steps/10_runtime_observability_and_semantics.md](./steps/10_runtime_observability_and_semantics.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/10_runtime_observability_and_semantics.md`

Topic:

- runtime observability.

Why it matters:

- adds the diagnostics needed to understand token usage, entropy, norms, and semantic behavior.

Use it when:

- you need to know what the system logs or should log during runs.

---

### Step 11 — [steps/11_levelwise_latent_merge_plan.md](./steps/11_levelwise_latent_merge_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/11_levelwise_latent_merge_plan.md`

Topic:

- levelwise latent merge / B3-lite direction.

Why it matters:

- this is the important failed multi-level compression attempt that later motivated safer residual designs.

Use it when:

- you want the detailed history of the rejected B3-lite path.

---

### Step 12 — [steps/12_geobias_rbf_and_fourier_pe_v2_plan.md](./steps/12_geobias_rbf_and_fourier_pe_v2_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/12_geobias_rbf_and_fourier_pe_v2_plan.md`

Topic:

- geometry bias RBF and improved Fourier positional encoding.

Why it matters:

- records the geometry-conditioning branch that explored bias redesign and Fourier PE improvements.

Use it when:

- checking the rationale behind GBR-main and FPE-main style work.

---

### Step 13 — [steps/13_query_side_dpt_adapter_multilevel_plan.md](./steps/13_query_side_dpt_adapter_multilevel_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/13_query_side_dpt_adapter_multilevel_plan.md`

Topic:

- query-side DPT-style / multi-level adapter planning.

Why it matters:

- this is the original backbone-side multi-level dense prediction direction.

Use it when:

- understanding the earlier multi-level idea before it was partially replaced by safer fusion-side internal refinement.

Current status:

- still useful as background, but no longer the preferred first implementation for multi-level fusion.

---

### Step 14 — [steps/14_point_rope_and_continuous_relative_bias_plan.md](./steps/14_point_rope_and_continuous_relative_bias_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/14_point_rope_and_continuous_relative_bias_plan.md`

Topic:

- PointRoPE and continuous relative bias planning.

Why it matters:

- this is the direct precursor to the current PointRoPE direction.

Use it when:

- checking the geometry encoding and bias hypotheses before the consolidated roadmap.

---

### Step 15 — [steps/15_pointrope_and_multilevel_fusion_roadmap.md](./steps/15_pointrope_and_multilevel_fusion_roadmap.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/15_pointrope_and_multilevel_fusion_roadmap.md`

Topic:

- current **near-term roadmap**.

Why it matters:

- this is the most important step file for the current generation of architectural planning.

It currently summarizes:

- the accepted FGPI-style reference,
- completed geometry results,
- rejected or deferred paths,
- priority ordering across PointRoPE, fusion-side refinement, compression-side residual multi-level work, and bias directions,
- the updated Candidate C split:
  - preferred C1 = Cascading Internal Fusion Assembly,
  - deferred C2 = backbone multi-layer query adapter.

Use it when:

- you want to know “what should we do next?”
- you want the current architectural priority order,
- you want the latest synthesis across multiple completed experiment families.

---

### Step 16 — [steps/16_large_scene_multi_memory_reference_frame_plan.md](./steps/16_large_scene_multi_memory_reference_frame_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan.md`

Topic:

- large-scene multi-memory planning with reference-frame awareness.

Why it matters:

- defines how to think about multiple MapAnything memories, each with different reference-conditioned coordinates/features.

Core conclusion:

- do not directly concatenate multiple memories as the first path,
- prefer independent local memory bundles plus routing / geometric pose selection.

Use it when:

- thinking about large scenes,
- planning multi-memory routing,
- deciding how to organize multiple local memories.

Chinese reference:

- [steps/16_large_scene_multi_memory_reference_frame_plan_zh.md](./steps/16_large_scene_multi_memory_reference_frame_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/16_large_scene_multi_memory_reference_frame_plan_zh.md`

---

### Step 17 — [steps/17_view_token_memory_usage_plan.md](./steps/17_view_token_memory_usage_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan.md`

Topic:

- conservative use of memory-side view tokens (`all_scale_tokens`).

Why it matters:

- records how to use stored per-view global tokens without contaminating the current baseline path.

Core conclusion:

- treat `all_scale_tokens` as a side-channel,
- first do diagnostics,
- then optional attention pooling / latent modulation,
- later consider memory routing,
- do not inject raw tokens into dense fusion yet,
- do not enable query CLS guidance yet.

Use it when:

- investigating camera/view token usage,
- deciding whether these tokens belong in memory-side modulation or future routing.

Chinese reference:

- [steps/17_view_token_memory_usage_plan_zh.md](./steps/17_view_token_memory_usage_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/17_view_token_memory_usage_plan_zh.md`

---

### Step 18 — [steps/18_rio10_wai_training_fix_log.md](./steps/18_rio10_wai_training_fix_log.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/18_rio10_wai_training_fix_log.md`

Topic:

- RIO10 WAI/ACE backend resize and sparse-depth attachment fix log.

Why it matters:

- records why direct WAI training failed on DINOv2 patch-size constraints,
- records the corrected ACE-backend RIO10 training path,
- records sparse-depth matching behavior for RIO10.

Use it when:

- debugging RIO10 dataset/backend issues,
- checking sparse-depth attachment for ACE-backend training,
- preparing RIO10 runs before the conservative migration recipe.

---

### Step 19 — [steps/19_rio10_conservative_indoor6_migration_plan.md](./steps/19_rio10_conservative_indoor6_migration_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan.md`

Topic:

- conservative migration from the best Indoor6 ACE-G baseline to RIO10.

Why it matters:

- defines the lowest-risk RIO10 stabilization recipe before adding new architecture.

Core conclusion:

- preserve true-global `per_iter` Indoor6 baseline semantics,
- train RIO10 with ACE backend,
- extract memory through WAI/MapAnything ASB single-forward global memory,
- use `patch_depth_sampling=nearest_valid`,
- use sparse depth only for valid-coordinate guided sampling,
- keep auxiliary sparse-depth/reference supervision off.

Use it when:

- launching the next RIO10 stabilization run,
- comparing guided sampling vs no guided sampling,
- deciding what variables must remain fixed before PointRoPE/CIFA/multi-memory extensions.

Chinese reference:

- [steps/19_rio10_conservative_indoor6_migration_plan_zh.md](./steps/19_rio10_conservative_indoor6_migration_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/19_rio10_conservative_indoor6_migration_plan_zh.md`

---

### Step 20 — [steps/20_rio10_generalization_diagnostic_plan.md](./steps/20_rio10_generalization_diagnostic_plan.md)

Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan.md`

Topic:

- diagnostic plan for poor RIO10 generalization before further architecture changes.

Why it matters:

- separates data/eval contract issues, vanilla DINO ACE baseline weakness, LMC/memory/fusion failure, sparse-guided sampling bias, and training-parameter effects.

Core conclusion:

- treat the current RIO10 run as incomplete partial evidence,
- audit data/pose/calibration and train-set eval first,
- establish a same-split vanilla DINO ACE baseline,
- only then ablate sparse sampling ratio, S2 fusion, training parameters, and memory coverage,
- do not add PointRoPE/CIFA/multi-memory/view-token changes until the failure mode is identified.

Use it when:

- RIO10 generalization is weak and you need to locate the cause,
- deciding whether the issue is data/eval, baseline, or LMC-specific,
- planning the next diagnostic experiment matrix before more architecture work.

Chinese reference:

- [steps/20_rio10_generalization_diagnostic_plan_zh.md](./steps/20_rio10_generalization_diagnostic_plan_zh.md)
  Absolute path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/ace_g_global_refactor/steps/20_rio10_generalization_diagnostic_plan_zh.md`

---

## 6. Quick task-to-document map

### “I want to know the current accepted baseline.”

Open:

- `EXPERIMENT_REFERENCE.md`

### “I want the current overall architecture roadmap.”

Open:

- `PLAN.md`
- `steps/15_pointrope_and_multilevel_fusion_roadmap.md`

### “I want to know what not to rerun.”

Open:

- `EXPERIMENT_REFERENCE.md`

### “I want to understand multi-level compression/fusion history.”

Open:

- `steps/11_levelwise_latent_merge_plan.md`
- `steps/13_query_side_dpt_adapter_multilevel_plan.md`
- `steps/15_pointrope_and_multilevel_fusion_roadmap.md`

### “I want the current preferred multi-level direction.”

Open:

- `steps/15_pointrope_and_multilevel_fusion_roadmap.md`

Specifically:

- Candidate C1 = Cascading Internal Fusion Assembly.

### “I want to understand large-scene / multi-memory planning.”

Open:

- `steps/16_large_scene_multi_memory_reference_frame_plan.md`

### “I want to understand camera/view token usage.”

Open:

- `steps/17_view_token_memory_usage_plan.md`

### “I want to migrate the Indoor6 best baseline to RIO10 conservatively.”

Open:

- `steps/19_rio10_conservative_indoor6_migration_plan.md`
- `steps/18_rio10_wai_training_fix_log.md`

### “I want to diagnose why RIO10 generalization is weak.”

Open:

- `steps/20_rio10_generalization_diagnostic_plan.md`
- `steps/19_rio10_conservative_indoor6_migration_plan.md`
- `steps/18_rio10_wai_training_fix_log.md`

### “I want broader future directions beyond the current stabilization track.”

Open:

- `LONG_TERM_RESEARCH_PLAN.md`

---

## 7. Practical usage rule

Before proposing any new ACE-G experiment, use this mini-checklist:

1. read `EXPERIMENT_REFERENCE.md`,
2. confirm the direction is not already rejected or duplicated,
3. read the relevant step file under `steps/`,
4. check `PLAN.md` if you need the surrounding problem map,
5. if the idea changes multiple contracts at once, move it mentally to `LONG_TERM_RESEARCH_PLAN.md` territory rather than forcing it into the near-term roadmap.

---

## 8. Final summary

If you only remember four files, remember these:

1. `PLAN.md` — high-level tracker and problem map.
2. `EXPERIMENT_REFERENCE.md` — baseline decisions and do-not-rerun rules.
3. `steps/15_pointrope_and_multilevel_fusion_roadmap.md` — current near-term roadmap.
4. `LONG_TERM_RESEARCH_PLAN.md` — broader future work beyond the current stabilization track.

Everything else in `steps/` is the detailed local document for a specific contract, mechanism, or branch of the ACE-G planning history.
