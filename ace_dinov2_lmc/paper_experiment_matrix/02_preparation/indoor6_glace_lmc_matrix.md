# Indoor6 GLACE+LMC Training Matrix Preparation

Status: preparation note, 2026-06-03.

## Goal

Fill the paper-critical gap: GLACE+LMC on Indoor6. This tests whether Indoor6 success is not specific to DINO+LMC.

## Local Data State

Confirmed roots:

- ACE scenes: `/home/xwh/data/indoor6_ace/{scene1,scene2a,scene3,scene4a,scene5,scene6}` with `train/` and `test/`.
- WAI scenes: `/home/xwh/data/mapanything-dataset/wai_data/indoor6/<scene>_{train,test,val}` with `images/`, `gt_depth/`, `colmap_depth/`, and `covisibility/`.
- Existing Indoor6 GLACE `features.npy`: none found under `/home/xwh/data/indoor6_ace`; feature extraction is a prerequisite for `model_backend=glace_lmc`.

## Matrix Phases

### Phase 0: GLACE global feature extraction

For every scene/split used by GLACE-LMC, generate `features.npy` in the ACE-style `train/` and `test/` directories. Start with two representative scenes:

- `scene3`: historically strong DINO+LMC scene and important for comparison.
- `scene4a`: another representative hard/large indoor scene.

Then expand to all six scenes.

### Phase 1: Memory extraction

Use GLACE encoder features for memory extraction, with Indoor6 sparse/GT depth contract:

- `DATASET_TYPE=indoor6`
- `DATASET_LOADER=wai` for WAI C1 memory extraction, or ACE loader for ACE-style train root after feature extraction.
- `PATCH_DEPTH_SAMPLING=nearest_valid`
- `depth_valid_range 0.02 100.0`
- BSE/ASB settings inherited from `memory_extraction/run_indoor6_full_baselines_4090.sh`.

Recommended variants:

- `c1_p4_glace_memory`: C1 memory + reference policy gate + post repair.
- `c0_p4_glace_memory`: C0 memory + reference policy gate + post repair, only for sanity/control.

### Phase 2: GLACE+LMC training

Primary training setting:

- `--model_backend glace_lmc`
- `--lmc_flow ace_g`
- `--lmc_mode global`
- start from `--num_latent_tokens 64`, but treat scene-token capacity as a paper-critical ablation rather than a fixed default.
- `--lmc_fusion_target local` should be tested first if negative transfer appears; otherwise default decoder/global path follows Wayspots script.
- sparse-depth-guided buffer sampling enabled with WAI `gt_depth` for Indoor6.
- post-train eval seeds: `1305 2026 4242 7777 9001` for final table; `1305 2026 4242` for smoke.

### Phase 3: scene-token capacity ablation

Current LMC runs use K=64 latent scene tokens. This is compact, but it is much smaller than ACE-G-style map-code settings such as 1024 or 4096 embeddings. The paper needs to show whether LMC is failing because K is too small, or whether compact memory is sufficient.

Recommended sweep:

- Main sweep on `scene3` and `scene4a`: K=64, 128, 256, 512.
- Extended check on one representative scene: K=1024.
- ACE-G-scale reference on one scene only if compute allows: K=4096.
- After selecting the best accuracy/compute point, run all six Indoor6 scenes only at that K.

Report median pose error, threshold accuracy, checkpoint size, latent-token count, training time, eval time, and peak VRAM. The desired paper result is not necessarily the largest K; it is the smallest K that closes most of the gap to the best method.

## Minimal Experiment Matrix

| Priority | Scenes | Variant | Purpose |
|---|---|---|---|
| P0 | scene3, scene4a | GLACE baseline | Establish GLACE Indoor6 base. |
| P1 | scene3, scene4a | GLACE+LMC C1 memory | Main compatibility check. |
| P2 | scene3, scene4a | zero/random memory controls | Confirm memory is not just capacity/regularization. |
| P3 | scene3, scene4a | K=64/128/256/512, optional 1024 | Paper-critical scene-token capacity ablation. |
| P4 | all 6 scenes | best GLACE+LMC variant and best K | Paper table. |
| P5 | one representative scene | optional K=4096 | ACE-G-scale capacity reference. |

## Execution Template

After `features.npy` and memory exist:

```bash
cd /home/xwh/project/ace_depth
CONDA_ENV=mapanything SCENE=scene3 ACE_ROOT=/home/xwh/data/indoor6_ace WAI_ROOT=/home/xwh/data/mapanything-dataset/wai_data/indoor6 MEMORY_PATH=/path/to/indoor6/glace/memory_bse.pt RUN_ROOT=/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6_glace_lmc GPU_ID=0 bash ace_dinov2_lmc/scripts/run_indoor6_glace_lmc_scene.sh
```

## Main Risk

GLACE global features can be harmful on some scenes. If GLACE baseline itself is weak on an Indoor6 scene, do not over-interpret GLACE+LMC failure; report it as a compatibility/boundary result and compare against DINO+LMC.
