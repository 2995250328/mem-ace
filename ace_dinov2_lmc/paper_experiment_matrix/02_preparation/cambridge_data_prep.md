# Cambridge Data Preparation Plan

Status: preparation note, updated 2026-06-03.

## Local Data State

Confirmed local roots:

- ACE-style Cambridge scenes: `/data/xwh/Cambridge/Cambridge_{GreatCourt,KingsCollege,OldHospital,ShopFacade,StMarysChurch}`.
- WAI-style Cambridge scenes: `/data/xwh/Cambridge_wai/Cambridge_<Scene>_{train,test}`.
- ACE checkpoints: `/data/xwh/checkpoints/ace_models/Cambridge/*.pt`.
- Every ACE-style scene has `reconstruction.nvm`, so missing sparse depth can be generated with `tools/project_cambridge_nvm_sparse_depth.py`.

## Data Audit Snapshot

Audited on 2026-06-03:

| Scene | Train RGB | Test RGB | NVM | GLACE features | Sparse depth |
|---|---:|---:|---|---|---|
| Cambridge_GreatCourt | 1531 | 760 | yes | train/test yes | missing |
| Cambridge_KingsCollege | 1220 | 343 | yes | missing | train/test yes |
| Cambridge_OldHospital | 895 | 182 | yes | missing | train/test yes |
| Cambridge_ShopFacade | 231 | 103 | yes | missing | missing |
| Cambridge_StMarysChurch | 1487 | 530 | yes | missing | missing |

RGB, pose, and calibration counts are aligned for all train/test splits.

## Required Data Contract

For each Cambridge scene used by LMC, verify:

- ACE train/test roots contain `rgb/`, `poses/`, `calibration/` with equal file counts.
- If `model_backend=glace_lmc` or ACE-FCN Stage2 with GLACE globals is used, both train and test split must contain `features.npy` aligned one-to-one with sorted RGB files.
- If sparse-depth-guided sampling is enabled, `train/sparse_depth` or equivalent WAI depth must exist and match image stems.
- Memory extraction must record `feature_source`, `coord_source`, `output_subsample`, `scene_center`, and depth source.

## Preflight Commands

```bash
cd /home/xwh/project/ace_depth
find /data/xwh/Cambridge -maxdepth 4 -type f -name features.npy | sort
find /data/xwh/checkpoints/ace_models/Cambridge -maxdepth 2 -type f -name "*.pt" | sort
```

Detailed audit command:

```bash
cd /home/xwh/project/ace_depth
python - <<'PY_AUDIT'
from pathlib import Path
root=Path('/data/xwh/Cambridge')
for scene in sorted(root.glob('Cambridge_*')):
    if not scene.is_dir():
        continue
    print(scene.name, 'nvm', (scene/'reconstruction.nvm').is_file())
    for split in ['train','test']:
        sr=scene/split
        def count(d, pats=('*.png','*.jpg')):
            dd=sr/d
            if not dd.is_dir():
                return -1
            return sum(len(list(dd.glob(p))) for p in pats)
        print(' ', split,
              'rgb', count('rgb'),
              'poses', count('poses',('*.txt',)),
              'calib', count('calibration',('*.txt',)),
              'features', (sr/'features.npy').is_file(),
              'sparse_depth', count('sparse_depth',('*.npz','*.npy')))
PY_AUDIT
```

## GLACE Feature Preparation

Use the GLACE repository feature extractor before GLACE-LMC training. The extractor takes a scene root and writes `train/features.npy` and `test/features.npy`.

Template:

```bash
cd /home/xwh/project/glace
conda run --no-capture-output -n mapanything python datasets/extract_features.py   /data/xwh/Cambridge/Cambridge_KingsCollege   --checkpoint /home/xwh/project/glace/CVPR23_DeitS_Rerank.pth   --batch_size 256   --num_workers 4
```

Do not use a `--scene/--encoder_path/--output` template; that is not the GLACE extractor interface.

## Sparse Depth Preparation

For scenes missing sparse depth, use the existing NVM projection script:

```bash
cd /home/xwh/project/ace_depth
python -m ace_dinov2_lmc.tools.project_cambridge_nvm_sparse_depth   /data/xwh/Cambridge   --scenes Cambridge_GreatCourt Cambridge_ShopFacade Cambridge_StMarysChurch   --max-depth-m 1000
```

After this, re-run the data audit and verify sparse-depth count matches RGB count.

## First Cambridge Experiment Scope

Run only one scene first:

1. `Cambridge_KingsCollege`: sparse depth exists; only GLACE features are missing.
2. `Cambridge_OldHospital`: sparse depth exists and is smaller test-wise; only GLACE features are missing.
3. `Cambridge_GreatCourt`: GLACE features exist, but sparse depth is missing and the scene is larger; use only for a GLACE-feature smoke or after sparse-depth projection.

Initial matrix:

- ACE checkpoint eval if needed.
- DINOACE baseline or existing checkpoint if available.
- DINO+LMC one memory setting.
- GLACE baseline if GLACE checkpoint/features exist.
- GLACE+LMC one memory setting only.

Do not run full Cambridge ablations until one scene passes data-contract and memory sanity checks.


## Adapted Scripts Added

These scripts are Cambridge-specific and should be used instead of the old root-level ACE Cambridge scripts when preparing LMC experiments:

- `ace_dinov2_lmc/tools/audit_cambridge_data.py`: audits ACE-style Cambridge split counts, GLACE `features.npy`, sparse depth, and NVM availability.
- `ace_dinov2_lmc/scripts/prepare_cambridge_sparse_depth.sh`: projects `reconstruction.nvm` into `train/test/sparse_depth` for scenes missing sparse depth.
- `ace_dinov2_lmc/scripts/extract_cambridge_glace_features.sh`: runs GLACE global feature extraction with the correct GLACE extractor interface.
- `ace_dinov2_lmc/scripts/extract_cambridge_glace_memory.sh`: extracts GLACE-encoder feature-space memory from an ACE-style Cambridge train split.
- `ace_dinov2_lmc/scripts/extract_cambridge_dino_memory.sh`: extracts DINOv2 feature-space memory from an ACE-style Cambridge train split.
- `ace_dinov2_lmc/scripts/run_cambridge_glace_lmc_scene.sh`: runs one Cambridge GLACE+LMC scene after features and memory exist.
- `ace_dinov2_lmc/scripts/run_cambridge_dino_lmc_scene.sh`: runs one Cambridge DINO+LMC scene after DINO memory exists.

## Recommended First Execution Order

Start with `Cambridge_KingsCollege` because sparse depth exists and only GLACE features are missing.

```bash
cd /home/xwh/project/ace_depth
python ace_dinov2_lmc/tools/audit_cambridge_data.py /data/xwh/Cambridge --scenes Cambridge_KingsCollege
```

Generate GLACE global features:

```bash
cd /home/xwh/project/ace_depth
SCENES_STR=Cambridge_KingsCollege DRY_RUN=false bash ace_dinov2_lmc/scripts/extract_cambridge_glace_features.sh
```

Extract GLACE memory:

```bash
cd /home/xwh/project/ace_depth
SCENE=Cambridge_KingsCollege GPU_ID=0 bash ace_dinov2_lmc/scripts/extract_cambridge_glace_memory.sh
```

Run GLACE+LMC after copying the printed `memory_path` into `MEMORY_PATH`:

```bash
cd /home/xwh/project/ace_depth
SCENE=Cambridge_KingsCollege GPU_ID=0 MEMORY_PATH=/path/to/memory_bse.pt bash ace_dinov2_lmc/scripts/run_cambridge_glace_lmc_scene.sh
```

For DINO+LMC, use `extract_cambridge_dino_memory.sh` and then `run_cambridge_dino_lmc_scene.sh` with the resulting memory path.

## Existing Cambridge Script Caveat

The root-level scripts `/home/xwh/project/ace_depth/scripts/train_cambridge.sh` and `train_cambridge_ensemble.sh` are ACE baseline scripts. The ensemble script follows the ACE/Poker clustering protocol (`--num_clusters`, `--cluster_idx`) and is not directly comparable to a single-scene LMC run unless we explicitly reproduce the same cluster/merge protocol for LMC. Treat them as baseline references, not as direct LMC launchers.
