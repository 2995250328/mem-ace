# Idea: Integrate BSE Memory into DINOv2-LMC Training

## Goal
Replace the existing simple-voxel memory loading in the DINOv2-LMC trainer with the BSE (Bilateral Supervoxel Extraction) memory output from the memory_extraction pipeline. Validate whether BSE-based memory construction produces better relocalization results than the previous voxelization approach.

## Constraints
- **Preserve existing LMC logic**: Do NOT modify the LMC module, S1/S2 training loop, or regression head.
- **Use only compatible fields**: Extract 3D point positions and corresponding features from BSE memory. Ignore new fields like ray directions, ray strategy maps, Plücker encodings, etc.
- **Minimal changes**: Only modify the memory loading/adapter layer. The rest of the training pipeline stays byte-for-byte identical.

## Scope
1. Understand the current memory format consumed by `trainer_dinov2_lmc.py`
2. Understand the new BSE memory format saved by `memory_extraction/run_memory_extraction.py`
3. Write an adapter that loads BSE memory and presents it in the format expected by the trainer
4. Wire the adapter into the training entry point
5. Run experiments comparing BSE vs. simple-voxel memory on 7-Scenes

## Success Criterion
- Training runs successfully with BSE memory
- Quantitative comparison (pct<5cm, median error) shows whether BSE is better
