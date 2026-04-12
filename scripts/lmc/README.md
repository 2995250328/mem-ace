# LMC Ablation Scripts

This directory contains batch scripts for ACE DINOv2 LMC (Latent Memory Compression). For full documentation, see [LMC.md](../../LMC.md) in the project root.

## ⚠️ Script Compatibility Notice

**The scripts `ablation_compression_ratio.sh`, `ablation_multimodal.sh`, `ablation_compression_onoff.sh`, and `example_workflow.sh` use parameters that do NOT exist in the current implementation:**

- `--enable_compression`, `--compression_ratio`
- `--use_depth`, `--use_superpoint`, `--use_intrinsics`

These scripts were written for a different LMC variant (tasks/ace/ with autoencoder compression and multi-modal fusion). **They will fail if run as-is.** You must adapt them to use the actual parameters (`--use_lmc`, `--memory_path`, `--lmc_mode`) or implement the alternate variant.

## Current Implementation Ablation

The current LMC uses GeoLMC with pre-saved memory. Use the following patterns:

### LMC On/Off

```bash
# Vanilla baseline (no LMC)
./train_ace_dinov2_lmc.py SCENE output/vanilla.pt --use_lmc False --device cuda:0

# LMC (requires pre-saved memory file)
./train_ace_dinov2_lmc.py SCENE output/lmc.pt \
    --use_lmc True --memory_path /path/to/memory.pt \
    --device cuda:0
```

### GeoLMC Mode Ablation

```bash
MEMORY=/path/to/memory.pt
SCENE=/mnt/storage/xwh/7Scenes/pgt_7scenes_chess
OUTPUT=output/ablation_mode

for mode in global local hierarchical learned; do
    ./train_ace_dinov2_lmc.py "$SCENE" "$OUTPUT/chess_${mode}.pt" \
        --use_lmc True --memory_path "$MEMORY" \
        --lmc_mode $mode --device cuda:0
    
    ./test_ace_dinov2_lmc.py "$SCENE" "$OUTPUT/chess_${mode}.pt" \
        --session $mode --device cuda:0
done
```

### Usage Pattern

```bash
./scripts/lmc/<script_name>.sh <scene_path> <output_dir> [device]
```

## Adapting Existing Scripts

To make the existing scripts work with the current implementation:

1. Replace `--enable_compression False` with `--use_lmc False`
2. Replace `--enable_compression True --compression_ratio X` with `--use_lmc True --memory_path /path/to/memory.pt --lmc_mode global`
3. Remove `--use_depth`, `--use_superpoint`, `--use_intrinsics` (not supported)
4. For mode ablation, loop over `--lmc_mode global local hierarchical learned`

## Tips

- **Memory**: Reduce `--batch_size 2048` or `--buffer_batch_size 5` if OOM
- **Logging**: `./scripts/lmc/ablation_*.sh ... 2>&1 | tee ablation.log`
