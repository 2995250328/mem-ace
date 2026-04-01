#!/usr/bin/env python3
"""Check layers_idx in memory files."""

import torch

bse_path = '/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_sor_l2/20260331_140851/memory_bse.pt'
pooled_path = '/home/xwh/project/map-anything-experiments/Indoor6_scene2a_train_pooled_GT.pt'

print("BSE memory (raw format):")
bse = torch.load(bse_path, map_location='cpu')
print(f"  Keys: {list(bse.keys())}")
print(f"  layers_idx: {bse.get('layers_idx', 'NOT FOUND')}")
if 'features' in bse:
    print(f"  features shape: {bse['features'].shape}")
if 'pooled_features' in bse:
    print(f"  pooled_features shape: {bse['pooled_features'].shape}")

print("\nPooled memory:")
pooled = torch.load(pooled_path, map_location='cpu')
print(f"  Keys: {list(pooled.keys())}")
print(f"  layers_idx: {pooled.get('layers_idx', 'NOT FOUND')}")
print(f"  pooled_features shape: {pooled['pooled_features'].shape}")
