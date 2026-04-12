#!/usr/bin/env python3
"""Compare BSE memory vs Pooled memory formats."""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

from utils_lmc import load_memory_features


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def analyze_features(features, name):
    """Analyze feature statistics."""
    print(f"\n{name} Features:")
    print(f"  Shape: {features.shape}")
    print(f"  Dtype: {features.dtype}")
    print(f"  Mean: {features.mean():.6f}")
    print(f"  Std: {features.std():.6f}")
    print(f"  Min: {features.min():.6f}")
    print(f"  Max: {features.max():.6f}")
    print(f"  Norm (mean): {torch.norm(features, dim=-1).mean():.6f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(features).any()
    has_inf = torch.isinf(features).any()
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")

    # Distribution percentiles (sample to avoid memory issues)
    flat = features.flatten()
    if flat.numel() > 10_000_000:
        indices = torch.randperm(flat.numel())[:10_000_000]
        flat = flat[indices]
    percentiles = [0, 25, 50, 75, 100]
    values = [torch.quantile(flat, p/100.0).item() for p in percentiles]
    print(f"  Percentiles (0/25/50/75/100): {[f'{v:.4f}' for v in values]}")


def compare_memory_files(bse_path, pooled_path):
    """Compare two memory files."""

    print_section("Loading Memory Files")
    print(f"BSE Memory: {bse_path}")
    print(f"Pooled Memory: {pooled_path}")

    # Load BSE memory
    print("\nLoading BSE memory...")
    bse_dict = load_memory_features(bse_path, device='cpu')

    # Load Pooled memory
    print("Loading Pooled memory...")
    pooled_dict = load_memory_features(pooled_path, device='cpu')

    # Compare keys
    print_section("Dictionary Keys Comparison")
    bse_keys = set(bse_dict.keys())
    pooled_keys = set(pooled_dict.keys())

    print(f"BSE keys: {sorted(bse_keys)}")
    print(f"Pooled keys: {sorted(pooled_keys)}")
    print(f"Common keys: {sorted(bse_keys & pooled_keys)}")
    print(f"BSE only: {sorted(bse_keys - pooled_keys)}")
    print(f"Pooled only: {sorted(pooled_keys - bse_keys)}")

    # Compare features
    print_section("Feature Comparison")
    analyze_features(bse_dict['pooled_features'], "BSE")
    analyze_features(pooled_dict['pooled_features'], "Pooled")

    # Compare points
    print_section("Points Comparison")
    bse_points = bse_dict['pooled_points']
    pooled_points = pooled_dict['pooled_points']

    print(f"\nBSE Points:")
    print(f"  Shape: {bse_points.shape}")
    print(f"  Mean: {bse_points.mean(dim=0)}")
    print(f"  Std: {bse_points.std(dim=0)}")
    print(f"  Min: {bse_points.min(dim=0).values}")
    print(f"  Max: {bse_points.max(dim=0).values}")

    print(f"\nPooled Points:")
    print(f"  Shape: {pooled_points.shape}")
    print(f"  Mean: {pooled_points.mean(dim=0)}")
    print(f"  Std: {pooled_points.std(dim=0)}")
    print(f"  Min: {pooled_points.min(dim=0).values}")
    print(f"  Max: {pooled_points.max(dim=0).values}")

    # Compare scene centers
    print_section("Scene Center Comparison")
    if 'scene_center' in bse_dict:
        print(f"BSE scene_center: {bse_dict['scene_center']}")
    if 'scene_center' in pooled_dict:
        print(f"Pooled scene_center: {pooled_dict['scene_center']}")

    # Compare poses if available
    print_section("Poses Comparison")
    if 'all_poses' in bse_dict:
        print(f"BSE all_poses shape: {bse_dict['all_poses'].shape}")
    if 'all_poses' in pooled_dict:
        print(f"Pooled all_poses shape: {pooled_dict['all_poses'].shape}")

    # Feature dimension analysis
    print_section("Feature Dimension Analysis")
    bse_dim = bse_dict['pooled_features'].shape[-1]
    pooled_dim = pooled_dict['pooled_features'].shape[-1]

    print(f"BSE feature dim: {bse_dim}")
    print(f"Pooled feature dim: {pooled_dim}")
    print(f"Dimension ratio: {bse_dim / pooled_dim:.2f}x")

    if bse_dim % 4 == 0:
        print(f"\nBSE features can be split into 4 layers: {bse_dim // 4} dim each")

    # Cosine similarity between random samples
    print_section("Feature Distribution Comparison")

    # Sample 1000 random features from each
    n_samples = min(1000, bse_dict['pooled_features'].shape[0],
                    pooled_dict['pooled_features'].shape[0])

    bse_sample = bse_dict['pooled_features'][:n_samples]
    pooled_sample = pooled_dict['pooled_features'][:n_samples]

    # Normalize and compute self-similarity
    bse_norm = torch.nn.functional.normalize(bse_sample, dim=-1)
    pooled_norm = torch.nn.functional.normalize(pooled_sample, dim=-1)

    bse_sim = (bse_norm @ bse_norm.T).mean()
    pooled_sim = (pooled_norm @ pooled_norm.T).mean()

    print(f"BSE self-similarity (cosine): {bse_sim:.4f}")
    print(f"Pooled self-similarity (cosine): {pooled_sim:.4f}")

    print_section("Summary")
    print(f"✓ BSE memory: {bse_dict['pooled_features'].shape[0]} points, {bse_dim}D features")
    print(f"✓ Pooled memory: {pooled_dict['pooled_features'].shape[0]} points, {pooled_dim}D features")
    print(f"✓ Feature dimension mismatch: {bse_dim}D vs {pooled_dim}D ({bse_dim/pooled_dim:.1f}x)")
    print(f"✓ Both memories loaded successfully through load_memory_features()")


if __name__ == '__main__':
    bse_path = '/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/04_evaluation/memory_extract/scene2a/40v_v0.05_bilinear_sor_l2/20260402_121851/memory_bse.pt'
    pooled_path = '/home/xwh/project/map-anything-experiments/Indoor6_scene2a_train_pooled_GT.pt'

    compare_memory_files(bse_path, pooled_path)
