#!/usr/bin/env python3
"""
Validate extracted memory file.

Usage:
    python validate_memory.py <memory_file.pt>
    python validate_memory.py <memory_file.pt> --save_ply
"""
import sys
import torch
import numpy as np
from pathlib import Path


def validate_memory(memory_path: str, save_ply: bool = False):
    """Validate memory file and optionally save as PLY for visualization."""

    print("=" * 60)
    print("Memory File Validation")
    print("=" * 60)
    print(f"\nLoading: {memory_path}")

    # Load memory
    memory = torch.load(memory_path, map_location='cpu')

    # Print schema
    print(f"\n[Schema] Version: {memory.get('schema_version', 'unknown')}")

    # Print all keys
    print(f"\n[Keys] Available keys:")
    for key in sorted(memory.keys()):
        val = memory[key]
        if isinstance(val, torch.Tensor):
            print(f"  - {key}: Tensor {list(val.shape)}, dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            print(f"  - {key}: {type(val).__name__} (len={len(val)})")
        elif isinstance(val, np.ndarray):
            print(f"  - {key}: ndarray {list(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  - {key}: {type(val).__name__} = {val}")

    # Validate points
    points = memory.get('points')
    if points is None:
        print("\n[ERROR] No 'points' key found!")
        return False

    print(f"\n[Points]")
    print(f"  Count: {points.shape[0]:,}")
    print(f"  Shape: {points.shape}")
    print(f"  Range X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Range Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Range Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # Check for NaN/Inf
    if torch.isnan(points).any():
        print("  [WARNING] Points contain NaN values!")
    if torch.isinf(points).any():
        print("  [WARNING] Points contain Inf values!")

    # Validate features
    features = memory.get('features')
    if features is not None:
        print(f"\n[Features]")
        print(f"  Shape: {features.shape}")
        print(f"  Dim: {features.shape[1]}")
        print(f"  Range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Norm: {features.norm(dim=1).mean():.3f} (mean L2 norm)")

        if torch.isnan(features).any():
            print("  [WARNING] Features contain NaN values!")
        if torch.isinf(features).any():
            print("  [WARNING] Features contain Inf values!")

    # Validate colors
    colors = memory.get('colors')
    if colors is not None:
        print(f"\n[Colors]")
        print(f"  Shape: {colors.shape}")
        print(f"  Range: [{colors.min():.3f}, {colors.max():.3f}]")

    # Validate scene normalization
    scene_mean = memory.get('mu', memory.get('scene_mean'))
    scene_sigma = memory.get('sigma', memory.get('scene_sigma'))
    if scene_mean is not None and scene_sigma is not None:
        print(f"\n[Scene Normalization]")
        if isinstance(scene_mean, torch.Tensor):
            print(f"  Mean: {scene_mean.tolist()}")
        else:
            print(f"  Mean: {scene_mean}")
        if isinstance(scene_sigma, torch.Tensor):
            scene_sigma = float(scene_sigma.item())
        print(f"  Sigma: {float(scene_sigma):.4f}")

    # Validate view info
    camera_centers = memory.get('view_camera_centers', memory.get('camera_centers'))
    if camera_centers is not None:
        print(f"\n[Camera Centers]")
        print(f"  Count: {camera_centers.shape[0]}")
        print(f"  Range X: [{camera_centers[:, 0].min():.3f}, {camera_centers[:, 0].max():.3f}]")
        print(f"  Range Y: [{camera_centers[:, 1].min():.3f}, {camera_centers[:, 1].max():.3f}]")
        print(f"  Range Z: [{camera_centers[:, 2].min():.3f}, {camera_centers[:, 2].max():.3f}]")

    all_scale_tokens = memory.get('all_scale_tokens')
    if all_scale_tokens is not None:
        print(f"\n[Scale Tokens]")
        print(f"  Shape: {all_scale_tokens.shape}")

    # Validate ray directions
    ray_dirs = memory.get('ray_dirs')
    if ray_dirs is not None:
        print(f"\n[Ray Directions]")
        print(f"  Shape: {ray_dirs.shape}")
        norms = ray_dirs.norm(dim=1)
        print(f"  Norm range: [{norms.min():.4f}, {norms.max():.4f}] (should be ~1.0)")

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    issues = []

    if points is None:
        issues.append("Missing 'points' key")
    elif points.shape[0] == 0:
        issues.append("Empty point cloud (0 points)")

    if features is None:
        issues.append("Missing 'features' key")

    if torch.isnan(points).any() if points is not None else False:
        issues.append("Points contain NaN")

    if features is not None and torch.isnan(features).any():
        issues.append("Features contain NaN")

    if len(issues) == 0:
        print("[OK] Memory file is valid!")
        status = True
    else:
        print("[ISSUES FOUND]")
        for issue in issues:
            print(f"  - {issue}")
        status = False

    # Save PLY for visualization
    if save_ply and points is not None:
        ply_path = Path(memory_path).with_suffix('.ply')
        save_point_cloud_ply(points, colors, ply_path)
        print(f"\n[PLY Saved] {ply_path}")

    return status


def save_point_cloud_ply(points: torch.Tensor, colors: torch.Tensor, output_path: Path):
    """Save point cloud as PLY file for visualization."""

    points_np = points.detach().cpu().numpy().astype(np.float64)

    if colors is not None:
        colors_np = colors.detach().cpu().numpy()
        # Normalize colors to [0, 1] if needed
        if colors_np.max() > 1.1:
            colors_np = colors_np / 255.0
        colors_np = np.clip(colors_np, 0, 1).astype(np.float64)
    else:
        colors_np = np.ones_like(points_np) * 0.5

    # Write PLY
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Vertices
        for i in range(len(points_np)):
            x, y, z = points_np[i]
            r, g, b = (colors_np[i] * 255).astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_memory.py <memory_file.pt> [--save_ply]")
        print("\nExample:")
        print("  python validate_memory.py memory_bse.pt --save_ply")
        sys.exit(1)

    memory_path = sys.argv[1]
    save_ply = '--save_ply' in sys.argv

    if not Path(memory_path).exists():
        print(f"Error: File not found: {memory_path}")
        sys.exit(1)

    success = validate_memory(memory_path, save_ply)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
