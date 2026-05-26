#!/usr/bin/env python3
"""Export COLMAP triangulated_model points3D to ASCII PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap


def export_ply(model_dir: Path, out_ply: Path | None = None) -> tuple[Path, int]:
    model_dir = model_dir.resolve()
    if out_ply is None:
        out_ply = model_dir.parent / "triangulated_points3D.ply"
    out_ply = out_ply.resolve()
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(model_dir))
    pts, cols = [], []
    for p in rec.points3D.values():
        xyz = np.asarray(p.xyz, dtype=np.float64)
        if np.all(np.isfinite(xyz)):
            pts.append(xyz)
            c = (
                np.asarray(p.color, dtype=np.uint8)
                if hasattr(p, "color")
                else np.array([255, 255, 255], dtype=np.uint8)
            )
            cols.append(c)

    pts_arr = np.asarray(pts, dtype=np.float32)
    cols_arr = np.asarray(cols, dtype=np.uint8)

    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts_arr)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts_arr, cols_arr):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

    return out_ply, len(pts_arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to COLMAP triangulated_model directory",
    )
    parser.add_argument(
        "--out-ply",
        type=Path,
        default=None,
        help="Output PLY path (default: <workspace>/triangulated_points3D.ply)",
    )
    args = parser.parse_args()
    out_ply, n = export_ply(args.model_dir, args.out_ply)
    print(out_ply)
    print("num_points:", n)


if __name__ == "__main__":
    main()
