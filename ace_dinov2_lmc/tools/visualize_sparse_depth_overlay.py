#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_depth(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path)
    if "arr_0" in z.files:
        return z["arr_0"]
    for k in ("depth", "sparse_depth", "depth_map", "z", "depth_mm"):
        if k in z.files:
            return z[k]
    for k in z.files:
        v = z[k]
        if getattr(v, "ndim", 0) == 2 and np.issubdtype(v.dtype, np.floating):
            return v
    raise KeyError(f"No depth-like key in {npz_path}, keys={z.files}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay sparse depth points on RGB with colorbar.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--point-size", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--cmap", type=str, default="turbo")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    depth = load_depth(args.depth)
    ys, xs = np.where(depth > 0)
    vals = depth[ys, xs]
    if vals.size == 0:
        raise RuntimeError(f"No valid depth points in {args.depth}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    ax.imshow(image)
    sc = ax.scatter(xs, ys, c=vals, s=args.point_size, cmap=args.cmap, alpha=args.alpha, edgecolors="none")
    ax.set_title(f"{args.image.name} + {args.depth.name} (N={vals.size})")
    ax.axis("off")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Depth (meters)")
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(args.out)
    print(f"points={vals.size} depth_min={float(vals.min()):.4f} depth_max={float(vals.max()):.4f}")


if __name__ == "__main__":
    main()
