#!/usr/bin/env python3
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main() -> None:
    img_path = Path("/data/xwh/Wayspots/wayspots_bears/train/rgb/frame_00000.jpg")
    depth_path = Path("/data/xwh/Wayspots/wayspots_bears/train/sparse_depth_superpoint/frame_00000.npz")
    out_path = Path(
        "/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_debug/"
        "bears_train_frame_00000_sparse_depth_overlay.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(img_path).convert("RGB"))
    depth = np.load(depth_path)["arr_0"]
    ys, xs = np.where(depth > 0)
    vals = depth[ys, xs]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=140)
    ax.imshow(image)
    sc = ax.scatter(xs, ys, c=vals, s=14, cmap="turbo", alpha=0.9, edgecolors="none")
    ax.set_title("wayspots_bears train frame_00000 sparse depth overlay (SuperPoint)")
    ax.axis("off")
    colorbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Depth (meters)")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(out_path)
    print(f"points={len(vals)} min={float(vals.min()):.4f} max={float(vals.max()):.4f}")


if __name__ == "__main__":
    main()
