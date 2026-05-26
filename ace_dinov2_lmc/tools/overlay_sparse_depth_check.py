from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    rgb_dir = Path("/data/xwh/RIO10_ace/scene01_seq01_01/train/rgb")
    depth_dir = Path("/data/xwh/RIO10_sparse_depth/scene01_seq01_seq01_01/sparse_depth")
    out_dir = Path(
        "/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/train_compare/"
        "rio10_conservative_indoor6_migration/sparse_depth_overlay_check_large"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = ["000000", "000999", "001999", "002999", "004014"]
    frames = []
    for idx in requested:
        rgb = rgb_dir / f"frame-{idx}.jpg"
        dep = depth_dir / f"frame-{idx}.stable.depth.png"
        if rgb.exists() and dep.exists():
            frames.append((idx, rgb, dep))
    if not frames:
        raise SystemExit("No matching RGB/depth frames found")

    point_radius = 5
    summary = []
    for idx, rgb_path, depth_path in frames:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Failed to read {depth_path}")
        depth = depth_raw.astype(np.float32) / 1000.0
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = np.isfinite(depth) & (depth > 0.1) & (depth < 10.0)
        valid = int(mask.sum())
        overlay = rgb.copy()
        depth_vis = np.zeros_like(rgb)
        if valid:
            vals = depth[mask]
            norm = np.clip((depth - 0.1) / 9.9, 0.0, 1.0)
            cmap_u8 = (255.0 * (1.0 - norm)).astype(np.uint8)
            color_bgr = cv2.applyColorMap(cmap_u8, cv2.COLORMAP_TURBO)
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            color_rgb[~mask] = 0
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (point_radius * 2 + 1, point_radius * 2 + 1),
            )
            dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            color_vis = cv2.dilate(color_rgb, kernel, iterations=1)
            overlay[dil] = (
                0.15 * overlay[dil].astype(np.float32) + 0.85 * color_vis[dil].astype(np.float32)
            ).astype(np.uint8)
            depth_vis[dil] = color_vis[dil]
            dmin, dmed, dmax = float(vals.min()), float(np.median(vals)), float(vals.max())
        else:
            dmin = dmed = dmax = 0.0

        h, _ = rgb.shape[:2]
        gap = np.full((h, 8, 3), 255, dtype=np.uint8)
        bar = make_colorbar(h)
        triptych = np.concatenate([rgb, gap, depth_vis, gap, overlay, gap, bar], axis=1)
        title = (
            f"frame-{idx} radius={point_radius}px valid={valid} "
            f"depth_m min/med/max={dmin:.2f}/{dmed:.2f}/{dmax:.2f}"
        )
        cv2.putText(triptych, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(triptych, title, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

        out_triptych = out_dir / f"frame-{idx}_rgb_depth_overlay_triptych_large.jpg"
        out_overlay = out_dir / f"frame-{idx}_overlay_large.jpg"
        cv2.imwrite(str(out_triptych), cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(out_overlay), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        summary.append((idx, valid, dmin, dmed, dmax, str(out_triptych), str(out_overlay)))

    summary_path = out_dir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("frame\tvalid_pixels\tdepth_min_m\tdepth_median_m\tdepth_max_m\ttriptych\toverlay\n")
        for row in summary:
            f.write(
                "\t".join([row[0], str(row[1]), f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", row[5], row[6]])
                + "\n"
            )
    print(summary_path)
    for row in summary:
        print(f"frame-{row[0]} valid={row[1]} depth={row[2]:.2f}/{row[3]:.2f}/{row[4]:.2f} triptych={row[5]}")


def make_colorbar(height: int) -> np.ndarray:
    bar_w = 86
    bar = np.full((height, bar_w, 3), 255, dtype=np.uint8)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    bar_vals = (255.0 * (1.0 - y)).astype(np.uint8)[:, None]
    bar_color_bgr = cv2.applyColorMap(np.repeat(bar_vals, 20, axis=1), cv2.COLORMAP_TURBO)
    bar_color = cv2.cvtColor(bar_color_bgr, cv2.COLOR_BGR2RGB)
    x0 = 8
    bar[:, x0 : x0 + 20] = bar_color
    cv2.rectangle(bar, (x0, 0), (x0 + 20, height - 1), (0, 0, 0), 1)
    cv2.putText(bar, "depth", (34, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    for depth_tick in [0.1, 2.0, 4.0, 6.0, 8.0, 10.0]:
        yy = int(round((depth_tick - 0.1) / 9.9 * (height - 1)))
        yy = max(0, min(height - 1, yy))
        cv2.line(bar, (x0 + 20, yy), (x0 + 30, yy), (0, 0, 0), 1)
        text_y = max(13, min(height - 4, yy + 5))
        cv2.putText(
            bar,
            f"{depth_tick:g}m",
            (x0 + 33, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return bar


if __name__ == "__main__":
    main()
