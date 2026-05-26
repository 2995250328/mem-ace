---
name: wayspots-sparse-depth-workflow
description: Use when generating, filtering, exporting, and visualizing Wayspots sparse depth (SuperPoint/COLMAP known-pose triangulation) for ACE/WAI pipelines.
---

# Wayspots Sparse Depth Workflow

Use this skill for `tools/wayspots_known_pose_sparse_depth.py` based sparse-depth generation on Wayspots scenes.

## Scope

- SuperPoint-based sparse depth generation (`train` / `test`)
- strict density control (feature + geometry thresholds)
- projection from an existing COLMAP model using GT ACE poses
- point cloud export to `.ply` for quick inspection
- RGB + sparse depth colored-point overlay with colorbar

## Preconditions

1. Dataset exists under `/data/xwh/Wayspots/<scene_name>/`.
2. SuperPoint weights exist:
   `/home/xwh/project/ace_depth/ace_dinov2_lmc/superpoint_v1.pth`
3. SuperPoint repo exists and is injected into `PYTHONPATH`:
   `/data/xwh/SuperPointPretrainedNetwork`
4. Run in `mapanything` env (contains `torch` + `pycolmap`).

## Canonical Commands

### 1) SuperPoint sparse depth (strict profile)

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc

PYTHONPATH=/data/xwh/SuperPointPretrainedNetwork:$PYTHONPATH \
conda run -n mapanything python tools/wayspots_known_pose_sparse_depth.py \
  /data/xwh/Wayspots/wayspots_bears \
  --split train \
  --workspace /data/xwh/dataset_staging/wayspots/workspaces/wayspots_bears_train_superpoint_w20_strict \
  --output-subdir sparse_depth_superpoint_strict \
  --feature-backend superpoint \
  --superpoint-weights /home/xwh/project/ace_depth/ace_dinov2_lmc/superpoint_v1.pth \
  --superpoint-conf-thresh 0.05 \
  --superpoint-max-keypoints 1024 \
  --superpoint-nms-dist 8 \
  --superpoint-nn-thresh 0.60 \
  --matcher pairs \
  --match-window 20 \
  --num-threads 4 \
  --min-triangulation-angle 3.0 \
  --max-depth-m 1000 \
  --overwrite
```

Repeat for `--split test` with a dedicated workspace path.

### 2) Coverage check

```bash
python - <<'PY'
import glob, numpy as np
for split in ["train", "test"]:
    files = sorted(glob.glob(f"/data/xwh/Wayspots/wayspots_bears/{split}/sparse_depth_superpoint_strict/*.npz"))
    non_empty = 0
    for f in files:
        d = np.load(f)["arr_0"]
        if np.isfinite(d).any() and (d > 0).any():
            non_empty += 1
    total = len(files)
    ratio = (non_empty / total * 100) if total else 0.0
    print(split, "total", total, "non_empty", non_empty, "ratio", f"{ratio:.2f}%")
PY
```

### 3) Export triangulated point cloud to PLY

```bash
conda run -n mapanything python - <<'PY'
import pycolmap, numpy as np
from pathlib import Path

model_dir = Path("/data/xwh/dataset_staging/wayspots/workspaces/wayspots_bears_train_superpoint_w20_strict/triangulated_model")
out_ply = model_dir.parent / "triangulated_points3D.ply"

rec = pycolmap.Reconstruction(str(model_dir))
pts, cols = [], []
for p in rec.points3D.values():
    xyz = np.asarray(p.xyz, dtype=np.float64)
    if np.all(np.isfinite(xyz)):
        pts.append(xyz)
        c = np.asarray(p.color, dtype=np.uint8) if hasattr(p, "color") else np.array([255, 255, 255], dtype=np.uint8)
        cols.append(c)

pts = np.asarray(pts, dtype=np.float32)
cols = np.asarray(cols, dtype=np.uint8)
with open(out_ply, "w", encoding="utf-8") as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {len(pts)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(pts, cols):
        f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
print(out_ply)
print("num_points:", len(pts))
PY
```

### 4) Visualize sparse depth overlay

Use helper script:
`tools/debug_overlay_sparse_depth.py`

It writes:
`04_evaluation/wayspots_debug/bears_train_frame_00000_sparse_depth_overlay.png`

### 5) Recommended filtered SIFT reconstruction projection

For a more complete scene point cloud, prefer SIFT known-pose reconstruction as the source, then filter points and apply depth-map NMS during GT projection:

```bash
cd /home/xwh/project/ace_depth/ace_dinov2_lmc

conda run --no-capture-output -n mapanything python tools/project_colmap_sparse_depth.py \
  /data/xwh/Wayspots/wayspots_bears \
  --split train \
  --model-dir /data/xwh/dataset_staging/wayspots/workspaces/wayspots_bears_train_sift_pairs_w20/triangulated_model \
  --output-subdir sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4 \
  --min-track-length 4 \
  --max-reproj-error 2.0 \
  --min-depth-m 0.2 \
  --max-depth-m 100 \
  --depth-nms-radius 4 \
  --depth-nms-rank uniform \
  --visibility-mode observed \
  --export-filtered-ply /data/xwh/dataset_staging/wayspots/workspaces/wayspots_bears_train_sift_pairs_w20/triangulated_points3D_filtered_t4_e2.ply
```

Observed bears/train reference output:

- filtered points: `239202`
- non-empty frames: `581/581`
- average projected points/frame after observed-track NMS: about `3388`
- output: `/data/xwh/Wayspots/wayspots_bears/train/sparse_depth_sift_filt_t4_e2_d02_observed_nms_r4`

## Quality Tuning

- Fewer points: raise `--superpoint-conf-thresh`, raise `--superpoint-nms-dist`, lower `--superpoint-max-keypoints`.
- Cleaner geometry: raise `--min-triangulation-angle` (e.g., `3.0 -> 4.0`).
- For training supervision, prefer filtering point tracks and reprojection error before projection, then use depth-map NMS.
- Use `--visibility-mode observed` so each image only receives 3D points actually observed in its COLMAP tracks; avoid projecting the whole scene cloud into every frame.
- Avoid using heavily strict SuperPoint reconstruction as the only full-scene cloud; it is often too sparse for Wayspots.
- Safer default order:
  1. tune feature density
  2. tune triangulation angle
  3. inspect overlay and PLY
  4. then batch to all scenes

## Output Contract

- Depth files: `<scene>/<split>/<output-subdir>/*.npz` (key: `arr_0`, meters)
- Workspace model:
  - `triangulated_model/points3D.bin`
  - `triangulated_model/images.bin`
- Inspection artifact:
  - `triangulated_points3D.ply`
  - overlay PNG under `04_evaluation/wayspots_debug/`
