# Handoff Prompt: Dataset Choice for ACE-DINOv2 / ACE-G

You are continuing a discussion in `/home/xwh/project/ace_depth/ace_dinov2_lmc`.

The user is looking for an Indoor6-like dataset where simply replacing the original ACE backbone with DINO/DINOv2 is likely to improve performance. RIO10 is considered too pathological for the current purpose because long-term structure changes, map aging, object movement, sparse-depth instability, and memory coverage issues make it hard to isolate backbone quality.

## Goal

Evaluate two candidate datasets for ACE-DINOv2 / ACE-G adaptation:

1. NAVER LABS Indoor Localization Dataset
2. InLoc

Selection criteria:

```text
official or standard localization dataset
indoor / large indoor space
small-to-moderate number of scenes or floors
posed RGB images
camera intrinsics and extrinsics
depth, LiDAR point cloud, SfM, or COLMAP-like geometry usable for sparse depth / scene-coordinate supervision
not dominated by pathological long-term structural change
likely to benefit from DINO/DINOv2 patch features as ACE backbone
reasonable path to ACE-format conversion
```

## Short conclusion

```text
NAVER LABS Indoor: first recommendation.
InLoc: suitable, but second-stage / challenge recommendation.
```

## 1. NAVER LABS Indoor Localization Dataset

Recommended priority: highest.

Why it is promising:

```text
official indoor localization dataset
large real indoor spaces: department stores / shopping mall / metro station style environments
about 5 indoor datasets / areas
posed RGB images
camera intrinsics and extrinsics through kapture-style data organization
LiDAR point clouds / sparse LiDAR depth / SfM-refined poses are available or referenced
large enough to show DINOv2 advantage, but less pathological than RIO10
more similar to Indoor6 than RIO10 in terms of purpose: indoor localization with usable geometry
```

Why it fits DINO/DINOv2 backbone:

```text
large indoor scenes
repeated structures
weak texture
crowds / occlusion
viewpoint and illumination variation
high-quality RGB where patch-level semantic/geometric DINO features may help
```

Why it is better than RIO10 for the current question:

```text
RIO10 tests long-term map robustness and structural change.
NAVER LABS Indoor is more suitable for testing whether DINO as a stronger ACE backbone improves localization under realistic but not pathological indoor conditions.
```

Expected adaptation route:

```text
NAVER/kapture format
-> parse sensors / intrinsics
-> parse trajectories / camera poses
-> parse image records
-> optionally parse point cloud / sparse LiDAR / SfM points
-> convert to ACE-format:
   scene/train/rgb
   scene/train/poses
   scene/train/calibration
   scene/test/rgb
   scene/test/poses
   scene/test/calibration
-> first run vanilla ACE-DINOv2, not ACE-G
-> compare against ACE-FCN or existing ACE-style baseline
-> only then consider ACE-G / LMC / memory extraction
```

Main risks:

```text
data size is large
kapture-to-ACE converter needs to be written carefully
scene/floor split must be chosen conservatively
large scene may stress a single ACE head, so start with one floor/area pilot
```

Practical recommendation:

```text
Use NAVER LABS Indoor as the first new dataset to adapt.
Start from a small pilot floor/area.
Initial target is vanilla DINO ACE vs original ACE backbone, not LMC.
```

## 2. InLoc

Recommended priority: second.

Why it is promising:

```text
official indoor visual localization benchmark
large indoor building areas
RGB query images
RGB-D database / reference images
query 6DoF poses are available for evaluation
intrinsics and registered geometry exist
suitable for weak texture, repeated structures, and cross-view localization
```

Why it fits DINO/DINOv2 backbone:

```text
query/database appearance gap
weak texture
repeated indoor structures
large viewpoint differences
smartphone RGB query vs RGB-D map/database
DINOv2 patch features may be more robust than original ACE encoder features
```

Why it is not first priority:

```text
not naturally ACE-format
original protocol is retrieval + dense matching + view synthesis / pose verification
training/test construction for scene coordinate regression is less direct
needs careful handling of database RGB-D images, query images, poses, depth, and map coordinate frames
larger risk of mixing backbone quality with large-scene/memory-coverage issues
```

Expected adaptation route:

```text
select one building / floor subset
use database RGB-D perspective images as ACE train images
use query smartphone RGB photos as ACE test images
use database depth + pose to generate scene-coordinate supervision or sparse depth
convert to ACE-format
first run vanilla ACE-DINOv2
only later consider ACE-G / LMC
```

Main risks:

```text
protocol mismatch with ACE per-scene training
coordinate frame handling must be audited
query GT and eval conventions must be checked
large indoor areas may require partitioning or memory-aware training
engineering cost higher than NAVER LABS Indoor
```

Practical recommendation:

```text
Use InLoc after NAVER LABS Indoor is working.
Treat it as an indoor challenge benchmark, not the first backbone-validation dataset.
```

## Final recommendation

```text
First adapt: NAVER LABS Indoor Localization Dataset.
Second adapt: InLoc.

Do not start with RIO10 for this backbone question.
Do not start with ACE-G/LMC on the new dataset.
Start with vanilla ACE-DINOv2 vs original ACE-style baseline to isolate the effect of the DINO backbone.
```

## Suggested next actions

```text
1. Search/download NAVER LABS Indoor documentation and inspect exact kapture layout.
2. Identify one small floor/area for a pilot.
3. Write a NAVER-kapture-to-ACE converter.
4. Verify converted ACE-format with counts, stem alignment, intrinsics scaling, pose convention, and camera-center visualization.
5. Train vanilla ACE-DINOv2.
6. Only if vanilla DINO is healthy, add sparse depth / ACE-G memory later.
```
