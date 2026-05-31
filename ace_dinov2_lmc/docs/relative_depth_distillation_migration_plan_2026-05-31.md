# Relative Depth Distillation Migration Plan

Date: 2026-05-31

## Goal

Migrate the older ACE depth-distillation idea into the current `ace_dinov2_lmc`
training path, primarily for ACE-FCN-LMC Stage2 / GLACE concat experiments.

The intended signal is:

1. Use a monocular depth teacher to estimate image-space depth.
2. Project the current scene-coordinate prediction into the camera frame.
3. Use the predicted camera-space `z` as student pseudo depth.
4. Apply a relative-depth loss between teacher depth and student projected depth.

## Existing Code

The old implementation lives in the parent ACE project, not in the current
LMC trainer:

- `/home/xwh/project/ace_depth/ace_trainer_full.py`
- `/home/xwh/project/ace_depth/ace_trainer_depth.py`
- `/home/xwh/project/ace_depth/ace_network_full.py`
- `/home/xwh/project/ace_depth/ace_network_depth.py`
- `/home/xwh/project/ace_depth/train_ace_full.py`
- `/home/xwh/project/ace_depth/train_ace_depth.py`

Existing data flow:

- `DepthAnythingV2` is loaded as a frozen teacher.
- RGB is converted to uint8 / BGR and passed to `infer_image`.
- ACE predicts scene coordinates.
- Predicted scene coordinates are transformed by `gt_pose_inv`.
- Student depth is `cam_coords[:, 2]`.
- Teacher and student depths are normalized.
- `RelativeDepthLoss` combines scale-shift invariant loss, gradient loss, and
  sampled pairwise relative loss.

## Important Bugs In The Old Version

The old code should not be migrated verbatim.

1. Student depth is computed inside `torch.no_grad()`.

   This means the depth loss does not backpropagate into the ACE head in the
   old implementation. Keep teacher inference under `no_grad`, but keep student
   scene-coordinate prediction and camera-z projection grad-enabled.

2. Batched gradient loss is not dimension-safe.

   The old `compute_gradient()` assumes 2D `(H, W)` depth maps. When passed
   `(B, H, W)`, it can treat the batch dimension as a spatial dimension.

3. Scale/shift alignment is global over the whole batch.

   It should be per image. Otherwise unrelated frames can contaminate each
   other's relative scale.

4. Existing naming is ambiguous.

   `RelativeDepthLoss_weight` is an internal pairwise-vs-SSI weight, while
   `DepthLoss_weight` is the task-level weight. The new flags should make this
   distinction explicit.

## Teacher Model Recommendation

Default first implementation:

- `Depth Anything V2 Base` or `Small` as an online teacher.
- Use relative / scale-shift invariant supervision only.

Reason:

- The project already vendors `depth_anything_v2`.
- The old ACE depth code already knows how to call it.
- It is practical inside a training loop on the available GPUs.
- Its output semantics match relative depth supervision.

Second-stage teacher experiment:

- `Depth Pro` as offline precomputed teacher depth.

Reason:

- It predicts metric depth and sharp boundaries.
- It should be cached to disk, not run inside every training iteration.

Lower-priority offline candidates:

- `Metric3D v2`
- `UniDepthV2`
- `MoGe`
- `VGGT`
- `Marigold`

These are useful as later teacher-depth ablations or geometry sanity checks, but
they add more dependency and runtime risk than Depth Anything V2.

## Proposed New Module

Add a small helper module instead of reusing the old trainers:

- `ace_dinov2_lmc/relative_depth_distillation.py`

Responsibilities:

- Load online Depth Anything V2 teacher, if requested.
- Load precomputed teacher depth maps, if requested later.
- Convert RGB tensors to teacher input.
- Resize teacher depth to prediction resolution.
- Project predicted scene coordinates to camera-space depth.
- Compute per-image relative-depth loss.

## Proposed Loss

Use per-image loss over `(B, H, W)`:

- scale-shift invariant L1 or Huber loss
- valid-neighbor gradient matching
- sampled pairwise relative depth loss

Suggested defaults:

- task weight: `0.05`
- pairwise weight: `0.5`
- gradient weight: `0.5`
- max pair samples per image: `2048`
- start ratio: `0.3`

## Proposed CLI

Add flags to `options_dinov2_lmc.py`:

```bash
--use_relative_depth_loss True
--relative_depth_teacher depth_anything_v2_online
--relative_depth_teacher_encoder vitb
--relative_depth_teacher_checkpoint /path/to/depth_anything_v2_vitb.pth
--relative_depth_loss_weight 0.05
--relative_depth_start_ratio 0.3
--relative_depth_pair_weight 0.5
--relative_depth_grad_weight 0.5
--relative_depth_max_samples 2048
--relative_depth_apply_to stage2
```

Later offline mode:

```bash
--relative_depth_teacher precomputed_depth
--relative_depth_teacher_root /path/to/depthpro_cache
--relative_depth_teacher_kind depthpro
```

## Migration Targets

First target:

- ACE-FCN-LMC Stage2 with `ace_lmc_global_head_mode=glace_concat`

Reason:

- Stage1 local ACE-FCN memory already works.
- Stage2 is where additional global / dense regularization may help.
- The integration surface is smaller if local stack is frozen.

Later targets:

- Stage1 local LMC
- ACE-G S2 full-map path
- baseline ACE-FCN ablations

## Evaluation Plan

Start with one easy/stable Wayspots scene:

- `wayspots_bears`
- `wayspots_therock`

Compare:

- Stage2 no depth loss
- Stage2 + Depth Anything V2 online relative depth
- Stage2 + precomputed Depth Pro teacher, later

Report:

- 25cm/5deg
- 10cm/5deg
- 5cm/5deg
- 2cm/2deg
- 1cm/1deg
- nan/inf count during training
- depth loss value
- valid depth pixel ratio

## Implementation Notes

- Do not hardcode `/mnt/storage/xwh/checkpoints/...`; make checkpoint path a CLI
  flag.
- Teacher inference can be cached per batch only within the current step, but
  full persistent caching should be a separate offline teacher mode.
- If `use_half=True`, keep camera projection and loss math in float32.
- Clamp student camera-z to a sane range before normalization, but do not let
  clamping hide non-finite predictions.
- If non-finite student coordinates appear, skip only that depth-loss term and
  log the count.

