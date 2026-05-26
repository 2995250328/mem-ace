# Step 18: RIO10 WAI Training Resize Fix Log

## 1. Failure

During RIO10 WAI ACE-G training, S1 buffer creation failed in the DINOv2 encoder:

```text
AssertionError: Input size (920, 518) must be multiple of patch_size (14)
```

The failing path was:

```text
TrainerACEDINOv2LMC._train_ace_g()
-> _prepare_s1_buffer_for_iteration()
-> create_training_buffer_ace_g()
-> _create_training_buffer_with_scene_coords()
-> regressor.get_features()
-> DINOv2 encoder patch-size assertion
```

## 2. Root Cause

`CamLocDatasetWAIDINOv2` used `TF.resize(image, image_height)` on a PIL image.
For PIL images, an integer resize target means "resize the shorter edge", not
"resize to this exact height".

RIO10 WAI frames can have portrait-like dimensions after loading/normalization.
With `image_height=518`, this produced tensors shaped:

```text
[3, 920, 518]
```

instead of a DINOv2-compatible height-aligned tensor.

## 3. Fix

`/home/xwh/project/ace_depth/dataset_wai_dinov2.py` now resizes WAI images with
an explicit `(target_height, target_width)` tuple:

```text
target_height = round_to_patch_size(image_height)
target_width = round_to_patch_size(orig_width * target_height / orig_height)
```

This preserves aspect ratio while making both dimensions divisible by the DINOv2
patch size.

## 4. Validation

Smoke check on:

```text
/data/xwh/RIO10_wai/mapanything_wai/scene01_seq01_01_train
```

Result:

```text
base image:     torch.Size([3, 518, 294])
attached image: torch.Size([3, 518, 294])
sparse coords:  torch.Size([3, 37, 21])
H % 14 = 0, W % 14 = 0
```

The RIO10 training retry should use one GPU. The main Step15 C1 fusion-cascade
architecture work should use the remaining three GPUs for controlled ablations.

## 5. Chain Correction: Training Must Use ACE Backend

The indoor6 training contract uses ACE-format RGB / poses / calibration data:

```text
--data_backend ace
<scene>/train/{rgb,poses,calibration}
<scene>/test/{rgb,poses,calibration}
```

MapAnything/WAI is used for memory extraction because that stage needs the
MapAnything loader and forward pass. The ACE-G training stage should still use
ACE-format data when we want parity with indoor6.

A RIO10 ACE-format scene has therefore been generated at:

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

Counts:

```text
train frames = 4380
train poses = 4380
train calibration = 4380
test frames = 2069
```

The converter used is:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/tools/convert_rio10_wai_to_ace.py
```

Images are symlinked from the WAI scene, while poses and 3x3 calibration files
are written from `scene_meta.json`.

## 6. Sparse-Depth Sampling Check

For ACE backend, sparse depth is resolved from:

```text
--c1_aux_depth_root /data/xwh/RIO10_sparse_depth
--c1_aux_depth_kind sparse_depth
```

to:

```text
/data/xwh/RIO10_sparse_depth/scene01_seq01_seq01_01/sparse_depth
```

The ACE fallback matcher now supports sparse-depth PNG names such as:

```text
frame-000000.stable.depth.png
```

Smoke result through `TrainerACEDINOv2LMC._build_train_dataset()` with
`data_backend=ace`:

```text
matched sparse depth = 4356 / 4380
missing sparse depth = 24
image shape = (3, 518, 294)
coords shape = (3, 37, 21)
valid sparse coord patches in frame-000000 = 51
```

The 24 missing sparse-depth frames are handled as empty coordinate masks. Valid
coordinate sampling remains active for frames with sparse depth.

