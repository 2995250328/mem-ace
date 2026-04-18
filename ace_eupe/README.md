# ACE + EUPE Backbone

This subdirectory adds isolated EUPE backbone variants without changing the
existing ACE, ACE-DINOv2, or LMC code paths.

The implementation supports two separate EUPE replacements:

- EUPE ViT-B/16: `eupe_vitb16`, feature dim 768, stride 16
- EUPE ConvNeXt-B: `eupe_convnext_base`, feature dim 1024, stride 32
- ImageNet RGB normalization, matching the EUPE examples
- ACE's existing scene-coordinate head and buffer-based head training

## Train ViT

```bash
cd /home/xwh/project/ace_depth
python ace_eupe/train_ace_eupe.py \
  /mnt/storage/xwh/7Scenes/7scenes_source/office \
  best.pt \
  --eupe_root /home/xwh/project/EUPE \
  --eupe_model_name eupe_vitb16 \
  --eupe_checkpoint /home/xwh/project/EUPE/checkpoints/EUPE-ViT-B.pt \
  --image_resolution 448 \
  --device cuda:0
```

## Train ConvNeXt

```bash
cd /home/xwh/project/ace_depth
python ace_eupe/train_ace_eupe.py \
  /mnt/storage/xwh/7Scenes/7scenes_source/office \
  best.pt \
  --eupe_root /home/xwh/project/EUPE \
  --eupe_model_name eupe_convnext_base \
  --eupe_checkpoint /home/xwh/project/EUPE/checkpoints/EUPE-ConvNeXt-B.pt \
  --image_resolution 448 \
  --device cuda:0
```

The output map is written under:

```text
<experiment_root>/<dataset>/<scene>/<scene>_<eupe_model_name>_ep<epochs>_bs<batch_size>_<suffix>.pt
```

## Test

```bash
python ace_eupe/test_ace_eupe.py \
  /mnt/storage/xwh/7Scenes/7scenes_source/office \
  /path/to/office_eupe_ep16_bs5120_best.pt \
  --eupe_root /home/xwh/project/EUPE \
  --eupe_model_name eupe_vitb16 \
  --eupe_checkpoint /home/xwh/project/EUPE/checkpoints/EUPE-ViT-B.pt \
  --image_resolution 448 \
  --device cuda:0
```

## Notes

The trainer follows the existing DINOv2 variant's head-only training flow: the
EUPE backbone is used to fill the feature buffer, and the ACE head is trained
from sampled features.
