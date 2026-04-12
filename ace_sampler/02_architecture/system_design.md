# System Architecture Design

## 1. Directory Structure

The implementation lives directly in `ace_sampler/` (not in a `03_implementation/` subfolder,
since the module was created before the research workflow was adopted).

```
ace_sampler/                    # Python module + idea folder
├── __init__.py                 # Exports: SamplerNet, SamplerTrainer, fill_buffer_with_sampler
├── model.py                    # NEW: SamplerNet (~80K params)
├── uncertainty.py              # NEW: UncertaintyHead with MC Dropout
├── trainer.py                  # NEW: Phase 1 SamplerTrainer
├── buffer_sampler.py           # NEW: Phase 2 fill_buffer_with_sampler
├── options.py                  # NEW: add_sampler_train_args, add_sampler_buffer_args
train_ace_sampler.py            # NEW: Phase 1 entry script (project root)
```

Reused without modification:
```
ace_network.py                  # REUSE: Regressor (frozen in Phase 1)
dataset_origin.py               # REUSE: CamLocDataset
ace_util.py                     # REUSE: get_pixel_grid
ace_trainer.py                  # REUSE: Phase 2 calls fill_buffer_with_sampler as drop-in
```

## 2. Module Responsibilities

| File | Responsibility |
|---|---|
| `model.py` | `SamplerNet`: depthwise-separable CNN, predicts confidence map from encoder features |
| `uncertainty.py` | `UncertaintyHead`: lightweight coord predictor with Dropout2d for MC variance |
| `trainer.py` | `SamplerTrainer`: Phase 1 loop — compute error/variance maps, train SamplerNet via MSE |
| `buffer_sampler.py` | `fill_buffer_with_sampler`: Phase 2 — top-k + random sampling, identical buffer schema |
| `options.py` | Argument groups for Phase 1 (`add_sampler_train_args`) and Phase 2 (`add_sampler_buffer_args`) |
| `train_ace_sampler.py` | CLI entry: parse args → `SamplerTrainer(options).train()` |

## 3. Tensor Flow Map (Interface Contracts)

### Phase 1 — SamplerTrainer

```
Input image:        (1, 1, H, W)   float16/32  cuda   — grayscale, from CamLocDataset
                         ↓ Regressor.get_features()
Encoder features:   (1, 512, Hf, Wf)  float16  cuda   — Hf=H/8, Wf=W/8
                         ↓ Regressor.get_scene_coordinates()
Scene coords:       (1, 3, Hf, Wf)  float32   cuda   — 3D scene coordinates
                         ↓ project via gt_pose_inv + intrinsics
Reprojection error: (1, 1, Hf, Wf)  float32   cuda   — per-pixel error in pixels
                         ↓ (optional) UncertaintyHead.mc_coordinate_samples() T=10
MC variance:        (1, 1, Hf, Wf)  float32   cuda   — reprojection variance in px²
                         ↓ exp(-(α·error + β·variance))
Confidence target:  (1, 1, Hf, Wf)  float32   cuda   — ∈ [0, 1]
                         ↓ SamplerNet(features)
Confidence pred:    (1, 1, Hf, Wf)  float32   cuda   — ∈ [0, 1]
                         ↓ MSE loss
Loss:               scalar          float32   cuda
```

### Phase 2 — fill_buffer_with_sampler

```
Input image:        (1, 1, H, W)   float16/32  cuda
                         ↓ Regressor.get_features()
Encoder features:   (1, 512, Hf, Wf)  float16  cuda
                         ↓ SamplerNet (frozen)
Confidence map:     (Hf, Wf)       float32   cuda   — masked by image_mask
                         ↓ top-k (Part A) + multinomial (Part B)
Pixel coords:       (N, 2)         float32   cuda   — in original image space
                         ↓ grid_sample on features
Sampled features:   (N, 512)       float16   cuda   — stored in buffer
Buffer entry:       features(N,512) + target_px(N,2) + gt_poses_inv(N,3,4)
                    + intrinsics(N,3,3) + intrinsics_inv(N,3,3)
```

### Key Dimensions
- `H, W`: original image height/width (default 480 × ~640)
- `Hf = H/8, Wf = W/8`: feature map resolution (ACE FCN encoder, 8× downsampling)
- `N = samples_per_image` (default 1024 per image)
- Buffer total: `training_buffer_size` entries (default 8M)

## 4. Reuse Decisions

| Module | Decision | Source |
|---|---|---|
| ACE encoder + head | REUSE frozen | `ace_network.py:Regressor` |
| Dataset loader | REUSE unchanged | `dataset_origin.py:CamLocDataset` |
| Pixel grid | REUSE unchanged | `ace_util.py:get_pixel_grid` |
| Buffer schema | REUSE identical | `ace_trainer.py` buffer dict keys |
| SamplerNet | NEW | `ace_sampler/model.py` |
| UncertaintyHead | NEW | `ace_sampler/uncertainty.py` |
| Phase 1 training loop | NEW | `ace_sampler/trainer.py` |
| Phase 2 buffer filling | NEW (drop-in) | `ace_sampler/buffer_sampler.py` |

## 5. Unverified Assumptions & Hardware Constraints

- **Assumption**: The frozen head's reprojection error is a meaningful training signal.
  Requires the head to be at least partially trained (not random initialization).
- **Assumption**: SamplerNet confidence transfers from Phase 1 head to Phase 2 head.
  Partially mitigated by Part B random sampling.
- **Hardware**: Phase 1 runs on a single GPU. No DDP needed (~80K param network).
- **Memory**: Phase 1 peak VRAM ≈ same as ACE inference (~2GB for batch_size=1).
  MC Dropout with T=10 adds ~10× the UncertaintyHead forward pass cost (negligible).
