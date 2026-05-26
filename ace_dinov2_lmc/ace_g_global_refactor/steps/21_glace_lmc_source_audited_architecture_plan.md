# Step 21: Source-Audited GLACE-LMC Architecture and Training Plan

## 1. Purpose

This note defines a source-audited architecture and experiment plan for integrating ACE-DINOv2-LMC with GLACE on Wayspots bears.

The goal is to use LMC as a plugin-style enhancement for GLACE, not to rewrite or damage GLACE's original logic.

The immediate validation question is:

```text
Can the existing MapAnything/BSE memory extraction strategy improve GLACE on Wayspots bears?
```

The current scope explicitly does **not** switch to GLACE feature-space memory. GLACE feature memory remains a later direction if the current MapAnything/BSE memory cannot help under a safe plugin design.

## 2. Source-Audited Facts

### 2.1 GLACE feature structure

The GLACE repository confirms the default decoder/head input is:

```text
[global_256, local_512] = 768 channels
```

Evidence:

- Local ACE encoder output is 512 channels: `/home/xwh/project/glace/ace_network.py:21`.
- R2Former global descriptor is 256 channels: `/home/xwh/project/glace/datasets/extract_features.py:95`.
- Saved `features.npy` has shape `(len(dataset), 256)`: `/home/xwh/project/glace/datasets/extract_features.py:124`.
- Test-time GLACE expands global descriptors spatially and prepends them before local features: `/home/xwh/project/glace/ace_network.py:294-295`.
- Training reconstructs the same global-first feature layout before head updates: `/home/xwh/project/glace/ace_trainer.py:415-419`.
- Trainer treats the first `global_feat_dim` channels as the global slice: `/home/xwh/project/glace/ace_trainer.py:440`.

Therefore any GLACE-LMC plugin that operates on decoder features must preserve this channel contract:

```text
channels [0:256)   = GLACE global descriptor
channels [256:768) = GLACE local encoder feature
```

### 2.2 Why GLACE global channels should be protected

GLACE's head uses the same head activations for both multi-center routing and coordinate residual regression.

Evidence:

- Multi-center position decoder is enabled when `mean.numel() > 3`: `/home/xwh/project/glace/ace_network.py:108`.
- `fcc` predicts center logits from head activations: `/home/xwh/project/glace/ace_network.py:110`.
- Centers are stored as `heads.centers`: `/home/xwh/project/glace/ace_network.py:126`.
- The selected mean is the softmax-weighted center mixture: `/home/xwh/project/glace/ace_network.py:147`.
- Coordinate residual is predicted by `fc3(sc)`: `/home/xwh/project/glace/ace_network.py:148`.
- Final coordinate is residual plus selected mean: `/home/xwh/project/glace/ace_network.py:158`.

This means changing the semantics of the global 256 channels can affect both:

1. center selection / mixture, and
2. coordinate residual regression.

Therefore full 768-channel residual injection is high risk. It can perturb GLACE's global-local contract and multi-center behavior.

### 2.3 ACE-DINOv2-LMC GLACE integration

The ACE-DINOv2-LMC side already has a GLACE backend path.

Evidence:

- `--model_backend` supports `glace_lmc`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py:45`.
- GLACE root / encoder / feature options are defined at `/home/xwh/project/ace_depth/ace_dinov2_lmc/options_dinov2_lmc.py:149`, `:155`, and `:161`.
- GLACE-LMC requires `--use_lmc True` and a valid memory path: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1214`.
- GLACE-LMC currently requires `--lmc_flow ace_g`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1218`.
- Training assembles GLACE decoder features as global-first then local: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1085`, `:1098`, `:1116`.

The current integration is therefore structurally aligned with the GLACE repository's channel order.

### 2.4 Residual adapter behavior

`GLACEDecoderFeatureResidualAdapter` is implemented in:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:22
```

It supports two modes:

```text
decoder_delta_tanh_scalar
local_delta_tanh_scalar
```

Evidence:

- Mode choices: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:39`.
- Gate is a trainable parameter initialized from `residual_gate_init`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:43`.
- Effective residual gain is `torch.tanh(self.residual_logit)`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:45`.
- Local-only mode is active when `mode == "local_delta_tanh_scalar"` and `global_dim > 0`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:55`.
- Local-only mode preserves the global prefix: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:61`.
- Local-only mode residuals only the local tail: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:62-64`.
- Non-local mode residuals the full decoder tensor: `/home/xwh/project/ace_depth/ace_dinov2_lmc/glace_backend.py:66`.

The safe primary mode is therefore:

```text
--glace_residual_mode local_delta_tanh_scalar
```

The full decoder mode should be treated as legacy compatibility or ablation only:

```text
--glace_residual_mode decoder_delta_tanh_scalar
```

### 2.5 Checkpoint and evaluation path

The plugin path is saved and restored by the current training/evaluation code.

Evidence:

- LMC config records GLACE global dim, fusion query type, residual mode, gate init, and residual global dim: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:1805`, `:1808-1811`.
- Checkpoint saves `glace_residual_adapter_state_dict`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5538`.
- Checkpoint config records final GLACE residual gain: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:5571`.
- Evaluation detects `model_backend == 'glace_lmc'`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py:293-300`.
- Evaluation reconstructs the GLACE regressor from saved head state: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py:307`.
- Evaluation constructs and loads the residual adapter: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py:423-429`.
- Evaluation applies fusion and then the GLACE residual adapter before `network.get_scene_coordinates`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/test_ace_dinov2_lmc.py:610-622`.

## 3. Baselines and Prior GLACE-LMC Evidence

### 3.1 Wayspots bears baseline metrics

Canonical baseline summary:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_baselines/20260525_190055/summary.md
```

Reported metrics:

| Method | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | Median |
|---|---:|---:|---:|---:|---:|---|
| ACE | 95.345 | 86.034 | 72.586 | 6.552 | 0.862 | 1.105deg / 3.510cm |
| DINOACE | 76.207 | 62.414 | 34.828 | 6.552 | 0.000 | 2.317deg / 7.276cm |
| ACE-G25 | 95.517 | 84.138 | 71.552 | 28.103 | 3.966 | 0.954deg / 2.829cm |
| GLACE | 94.655 | 88.448 | 77.759 | 12.931 | 1.724 | 0.996deg / 3.085cm |

Interpretation:

- GLACE is strongest at 5cm/5deg.
- ACE-G25 is stronger on strict metrics and median translation.
- GLACE-LMC should preserve GLACE's 5cm recall and try to move strict metrics/median toward ACE-G25.

### 3.2 Prior GLACE-LMC run

Main retrieved run:

```text
/home/xwh/project/ace_depth/ace_dinov2_lmc/04_evaluation/wayspots_glace_lmc/train/memory_pooled_vs_asb/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260526_120550_aceg_fS2cie_global_res518_buf3.2M_F6.4M_K64_it28_ep24_bs10240_spi_s1buf_sp512_onecycle_improved
```

Best logged per-iteration result:

```text
iter23
pct5 = 75.3448
pct10_5 = 84.3103
pct25_5 = 90.5172
median = 0.9473deg / 3.3460cm
```

Source:

```text
best_K64_it28_wayspots_bears_glace_lmc_eval_log.txt:24-30
```

Cross-eval iter23:

```text
25cm/5deg = 90.00
10cm/5deg = 86.03
5cm/5deg  = 76.03
2cm/2deg  = 15.52
1cm/1deg  = 1.38
median    = 0.9768deg / 3.2485cm
```

Source:

```text
eval_summary_wayspots_bears_iter_23_cross.txt:1-10
```

Iter28 cross-eval degraded:

```text
5cm/5deg = 72.93
median = 0.9764deg / 3.2785cm
final_lmc_fusion_assembly_gamma = 0.0
```

Sources:

```text
eval_summary_wayspots_bears_iter_28_cross.txt:1-10
eval_summary_wayspots_bears_iter_28_cross.txt:61-71
```

Important caveat:

The actual `20260526_120550` run command did not include the newer explicit residual options:

```text
--glace_residual_mode local_delta_tanh_scalar
--glace_residual_gate_init 0.20
```

The current launcher does default to these values, but the old run should not be treated as a definitive test of the new local-only residual design.

Current launcher defaults:

- `GLACE_RESIDUAL_MODE=local_delta_tanh_scalar`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_bears.sh:17`.
- `GLACE_RESIDUAL_GATE_INIT=0.20`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_bears.sh:18`.
- `LMC_FUSION_REFINEMENT_MODE=cascade_internal`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_bears.sh:28`.
- `ACE_G_FUSION_LR_RATIO=0.03`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_bears.sh:31`.
- `BUFFER_SIZE_FINAL=8000000`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/scripts/run_wayspots_glace_lmc_bears.sh:35`.

## 4. Architecture Recommendation

### 4.1 Main design

Use LMC as a residual local-feature enhancer:

```text
base_decoder = concat(global_256, local_512)

memory_context = LMC(memory, query=base_decoder)

candidate_decoder = fusion(base_decoder, memory_context)

output_global = base_global
output_local  = base_local + tanh(gate) * (candidate_local - base_local)

output_decoder = concat(output_global, output_local)

scene_coords = original_GLACE_head(output_decoder)
```

The GLACE head and multi-center logic remain unchanged.

### 4.2 Why not full 768 residual as the main path

Full decoder residual is:

```text
output_decoder = base_decoder + tanh(gate) * (candidate_decoder - base_decoder)
```

This modifies both global and local channels.

Because GLACE's global channels participate in the head activations used by both `fcc` center routing and `fc3` coordinate residual prediction, full residual can alter the meaning of GLACE's scene-level descriptor and destabilize the learned multi-center behavior.

Therefore full decoder residual should remain only an ablation/legacy mode.

### 4.3 Memory feature-space mismatch

Current memory is MapAnything/BSE memory, not GLACE feature-space memory:

```text
memory_bse.pt
features roughly (38138, 3840)
layers_idx = [0, 6, 12, 18, final]
per-layer dim = 768
```

The correct interpretation is not:

```text
memory feature replaces GLACE feature
```

The correct interpretation is:

```text
memory provides geometry-aware context used to generate a residual correction in GLACE local feature space
```

This implies the plugin should use explicit learned projections/adapters between:

1. GLACE decoder query space,
2. MapAnything/BSE memory space, and
3. GLACE local residual output space.

The final residual output should be local-only unless a later experiment proves full-global modification is beneficial.

## 5. Training Strategy Recommendation

### 5.1 Current optimizer capability

Current code supports separate S2 optimizer groups for head vs fusion/residual when `ace_g_fusion_in_s2=True`.

Evidence:

- Fusion/residual params are added when `ace_g_fusion_in_s2=True`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4325`.
- S2 optimizer uses separate parameter groups: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4333`.
- Residual adapter parameters are grouped with fusion parameters: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4335`.
- Scheduler max LR is `[head_lr, fusion_lr]`: `/home/xwh/project/ace_depth/ace_dinov2_lmc/trainer_dinov2_lmc.py:4374`.

However, the inspected code did not reveal an explicit head-freeze warmup switch or independent residual-adapter LR.

### 5.2 Recommended training policy

The preferred training policy is PEFT-style:

```text
Stage A: preserve GLACE baseline behavior
    initialize from a trained vanilla GLACE head if available
    freeze GLACE head initially
    train only compressor + fusion + residual adapter

Stage B: cautious joint adaptation
    unfreeze GLACE head
    use low head LR
    keep fusion/residual LR higher than head LR
```

Rationale:

- GLACE is already strong on 5cm/5deg.
- LMC memory is not in GLACE feature space.
- Early joint head/fusion training can cause the head to adapt to noisy residuals and lose GLACE's baseline behavior.
- Adapter/LoRA/VPT-style references support frozen-base residual adaptation before low-LR joint tuning.

Current code appears to support LR-ratio experiments, but head-freeze warmup likely requires a small additional option or implementation.

## 6. Experiment Plan

### Experiment 1: True local-only residual rerun

Purpose:

Verify whether the current safer local-only residual configuration improves over the old run.

Core settings:

```text
--model_backend glace_lmc
--glace_residual_mode local_delta_tanh_scalar
--glace_residual_gate_init 0.20
--lmc_fusion_assembly_gamma_init 0.05
--ace_g_fusion_in_s2 True
--ace_g_fusion_lr_ratio 0.03
--buffer_size_final 8000000
```

Hypothesis:

The old run underperformed partly because residual/fusion contribution was too weak or not using the newer safe defaults. A true local-only residual run should improve over the old `75-76` 5cm/5deg result and move closer to vanilla GLACE.

Success criteria:

```text
5cm/5deg >= 77.759 - 0.5
final_glace_residual_gain clearly > 0.02
final_lmc_fusion_assembly_gamma != 0 when assembly gamma is enabled
2cm/2deg improves over vanilla GLACE 12.931 or median improves toward ACE-G25
```

### Experiment 2: Local residual gate sweep

Purpose:

Find whether the memory residual has a useful injection strength.

Sweep:

```text
--glace_residual_gate_init 0.10
--glace_residual_gate_init 0.20
--glace_residual_gate_init 0.30
```

Keep other settings fixed.

Interpretation:

- `0.10` best: memory signal is useful but fragile; keep residual conservative.
- `0.20` best: current default is reasonable.
- `0.30` best: memory is under-injected in current settings.
- all below GLACE: likely feature-space mismatch or training dynamics issue.

Report for every run:

```text
25cm/5deg
10cm/5deg
5cm/5deg
2cm/2deg
1cm/1deg
median rot/trans
final_glace_residual_gain
final_lmc_fusion_assembly_gamma
delta/local norm if available
best iteration, not only final iteration
```

### Experiment 3: Head-freeze plugin warmup

Purpose:

Test whether current GLACE-LMC degradation comes from early joint co-adaptation between the GLACE head and noisy memory residual.

Proposed schedule:

```text
warmup phase:
    freeze GLACE head / multi-center head parameters
    train compressor + fusion + residual adapter

finetune phase:
    unfreeze GLACE head
    head_lr = 0.1x to 0.2x normal S2 LR
    fusion/residual_lr = 3x to 10x head_lr
```

Hypothesis:

If memory contains useful scene geometry, the plugin should learn a helpful local residual while the strong GLACE head remains stable. Low-LR head finetuning can then adapt to the residual without losing baseline behavior.

Implementation note:

The inspected code did not expose this as a current CLI switch. It likely requires adding a small training option, for example:

```text
--glace_freeze_head_warmup_iters N
--glace_head_lr_ratio_after_warmup R
```

or an equivalent phase-specific optimizer construction.

## 7. Evaluation Contract

A GLACE-LMC run should not be judged only by 5cm/5deg.

Use a three-level contract:

### 7.1 Safety threshold

```text
5cm/5deg >= 77.759 - 0.5
```

The plugin should not materially degrade vanilla GLACE recall.

### 7.2 Useful improvement threshold

Any of:

```text
5cm/5deg > 77.759
2cm/2deg > 12.931
median trans < 3.085cm
median rot < 0.996deg
```

### 7.3 Strong target

```text
5cm/5deg >= 78
2cm/2deg approaches 20+
median approaches ACE-G25: 0.954deg / 2.829cm
```

The desired final behavior is:

```text
GLACE-like 5cm recall + ACE-G25-like strict accuracy
```

## 8. External Reference Principles

The following external design patterns support the proposed direction:

1. Adapter-style fine-tuning: freeze strong base model and train small residual modules.
2. LoRA-style residual updates: preserve original path and add a trainable delta.
3. Visual Prompt Tuning: train lightweight context while keeping the vision backbone fixed.
4. Zero-init / near-zero-init attention adapters: prevent early corruption of a strong base model.
5. Memory/context localization systems such as SACReg and NeuMap: map/database context must be projected/aligned before interacting with query features.

Transferred to GLACE-LMC:

```text
preserve GLACE path
add residual memory plugin
initialize safely
use learned projectors for feature-space mismatch
warm up plugin before joint head tuning
```

## 9. Recommended Next Actions

1. Run the true local-only residual configuration with explicit CLI flags and current script defaults.
2. Compare actual `run_command.txt` against `scripts/run_wayspots_glace_lmc_bears.sh` before interpreting results.
3. Record final residual gain and assembly gamma for every run.
4. If local-only gate 0.20 still underperforms GLACE, perform the gate sweep before changing memory source.
5. If all gate values underperform, implement head-freeze plugin warmup before switching to GLACE feature-space memory.
6. Only after these experiments should GLACE feature-space memory be treated as the next major direction.

## 10. Non-Goals for the Current Phase

Do not do the following in the current phase:

```text
modify GLACE repo core training logic
modify GLACE Head.forward()
modify multi-center center routing
replace GLACE global descriptor
switch immediately to GLACE feature-space memory
introduce per-token/per-channel gates before scalar local gate is validated
judge the current design from the old 20260526_120550 run alone
```

The current phase is a controlled validation of whether existing MapAnything/BSE memory can help GLACE through a safe local residual plugin.
