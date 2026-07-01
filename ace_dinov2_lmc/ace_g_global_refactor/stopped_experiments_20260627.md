# Stopped non-structure experiments on 2026-06-27

Stop time: 2026-06-27 12:49:07 CST

Reason: free GPUs for structure-level fusion modification/validation. These runs are mainly training-protocol or hyper-parameter probes, not the current priority.

## Stopped sessions

### 1. Wayspots Stage2 R2 big-buffer probe

- Session: `indoor6_wayspots_stage2_r2_bigbuf_20260626_gpu01`
- Window: `wayspots`
- GPU/process at stop decision: GPU0, PID `112332`
- Run root: `/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stage2/stage2_r2_fusion/20260627_bears_sq_it10_buf10m_final12m_h256_gpu0`
- Active scene/config when stopped: `wayspots_bears`, Stage2 R2 `glace_concat`, `it10`, `10M -> 12M`, `bs8192`, eval/post-train hypotheses 256
- Progress at stop decision: Bears had completed at least `iter_01` and `iter_02` summaries and was in evaluation around frame 239.
- Importance: low for current structure work. This is a Stage2 training-protocol probe.
- Resume later by rerunning the same launcher/root if needed:
  - launcher: `ace_dinov2_lmc/scripts/launch_indoor6_wayspots_stage2_r2_bigbuf_probe_gpu01.sh`
  - suggested env: `ACTION=worker DATASET=wayspots WAYSPOTS_GPU=<gpu> WAYSPOTS_RUN_ROOT=<run_root>`

### 2. Cambridge Stage2 R2 hparam probe

- Session: `cambridge_stage2_r2_hparam_20260626_gpu01`
- Window: `stmary`
- GPU/process at stop decision: GPU1, PID `26464`
- Run root: `/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge/stage2/r2_fusion/20260626_court_stmary_it12-14_buf20-24m_h256_gpu01`
- Active scene/config when stopped: `Cambridge_StMarysChurch`, Stage2 R2 `glace_concat`, `it14`, `24M -> 24M`, `lr0.001`, `bs8192`, eval/post-train hypotheses 256
- Progress at stop decision: `stage2_r2_it14_buf24m_lr001...` had completed summaries through `iter_13`; process was in final `iter_14` training.
- Last known result before stop:
  - earlier 20M lr0.001 complete: best StMary median roughly `7.917 cm / 0.263 deg`
  - 24M lr0.001 partial through iter13 was already better than 20M; prior observed best around `7.6 cm / 0.25 deg`
- Importance: medium but not urgent. This is hparam/protocol refinement; enough trend signal exists.

### 3. Indoor6 DINOv2 + MapAnything efficiency matrix

- Session: `indoor6_dino_ma_eff_20260627_gpu23`
- Windows: `scene2a`, `scene5`
- GPU/processes at stop decision:
  - GPU2, PID `121841`, scene2a
  - GPU3, PID `123124`, scene5
- Run root: `/data/xwh/ace_dinov2_lmc/04_evaluation/indoor6/stage1/dinov2_mapanything_efficiency/20260627_scene2a_scene5_budget_matrix_gpu23`
- Active config when stopped: `fast4_b5_f10`, DINOv2 + MapAnything memory, `lmc_iterations=4`, `5M -> 10M`, `bs10240`, `samples_per_image=384`, eval/post-train hypotheses 256
- Progress at stop decision: scene2a and scene5 had completed at least `iter_01` and `iter_02` summaries.
- Importance: low for current structure work. This is a training-efficiency matrix, not fusion architecture validation.

## Current priority after stop

Use freed GPUs for structure-level fusion experiments. Do not spend the next cycle on additional Stage2 buffer/iter/LR tuning unless needed as a control.

