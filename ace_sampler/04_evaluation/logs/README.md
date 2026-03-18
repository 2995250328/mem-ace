# Evaluation Logs

Experiment results go here. Stage 6 is pending.

## Planned Experiments

1. Baseline: ACE random sampling (7-Scenes all scenes)
2. ACE + SamplerNet (reprojection error only, β=0)
3. ACE + SamplerNet + MC Dropout (full method)
4. Ablation: sampler_ratio ∈ {0.3, 0.5, 0.7, 0.9}
5. Ablation: random SamplerNet (untrained, Phase 1 skipped)

## Metrics

- Median translation error (cm)
- Median rotation error (°)
- % frames within 5cm/5°
- Phase 1 wall-clock time
- Total training time vs. baseline
