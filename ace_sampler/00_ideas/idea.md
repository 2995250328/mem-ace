# Idea: Learnable Buffer Sampling for ACE

ACE fills its training buffer by randomly sampling pixels from each training image.
This is suboptimal: the model wastes capacity re-learning regions it already localizes well,
while under-sampling informative regions where reprojection error is still high.

**Core idea**: train a lightweight `SamplerNet` on top of frozen ACE encoder features to
predict per-pixel confidence from the current model's reprojection error. Use this confidence
map to bias buffer sampling toward informative pixels.

**Optional extension**: augment the confidence target with MC Dropout uncertainty from a
lightweight `UncertaintyHead`, so that pixels with high prediction variance are also
preferred — even if their current reprojection error happens to be low.

**Two-phase workflow**:
- Phase 1: train SamplerNet (and optionally UncertaintyHead) using a frozen, pre-trained ACE head
- Phase 2: plug trained SamplerNet into `ace_trainer.py` buffer filling as a drop-in replacement

**Constraints**:
- SamplerNet must be tiny (~80K params) — it runs per-image during buffer filling
- Buffer schema must remain identical to original ACE (no downstream changes)
- Phase 2 must be backward-compatible: `--sampler_path None` = original behavior
