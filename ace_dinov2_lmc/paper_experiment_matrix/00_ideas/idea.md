# Idea: Final-Paper Experiment Matrix for LMC Relocalization

The work currently has two strong empirical anchors. First, DINO+LMC performs extremely well on Indoor6 and appears to surpass SOTA. Second, GLACE+LMC improves broadly on Wayspots. However, DINO+LMC is not uniformly competitive: on datasets where the DINOACE base model is weak, LMC improves DINOACE but remains below SOTA.

The final paper should therefore avoid claiming that one backbone dominates everywhere. Instead, it should make a sharper claim: LMC is a memory-compression and scene-conditioning mechanism that improves compatible relocalization backbones, with especially strong gains under large indoor memory/query shifts and with GLACE-style global/local coordinate regressors.

The experiment matrix must answer likely reviewer questions:
- Is Indoor6 success real or cherry-picked?
- Does GLACE+LMC also help Indoor6, not only Wayspots?
- Are Wayspots gains complete across scenes and thresholds?
- Does the method scale beyond small indoor scenes?
- Does LMC help only SCR, or can it help APR/map-relative pose regression too?
- Which part of LMC matters: memory extraction, compression, fusion, gate, query training, or backbone choice?

The plan should prioritize reviewer-proof coverage over running every possible dataset/model combination.
