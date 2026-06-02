# Idea: Stage2 Global Reliability Control for ACE-FCN-LMC

We are studying ACE-FCN-LMC Stage2 on Wayspots, where Stage1 trains an ACE-FCN feature-space local memory compressor without GLACE global features, and Stage2 adds GLACE image-level global feature conditioning on top of the Stage1 local stack.

Current observations show scene-dependent global reliability. Bears benefits from stronger GLACE global injection: bounded gate max=0.1 reaches Acc5≈82.41 and raw concat reaches ≈83.28. SquareBench shows negative transfer: Stage1 local reaches Acc5≈53.03, bounded gate max=0.01 protects it at ≈51.65, but max=0.1 drops to ≈37.44 and raw/unconstrained global drops to ≈32.x.

Therefore the problem is not selecting one scalar gate upper bound. The research goal is to design Stage2 mechanisms that exploit GLACE global features when reliable, but prevent them from destroying the Stage1 local coordinate predictor when unreliable.

Candidate directions:
- Stage1 local consistency loss during Stage2.
- Zero-initialized residual global branch with regularization.
- Per-image or per-token adaptive global gates instead of one scene scalar.
- Validation-based gate selection or checkpoint selection.
- Local/global dual-head output with selection or fusion.

Initial validation should focus on Bears and SquareBench before scaling to all Wayspots scenes.
