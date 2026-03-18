# ACE-FCN-LMC

**ACE with FCN Encoder and Geometric Latent Memory Compression**

## Summary

This idea extends the ACE visual relocalization system by combining the original FCN encoder with the GeoLMC two-stage iterative training framework. The result surpasses the ACE-G baseline on Indoor6 while maintaining ACE's original inference speed (~30 FPS).

## Key Insight

The DINOv2-LMC training framework (S1 memory alignment + S2 reprojection refinement) is encoder-agnostic. By subclassing the DINOv2 trainers and overriding only `_create_regressor()`, we get the full LMC training pipeline with the lightweight FCN encoder — ~1500 lines of new code vs ~6000 from scratch.

## Files

| File | Role |
|---|---|
| `../ace_network_ace.py` | ACEEncoder + RegressorACE |
| `../options_ace_lmc.py` | CLI argument parser |
| `../trainer_ace_fcn.py` | TrainerACEFCN / TrainerACEFCNLMC |
| `../train_ace_lmc.py` | Training entry point |
| `../test_ace_lmc.py` | Evaluation script |

## Quick Start

```bash
# Vanilla training
python train_ace_lmc.py /data/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0

# LMC training (requires pooled memory)
python train_ace_lmc.py /data/xwh/indoor6_ace/scene3 output/scene3.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0 \
    --use_lmc True --memory_path /path/to/pooled_memory.pt \
    --lmc_iterations 28 --num_latent_tokens 64

# Evaluation
python test_ace_lmc.py /data/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0
```

## Research Workflow Status

| Stage | Status |
|---|---|
| 1. Proposal | complete |
| 2. Review | complete |
| 3. Architecture | complete |
| 4. Skeleton | complete |
| 5. Implementation | complete |
| 6. Evaluation | **pending** |

## Next Step

Run `/research-eval ace_fcn_lmc` after placing experiment logs in `04_evaluation/logs/`.

## Documentation

- `00_ideas/idea.md` — raw idea and motivation
- `00_ideas/reuse_map.md` — reuse contract
- `01_design/proposal.md` — formal proposal
- `02_architecture/system_design.md` — system design
- `03_implementation/implementation_notes.md` — implementation details
- `04_evaluation/eval_plan.md` — evaluation plan
