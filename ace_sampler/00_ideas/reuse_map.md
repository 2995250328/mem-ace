# Reuse Map

| Existing File | Relevant Class/Function | Can Be Used For |
|---|---|---|
| `../ace_network.py` | `Regressor` | Frozen ACE regressor in Phase 1 (feature extraction + coord prediction) |
| `../ace_trainer.py` | `AceTrainer._fill_buffer` | Phase 2 drop-in: replace `torch.multinomial` with `fill_buffer_with_sampler` |
| `../dataset_origin.py` | `CamLocDataset` | Training data loader for Phase 1 SamplerNet training |
| `../ace_util.py` | `get_pixel_grid` | Pixel grid for reprojection error computation |
| `../ace_encoder_pretrained.pt` | — | Frozen FCN encoder weights (loaded via `Regressor`) |

## What is NOT reused (new code)

| New File | Reason |
|---|---|
| `ace_sampler/model.py` | SamplerNet architecture (new, no existing analog) |
| `ace_sampler/uncertainty.py` | UncertaintyHead with MC Dropout (new) |
| `ace_sampler/trainer.py` | Phase 1 training loop (new logic) |
| `ace_sampler/buffer_sampler.py` | Phase 2 buffer filling with confidence-guided sampling (new) |
| `ace_sampler/options.py` | Argument parsers for Phase 1 and Phase 2 (new) |
| `train_ace_sampler.py` | Phase 1 entry script (new) |
