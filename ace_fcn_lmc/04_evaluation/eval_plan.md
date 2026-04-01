# Evaluation Plan: ACE-FCN-LMC

## Actual Log Location

Training logs are written to the hierarchical output directory, NOT to this folder directly:

```
output/
└── {dataset}/
    └── {scene}/
        ├── ace_fcn_vanilla/{timestamp}_{config_tag}/
        │   ├── training_full_log.txt   ← full training log
        │   ├── post_train_eval.txt     ← eval metrics (tab-separated)
        │   ├── run_config.json         ← full config snapshot
        │   └── eval_results/           ← JSON + txt eval results
        ├── ace_fcn_lmc_iter/{timestamp}_{config_tag}/
        └── ace_fcn_lmc_aceg/{timestamp}_{config_tag}/
```

The `04_evaluation/logs/` directory here is the staging area for the `/research-eval` skill.

## Collecting Logs for research-eval

Run this script from the project root to copy relevant eval files here:

```bash
python ace_fcn_lmc/04_evaluation/collect_logs.py
```

Or manually copy:
```bash
# Example: copy all post_train_eval.txt files
find output/ -name "post_train_eval.txt" -exec cp {} ace_fcn_lmc/04_evaluation/logs/ \;

# Or copy entire run directories
cp -r output/7Scenes/pgt_7scenes_chess/ace_fcn_lmc_aceg/*/  ace_fcn_lmc/04_evaluation/logs/
```

## Running Evaluation

```bash
# 7-Scenes Chess
python ace_fcn_lmc/test_ace_lmc.py \
    /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
    <checkpoint.pt> \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 --session eval_chess

# Indoor6 Scene3
python ace_fcn_lmc/test_ace_lmc.py \
    /mnt/storage/xwh/indoor6_ace/scene3 \
    <checkpoint.pt> \
    --encoder_path ace_encoder_pretrained.pt \
    --device cuda:0 --session eval_scene3
```

## Triggering Stage 6

Once logs are in `04_evaluation/logs/`, run:
```
/research-eval ace_fcn_lmc
```
