# Indoor6 scene1 训练命令

参考 `output/indoor6_ace/scene1` 下已有 run_command，整理两条命令：**LMC 对应 baseline（vanilla）** 与 **当前最优 LMC**，分别在 **cuda:2**、**cuda:3** 上运行。

---

## 1. LMC 对应 Baseline（vanilla，cuda:2）

与 LMC 实验对齐：同 buffer（2.56M）、同迭代数（10）、同 batch/lr 等，仅关闭 LMC。

```bash
cd /home/xwh/project/ace_depth

python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/indoor6_ace/scene1 \
    scene1_dino_iterbase.pt \
    --experiment_root output \
    --device cuda:2 \
    --use_lmc False \
    --apply_baseline_contract False \
    --vanilla_iterations 10 \
    --use_half False \
    --training_buffer_size 2560000 \
    --samples_per_image 384 \
    --batch_size 5120 \
    --image_resolution 518 \
    --epochs 24 \
    --learning_rate_max 8e-4 \
    --output_layout hierarchical \
    --eval_each_iteration True \
    --keep_best_only True \
    --best_metric pct5 \
    --eval_after_train True
```

---

## 2. 当前最优 LMC（cuda:3）

对应目录：`dino_ace_lmc_s1s2/20260302_003913_scene1_local_buf2.6M_K48_it10_bs5120_onecycle_improved`，best pct5≈56.95%（iter09）。

```bash
cd /home/xwh/project/ace_depth

python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/indoor6_ace/scene1 \
    scene1_lmc_local.pt \
    --experiment_root output \
    --device cuda:3 \
    --use_lmc True \
    --use_half False \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/scene1_train/40views/20260227_223246/Indoor6_scene1_train_pooled_GT.pt \
    --lmc_mode local \
    --lmc_auto_mode_by_visibility False \
    --num_latent_tokens 48 \
    --lmc_iterations 10 \
    --lmc_warmup_steps 1200 \
    --lmc_train_steps 300 \
    --s1_batch_size 8 \
    --s1_learning_rate_max 6e-5 \
    --s2_learning_rate_max 8e-4 \
    --batch_size 5120 \
    --image_resolution 518 \
    --eval_each_iteration True \
    --keep_best_only True \
    --best_metric pct5
```

---

## 说明

- **数据路径**：`/mnt/storage/xwh/indoor6_ace/scene1`
- **输出**：在 `--experiment_root output` 下为 hierarchical，即 `output/indoor6_ace/scene1/dino_ace_baseline/` 或 `dino_ace_lmc_s1s2/` 下带时间戳的 run 目录。
- **Baseline** 与 **LMC** 的 buffer/迭代数/分辨率等一致，便于直接对比 pct5。
