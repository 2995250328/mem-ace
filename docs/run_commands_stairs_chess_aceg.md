# Stairs / Chess 最优 LMC 训练命令（对齐 heads_aceg_full_refill）

基于 heads 最优结果（pct5=89.70%，ACE-G + full refill）的配置，仅改场景、memory 路径与 GPU。

---

## Stairs（cuda:2）

```bash
cd /home/xwh/project/ace_depth

python train_ace_dinov2_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_stairs \
    stairs_aceg_full_refill.pt \
    --device cuda:2 \
    --run_name stairs_aceg_full_refill \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/stairs_train/20views/20260126_234545/7Scenes_stairs_train_pooled_GT.pt \
    --dinov2_path /data/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --epochs 24 \
    --batch_size 5120 \
    --samples_per_image 384 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full
```

**若 stairs 显存不足**：将 `--batch_size 5120` 改为 `--batch_size 512`。

---

## Chess（cuda:3）

```bash
cd /home/xwh/project/ace_depth

python train_ace_dinov2_lmc.py \
    /data/xwh/7Scenes/pgt_7scenes_chess \
    chess_aceg_full_refill.pt \
    --device cuda:3 \
    --run_name chess_aceg_full_refill \
    --use_lmc True \
    --memory_path /home/xwh/project/map-anything-experiments/memory_extract/chess_train/40views/20260227_232518/7Scenes_chess_train_pooled_GT_sparse_removal1000.pt \
    --dinov2_path /data/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
    --lmc_mode global \
    --num_latent_tokens 64 \
    --lmc_iterations 28 \
    --lmc_warmup_steps 2000 \
    --lmc_train_steps 600 \
    --s1_learning_rate_max 1e-4 \
    --s2_learning_rate_max 1e-3 \
    --training_buffer_size 2560000 \
    --buffer_size_final 7680000 \
    --epochs 24 \
    --batch_size 5120 \
    --samples_per_image 384 \
    --buffer_batch_size 1 \
    --buffer_on_cpu True \
    --image_resolution 518 \
    --s1_batch_size 16 \
    --eval_each_iteration True \
    --eval_after_train True \
    --keep_best_only True \
    --best_metric pct5 \
    --use_half True \
    --lmc_memory_preflight True \
    --lmc_memory_preflight_strict True \
    --lmc_flow ace_g \
    --ace_g_fusion_in_s2 True \
    --ace_g_fusion_lr_ratio 0.01 \
    --ace_g_cross_iter_eval True \
    --s1_use_buffer True \
    --s1_loss_mode sample_per_image \
    --s1_buffer_refill_mode full
```

---

## 并行执行

两个终端分别执行上述两条命令即可；stairs 占 cuda:2，chess 占 cuda:3。
