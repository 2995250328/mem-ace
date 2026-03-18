# 实现说明：ACE-FCN-LMC

## 状态：已完成（在 7-Scenes 和 Indoor6 上验证）

## 已实现文件

### `ace_network_ace.py`
- `ACEEncoder`：包装 `ace_network.Encoder`，内部处理 RGB→灰度转换，暴露 `feature_dim=512`、`patch_size=8`
- `RegressorACE`：组合 ACEEncoder + Head（来自 `ace_network_dinov2.py`），提供 `create_from_encoder()` 工厂方法
- 关键：`register_buffer("rgb_weights", ...)` 用于高效灰度转换，无需修改数据集加载器

### `options_ace_lmc.py`
- 用 ACE FCN 专属参数扩展 DINOv2 LMC 参数解析器
- 添加 `--encoder_path`（必需），移除 DINOv2 专属的 `--dinov2_path`
- 移除 `--image_resolution` 的 patch-size 约束（默认 480）
- 添加 `--data_backend`（ace/wai）以支持 WAI 数据集

### `trainer_ace_fcn.py`
- `TrainerACEFCN(TrainerACEDINOv2)`：仅覆盖 `_create_regressor()`
- `TrainerACEFCNLMC(TrainerACEDINOv2LMC)`：覆盖 `_create_regressor()` 和 `_evaluate_checkpoint()`
- `_evaluate_checkpoint()` 使用 `test_ace_lmc.run_evaluation_lmc` 而非 DINOv2 评估

### `train_ace_lmc.py`
- 入口点，镜像 `train_ace_dinov2_lmc.py`
- 支持 vanilla 模式（`--use_lmc False`）和 LMC 模式（`--use_lmc True`）
- 集成 `ResultManager` 用于层级化输出组织
- 训练后评估通过 `run_post_train_eval()` 并清理 GPU 显存

### `test_ace_lmc.py`
- 已训练 ACE FCN 模型的评估脚本
- `run_evaluation_lmc()`：加载 RegressorACE，运行 DSAC* RANSAC，计算指标
- 输出：`median_rErr`、`median_tErr`、`pct5`、`pct25_5`、`avg_time`

## 已验证配置

### 7-Scenes Chess（vanilla）
```bash
python train_ace_lmc.py /data/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0 \
    --training_buffer_size 2560000 --epochs 24 --batch_size 5120
```

### Indoor6 Scene3（LMC，超越 ACE-G）
```bash
./train_ace_lmc.py /data/xwh/indoor6_ace/scene3 scene3_lmc.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:1 \
    --use_lmc True \
    --memory_path /path/to/Indoor6_scene3_train_pooled_GT.pt \
    --lmc_iterations 28 --num_latent_tokens 64 --num_attn_layers 2 \
    --epochs 24 --batch_size 5120 --training_buffer_size 2560000
```

## 已知问题 / 注意事项

- 生产运行推荐 `--use_half True`（节省约 50% 检查点大小）
- 大场景在 12GB GPU 上需要 `--buffer_on_cpu True`
- WAI 数据后端（`--data_backend wai`）需要父目录中有 `map-anything` 仓库
- 记忆预检查（`--lmc_memory_preflight True`）在训练前验证 pooled 记忆格式
