# 复用映射

| 现有文件 | 相关类/函数 | 可用于 |
|---|---|---|
| `../ace_network.py` | `Regressor` | Phase 1 中冻结的 ACE 回归器（特征提取 + 坐标预测） |
| `../ace_trainer.py` | `AceTrainer._fill_buffer` | Phase 2 即插即用替换：将 `torch.multinomial` 替换为 `fill_buffer_with_sampler` |
| `../dataset_origin.py` | `CamLocDataset` | Phase 1 SamplerNet 训练的数据加载器 |
| `../ace_util.py` | `get_pixel_grid` | 用于重投影误差计算的像素网格 |
| `../ace_encoder_pretrained.pt` | — | 冻结的 FCN 编码器权重（通过 `Regressor` 加载） |

## 不复用的部分（新代码）

| 新文件 | 原因 |
|---|---|
| `ace_sampler/model.py` | SamplerNet 架构（全新，无现有对应实现） |
| `ace_sampler/uncertainty.py` | 带 MC Dropout 的 UncertaintyHead（全新） |
| `ace_sampler/trainer.py` | Phase 1 训练循环（全新逻辑） |
| `ace_sampler/buffer_sampler.py` | Phase 2 基于置信度引导采样的缓冲区填充（全新） |
| `ace_sampler/options.py` | Phase 1 和 Phase 2 的参数解析器（全新） |
| `train_ace_sampler.py` | Phase 1 入口脚本（全新） |
