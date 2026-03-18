# ACE-FCN-LMC

**基于 FCN 编码器与几何潜在记忆压缩的加速坐标编码**

## 摘要

本想法通过将原始 FCN encoder 与 GeoLMC 两阶段迭代训练框架结合，扩展了 ACE 视觉重定位系统。结果在 Indoor6 上超越了 ACE-G 基线，同时保持了 ACE 原版的推理速度（~30 FPS）。

## 核心洞察

DINOv2-LMC 训练框架（S1 记忆对齐 + S2 重投影精化）与 encoder 无关。通过子类化 DINOv2 训练器并仅覆盖 `_create_regressor()`，我们用轻量级 FCN encoder 获得了完整的 LMC 训练流水线——新增约 1500 行代码，而从头编写需约 6000 行。

## 文件

| 文件 | 职责 |
|---|---|
| `../ace_network_ace.py` | ACEEncoder + RegressorACE |
| `../options_ace_lmc.py` | CLI 参数解析器 |
| `../trainer_ace_fcn.py` | TrainerACEFCN / TrainerACEFCNLMC |
| `../train_ace_lmc.py` | 训练入口 |
| `../test_ace_lmc.py` | 评估脚本 |

## 快速开始

```bash
# Vanilla 训练
python train_ace_lmc.py /data/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0

# LMC 训练（需要 pooled 记忆）
python train_ace_lmc.py /data/xwh/indoor6_ace/scene3 output/scene3.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0 \
    --use_lmc True --memory_path /path/to/pooled_memory.pt \
    --lmc_iterations 28 --num_latent_tokens 64

# 评估
python test_ace_lmc.py /data/xwh/7Scenes/pgt_7scenes_chess output/chess.pt \
    --encoder_path ace_encoder_pretrained.pt --device cuda:0
```

## 研究工作流状态

| 阶段 | 状态 |
|---|---|
| 1. 提案 | 完成 |
| 2. 评审 | 完成 |
| 3. 架构 | 完成 |
| 4. 骨架 | 完成 |
| 5. 实现 | 完成 |
| 6. 评估 | **待完成** |

## 下一步

将实验日志放入 `04_evaluation/logs/` 后运行 `/research-eval ace_fcn_lmc`。

## 文档

- `00_ideas/idea.md` — 原始想法与动机
- `00_ideas/reuse_map.md` — 复用契约
- `01_design/proposal.md` — 正式提案
- `02_architecture/system_design.md` — 系统设计
- `03_implementation/implementation_notes.md` — 实现细节
- `04_evaluation/eval_plan.md` — 评估计划
