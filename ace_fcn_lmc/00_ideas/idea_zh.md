# 想法：ACE FCN + GeoLMC（ace_fcn_lmc）

## 核心想法

将原始 ACE 的 FCN encoder 与 GeoLMC（几何潜在记忆压缩）两阶段训练框架结合，
在保持 ACE 原版推理速度（~30 FPS）的前提下，通过引入外部 3D 点云记忆（pooled memory）
来提升场景坐标回归精度，超越 ACE-G 基线。

## 动机

- ACE vanilla：训练快、推理快，但精度受限于单阶段 buffer 训练
- DINOv2 LMC：精度高但推理慢（~10-15 FPS），且需要 RGB 输入
- 目标：用 FCN encoder（灰度图、8x 下采样、512-dim）替换 DINOv2，
  继承 LMC 两阶段迭代训练框架，在低资源场景下超越 ACE-G

## 关键设计决策

1. **Encoder 替换**：ACEEncoder 包装原版 FCN encoder，提供与 DINOv2 一致的接口
2. **Trainer 继承**：TrainerACEFCN/TrainerACEFCNLMC 仅覆盖 `_create_regressor()`，
   所有训练逻辑继承自 TrainerACEDINOv2/TrainerACEDINOv2LMC
3. **两种训练模式**：
   - Vanilla：单阶段 buffer 训练（`--use_lmc False`）
   - LMC：两阶段迭代训练（`--use_lmc True --memory_path <pooled.pt>`）
4. **无分辨率约束**：FCN encoder 支持任意分辨率（无 patch-size 限制）

## 验证状态

- [x] 7-Scenes Chess：vanilla 和 LMC 模式均已验证
- [x] Indoor6 Scene3：LMC 完整训练已验证，超越 ACE-G 基线
- [x] 多轮迭代训练（vanilla_iterations=3）已验证
