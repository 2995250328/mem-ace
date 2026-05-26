# Step 16：大场景多 Memory 参考系方案

## 1. 目的

本文档记录 ACE-DINOv2-LMC / ACE-G 扩展到大场景多 MapAnything memory 时的当前设计决策。

核心问题不只是 memory 大小或推理成本。MapAnything 的几何和特征来自一个参考条件化的 forward。如果大场景被拆成多个 memory，每个 memory 可能有不同的参考帧，也可能有不同的参考条件化特征空间。因此系统不能默默地把所有 memory token 当成同一个全局一致 memory 来处理。

本文回答：

1. 多个 memory 是否应该拼接成一个大 memory；
2. 是否应该每个 memory 独立训练/评估；
3. 如何引入路由和最终 pose 选择；
4. 在更激进的共享模型设计前，必须保存哪些 metadata。

## 2. 当前代码和实验约束

当前代码契约是单 memory：

- 训练只接受一个 `--memory_path`；
- `load_memory_features()` 只接受一个 memory path；
- clustered memory package 会被显式拒绝，应传入 per-cluster memory file；
- trainer 构造一个 `memory_dict`；
- `GeoLMC` 压缩一个 memory；
- fusion 将一个 compressed memory 扩展到 query batch；
- test 阶段为所有测试帧缓存一个 compressed memory。

已有 ensemble 支持并不是原生多 memory fusion。它评估多个独立 checkpoint + memory bundle，然后根据 DSAC inlier count 等几何证据选择最终 pose。

之前的参考系工作也约束了设计：

- C0 仍是当前默认：MapAnything features 可以是 reference-conditioned，但 ACE head 预测世界坐标系下的 scene coordinates。
- C1 已存在：memory points/targets 可以被变换到参考帧，预测再恢复到 world 后做 reprojection。
- C1 已验证可跑通，但在之前 scene2a 实验中弱于 repaired C0，因此不应作为第一条大场景主线替换 C0。
- 一个一致的 MapAnything memory 应来自一次 MapAnything forward。多次 forward 可能使用不同 `T0` reference，因此产生不兼容的参考条件化特征空间。

## 3. 决策：第一阶段不要拼接 memories

现在不推荐直接把多个 MapAnything memory 拼成一个大 memory。

表面上看很简单：

```text
memory_big = concat(memory_0, memory_1, ..., memory_n)
train once with --memory_path memory_big.pt
```

但这样不安全，因为：

1. **参考帧不同。** 每个 memory 可能由不同 MapAnything reference view 生成。
2. **特征是 reference-conditioned。** 即使点云可以变换到共同世界坐标，关联特征仍可能携带不同参考上下文语义。
3. **当前 compressor/fusion 没有 frame awareness。** 它把 `pooled_points`、`pooled_features`、`scene_center` 当成一个一致集合处理，没有 `memory_id`、`frame_id` 或 per-token reference metadata。
4. **失败模式难以诊断。** 结果变差可能来自覆盖、路由、特征错配、坐标错配、latent 容量或训练不稳定。

只有在显式 frame-aware metadata 和模型支持存在后，才重新考虑 direct-concat 实验。

## 4. 近期路径：独立 local-memory bundles

推荐的第一条大场景路线是：

```text
memory_0 -> checkpoint_0 -> pose hypothesis_0
memory_1 -> checkpoint_1 -> pose hypothesis_1
memory_2 -> checkpoint_2 -> pose hypothesis_2
...
query -> route / evaluate hypotheses -> final pose
```

每个 MapAnything memory 被视为独立 local-map bundle：

```text
bundle_i = {
  memory_path_i,
  reference metadata_i,
  training config_i,
  checkpoint_i,
  eval outputs_i
}
```

这能隔离每个参考条件化特征空间，并复用当前单 memory 训练/评估代码。

默认训练目标保持 C0 world regression：

```text
memory features: bundle 内 reference-conditioned
memory points: 可用时采用 world-frame C0 contract
head output: P_world
loss/eval: world reprojection / DSAC
```

C1 可以保留为诊断分支，但除非新证据表明它优于 C0，否则不作为大场景 baseline 主线。

## 5. 测试时路由和最终 pose 选择

第一版路由层应该保守且非学习。

### 5.1 All-bundle baseline

初始小矩阵中，让每个 bundle 都在每个 query frame 上运行：

```text
for query in test_set:
  for bundle in bundles:
    predict pose_hypothesis(bundle, query)
  select final pose by geometric score
```

初始最终 pose 规则：

```text
final_pose = pose from bundle with highest DSAC inlier count
```

如果 inlier count 不够稳定，再比较：

```text
score = inlier_count - lambda * reprojection_error
```

或使用基于以下信息的小型手工排序：

- inlier count；
- reprojection error；
- pose stability；
- 已有 confidence。

### 5.2 必须报告的结果

每个实验需要报告：

1. best single bundle；
2. routed ensemble result；
3. per-bundle standalone results；
4. per-bundle selection frequency；
5. oracle upper bound。

oracle upper bound 只用于分析：

```text
oracle(query) = hypothesis with lowest true pose error
```

它回答多 memory 是否包含互补有效假设。如果 oracle 也弱，问题在 memory 覆盖或 per-bundle 模型质量。如果 oracle 强但 routed ensemble 弱，问题在 routing。

## 6. 后续路径：top-k memory routing

当 all-bundle routed evaluation 证明优于 best single bundle 后，再引入 top-k routing 降低推理成本。

可能的非学习 routing 信号：

1. 使用 DINOv2/global descriptors 做 query-to-reference image retrieval；
2. 最近 selected memory views；
3. 场景区域或 coverage metadata；
4. 若可用，使用 coarse pose prior；
5. 对视频式序列，使用前一帧时序连续性。

路由目标应优先保证 recall。错误的早期路由会在几何验证前丢弃正确 local map。

推荐第一版 top-k policy：

```text
retrieve top-k memory bundles by query/reference image similarity
run ACE-G only on top-k
select final pose by DSAC inlier count or reprojection score
```

在尝试 `k=1` 前，先用 `k=2` 或 `k=3`。

## 7. Metadata 契约

多 memory 系统比单 memory 设置更依赖 metadata。每个 memory bundle 至少应记录：

```text
memory_id
scene_id
mapanything_forward_id
reference_image_id
reference_pose
reference_frame_convention
world_frame_convention
T_world_ref or T_ref_world, with direction explicitly named
selected_view_ids
selected_view_poses
feature_model
feature_model_checkpoint
feature_conditioning_reference
memory_coordinate_frame
was_transformed_to_world
transform_validation_status
coverage_region / bbox / center / radius
```

如果 memory 缺少 reference-frame 和 transform metadata，就不能参与 concat 或 shared-latent 实验，只能作为独立 local bundle 使用。

## 8. 未来研究：共享模型与 per-memory latents

在 independent bundles 和 routing 验证后，可以考虑共享模型设计：

```text
shared DINOv2 / shared ACE head
memory_i -> compressed_latent_i
query -> router -> top-k latents
query + latent_i -> pose hypothesis_i
rank hypotheses geometrically
```

可能变体：

1. shared head + per-memory compressed latent cache；
2. per-memory lightweight adapter 或 LoRA-style specialization；
3. fusion 中加入 memory-id / reference-pose embedding；
4. frame-aware attention masks；
5. 在非学习 routing 建立 upper bound 后，再训练 learned query router。

这不是近期路径，因为它需要改动：

- checkpoint config；
- memory loader API；
- trainer S1/S2 loop；
- test-time compressed-memory cache；
- routing supervision/evaluation；
- metadata validation。

## 9. 验证计划

### 9.1 Smoke test

先使用两个 memory bundles：

```text
bundle_0: local memory A
bundle_1: local memory B
```

运行缩小训练：

```text
lmc_iterations = 1 or 2
small buffer if needed
same C0 config for both bundles
```

确认：

- 每个 bundle 可以独立训练；
- 每个 checkpoint 在 test 时重建自己的 memory；
- ensemble/routing 脚本可以比较 hypotheses；
- per-frame selection 记录被保存。

### 9.2 Full test

Smoke test 通过后：

1. 用 accepted ACE-G contract 训练每个 bundle；
2. 独立评估每个 bundle；
3. 评估 all-bundle routed ensemble；
4. 如果有 image retrieval metadata，评估 top-k routing；
5. 与 best single-memory baseline 和已有 scene-level ensemble baseline 比较。

### 9.3 提升标准

只有满足以下条件，才提升 multi-memory 路线：

- routed ensemble 在主指标上超过 best single bundle；
- oracle upper bound 显示明显互补性；
- 除非某个 bundle 自身也超过 baseline，否则不能出现几乎所有帧都只选一个 bundle；
- 失败能通过 routing 或 coverage diagnostics 解释；
- metadata 足以复现 bundle/reference-frame contract。

如出现以下情况，则拒绝或暂停：

- routed ensemble 不超过 best single bundle；
- oracle 不优于 best single bundle；
- routing 频繁选择几何无效假设；
- metadata 缺失导致无法诊断 frame contract；
- 收益依赖没有 frame-aware 支持的 direct concatenation。

## 10. 与 Step15 方向的关系

这个大场景多 memory 计划与 Step15 架构方向正交。

第一阶段不要把它与以下方向混合：

- 新 PointRoPE radius policies；
- query-side multi-level residual adapters；
- Layer12-anchored residual compression；
- 新 distance-bias modules。

先用 accepted C0/FGPI-style contract 验证系统级多 memory 路由。只有 routing/bundle interface 稳定后，才把 Step15 架构改进应用到每个 local bundle 内。

## 11. 当前推荐

对于需要多个 MapAnything memories 的大场景：

1. 现在不要拼接 memories；
2. 创建独立 local memory bundles；
3. 在 C0 world regression 下分别训练/评估每个 bundle；
4. 通过几何证据选择最终 pose；
5. all-bundle ensemble 有收益后，再加入 top-k retrieval routing；
6. shared-model / per-memory-latent fusion 延后到 independent-bundle baseline 验证之后。
