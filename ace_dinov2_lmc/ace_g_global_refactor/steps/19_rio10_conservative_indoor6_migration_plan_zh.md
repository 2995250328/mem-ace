# Step 19：从 Indoor6 最佳 ACE-G Baseline 到 RIO10 的保守迁移方案

## 1. 目的

本文档定义从当前最佳 Indoor6 ACE-G baseline 迁移到 RIO10 的最保守方案。

目标不是引入新架构，而是尽可能保留已经接受的 Indoor6 行为，只改变数据域，并用最低风险的方式使用 RIO10 稀疏深度。

迁移原则是：

```text
Indoor6 best ACE-G baseline
+ RIO10 数据域
+ 只使用 sparse-depth guided sampling
+ 匹配的 MapAnything memory 构建 contract
```

本文档有一个必须先读的前置文档，以及两个可选补充参考。

必须先读：

- `steps/18_rio10_wai_training_fix_log.md` —— 这是必读文档，因为它建立了修正后的 RIO10 backend chain、ACE-format training 路径、resize fix、sparse-depth attachment 行为，以及 sparse-depth coverage 有限这一关键事实，而这正是“只做 guided sampling、不做 auxiliary supervision”判断的依据。

按需再读：

- `EXPERIMENT_REFERENCE.md`：确认已接受的 Indoor6 baseline；
- `steps/16_large_scene_multi_memory_reference_frame_plan.md`：如果后续考虑 multi-memory 扩展。

## 2. 需要保留的 Baseline

当前接受的 Indoor6 reference 是 FGPI-style 4090 run：

```text
4090_forceglobal_s1_periter
```

重要的不是只看指标，而是它代表一个稳定的 true-global ACE-G baseline，并且使用 per-iteration S1 语义。

迁移到 RIO10 时，除非后续做单变量 ablation，否则下面这些设置应完整保留：

```text
--use_lmc True
--lmc_flow ace_g
--lmc_mode global
--lmc_profile legacy

--s1_loss_step_mode per_iter

--lmc_key_slice_idx 2
--lmc_feature_hierarchy_mode selected_key_concat_value
--lmc_fusion_geometry_mode value_only_raw
```

语义要求是：

```text
requested_lmc_mode = global
effective_lmc_mode = global
lmc_auto_mode_by_visibility = False
```

也就是说，这次迁移必须保持 true global。不能通过 visibility fallback 或 scene heuristic 偷偷变成 effective-local。

## 3. 稀疏深度允许做什么

稀疏深度第一轮只应作为采样先验使用。

允许：

- 将 sparse depth / sparse coordinate mask 接到训练 dataset；
- 用有效 sparse-coordinate 位置引导 buffer 采样；
- 对没有 sparse depth 的 frame 保留 empty mask；
- 比较同一个模型在有/无 guided sampling 下的表现。

第一轮不允许：

- sparse-depth auxiliary supervision；
- auxiliary reference-coordinate loss；
- depth consistency loss；
- 由 sparse depth 派生的 latent 或 fusion loss；
- 任何把 sparse depth 变成监督目标的额外 loss。

训练侧的含义应该是：

```text
sparse depth -> 告诉 buffer 哪些 patch 位置更有几何支撑
sparse depth -> 不通过额外 objective 监督网络
```

因此最小配置 contract 是：

```text
--buffer_sample_valid_coords True
--c1_aux_ref_loss_weight 0.0
```

如果继续使用已有 valid-coordinate sampling 参数，它们也只应被理解成采样策略旋钮：

```text
--buffer_valid_coord_sample_ratio <从受控 recipe 保留>
--buffer_valid_coord_neighbor_radius <从受控 recipe 保留>
--buffer_valid_coord_neighbor_mode <从受控 recipe 保留>
```

不要把这些参数解释成监督参数。

## 4. RIO10 数据 Backend 拆分

RIO10 应继续保持已经建立的 backend 拆分：

```text
memory extraction: WAI / MapAnything path
ACE-G training:    ACE backend
```

原因：

- MapAnything memory extraction 需要 WAI loader 和 MapAnything forward path；
- 如果希望和 Indoor6 baseline 训练语义对齐，ACE-G training 最接近的方式是使用 ACE-format 的 `rgb`、`poses`、`calibration` 目录；
- Step18 已记录 RIO10 ACE-format scene 位置：

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

稀疏深度 root 仍作为外部 metadata，在 ACE-backend training 时附着：

```text
--c1_aux_depth_root /data/xwh/RIO10_sparse_depth
--c1_aux_depth_kind sparse_depth
```

Step18 记录了：

```text
matched sparse depth = 4356 / 4380
missing sparse depth = 24
valid sparse coord patches in frame-000000 = 51
```

这支持 guided sampling，但也说明 patch 级 sparse-depth 覆盖有限。因此第一轮更不应该把 sparse depth 当成 auxiliary supervision target。

## 5. 匹配的 Memory 构建 Contract

RIO10 memory 应遵循当前 accepted ACE-G 路径的保守 global MapAnything contract。

必须使用的 memory 构建选择：

```text
dataset_loader = wai
dataset_type = rio10
wai_view_mode = anchor_support / asb
MapAnything forward = single forward only
memory regime = single global memory
```

关键约束是 single-forward consistency。

MapAnything 产生的是 reference-conditioned geometry 和 features。如果多个 group 分别 forward，每个 group 可能有不同的隐式参考帧。把这些输出拼成一个 memory，会混合不兼容的坐标和 reference-conditioned features。

因此这次迁移中：

- 不要把一个 memory 拆成多个 MapAnything forward；
- 不要拼接多个单独提取的 memories；
- 还不要引入 multi-memory routing；
- 不要在第一轮实验里让一个 shared model 同时吃多个 reference-conditioned memories。

先使用一个 reference-consistent global memory。Multi-memory routing 属于后续 Step16 方向，不属于这次保守 RIO10 迁移。

## 6. Sparse-Depth Memory Sampling：使用 `nearest_valid`

针对 RIO10 sparse depth，唯一推荐的 memory-extraction 修正是：

```text
--patch_depth_sampling nearest_valid
```

而不是：

```text
--patch_depth_sampling nearest
```

理由：

- RIO10 sparse depth 很稀疏，patch 中可能有空洞或无效位置；
- `nearest` 可能选到最近但无效或不代表该 patch 的像素；
- `nearest_valid` 能让 geometry construction 绑定到 patch 内有效 sparse depth；
- 这改善的是 memory geometry quality，不改变 ACE-G 模型结构或训练目标。

这属于数据 contract 修复，不是新的模型机制。

因此 memory extraction contract 应该是：

```text
WAI / MapAnything extraction
+ ASB view selection
+ single forward
+ sparse-depth-aware patch sampling with nearest_valid
+ output in the existing pooled/BSE-compatible memory format
```

## 7. 最小实验矩阵

第一轮 RIO10 迁移只做两组 run。

### Run A：纯 Indoor6 Baseline 迁移

目的：测量把 Indoor6 best baseline 移到 RIO10 后，在不使用 sparse-depth guided sampling 的情况下表现如何。

保持：

```text
true global
per_iter
legacy profile
selected layer12 key
selected_key_concat_value
value_only_raw
same memory construction contract
```

关闭：

```text
buffer_sample_valid_coords = False
c1_aux_ref_loss_weight = 0.0
```

### Run B：Baseline + Sparse-Depth Guided Sampling

目的：隔离 sparse depth 作为采样先验的效果。

与 Run A 完全相同，只改：

```text
buffer_sample_valid_coords = True
c1_aux_ref_loss_weight = 0.0
```

A 和 B 的唯一区别应该是：valid sparse-coordinate locations 是否影响 buffer sampling。

如果 Run B 优于 Run A，解释应当是：

```text
sparse depth 改善了 sampled training locations
```

而不是：

```text
sparse depth 增加了第二个监督任务
```

## 8. 第一轮迁移中不能改变的变量

不要改架构：

- 不上 PointRoPE；
- 不上 Cascading Internal Fusion Assembly；
- 不上 query-side DPT/FPN adapter；
- 不改 multi-level compression；
- 不改 view-token 或 `all_scale_tokens` conditioning。

不要改 memory regime：

- 不做 multi-memory concatenation；
- 不做 routing；
- 不把 clustered memory package 直接传给标准 trainer；
- 不混合不同 reference 的 memory。

不要改 loss 语义：

- 不做 sparse-depth auxiliary loss；
- 不做 C1 auxiliary reference supervision；
- 不加新的 geometry consistency loss。

不要改 evaluation 语义：

- 如果要匹配 accepted checkpoint selection rule，就保持 `best_metric=pct5`；
- 汇报 final results 时保持和 Indoor6 reference 一致的 post-train evaluation protocol；
- 不用 incomplete run 的最新可见 eval 文件来排名；
- result summary 中保留确切 source files 和 iteration provenance。

## 9. 推荐执行顺序

1. 构建或确认 RIO10 ACE-format scene：

```text
/data/xwh/RIO10_ace/scene01_seq01_01
```

2. 用 WAI / MapAnything 提取 RIO10 memory：

```text
dataset_loader = wai
dataset_type = rio10
wai_view_mode = anchor_support
patch_depth_sampling = nearest_valid
single MapAnything forward
```

3. 使用 ACE backend 训练 Run A，不开启 guided sampling。

4. 使用 ACE backend 训练 Run B，只开启 guided sampling。

5. 用相同 checkpoint-selection 和 post-train evaluation protocol 评测二者。

6. 在尝试任何架构扩展前，先比较 A vs B。

## 10. 为什么这是保守迁移

这套 recipe 改动的变量最少：

```text
changed: dataset domain
changed: optional sampling prior in Run B
fixed: ACE-G architecture
fixed: global/per_iter baseline semantics
fixed: memory reference-frame contract
fixed: loss semantics
fixed: evaluation semantics
```

这很重要，因为之前 RIO10 sparse-depth full run 已经显示出弱提升和多个混杂因素：

- patch-level sparse depth 很稀疏；
- 部分训练 frame 没有 sparse-depth match；
- memory extraction 使用了 `nearest`，但 sparse-depth 场景更推荐 `nearest_valid`；
- run 使用了旧的 RIO10 preset；
- S1 似乎较早饱和；
- 结果本身不完整，目前 best 来自中间 iteration。

最安全的回应不是添加更多机制，而是恢复最强 Indoor6 contract，修复 sparse-depth memory sampling contract，并把 guided sampling 隔离成一个受控变量。

## 11. 最终 Recipe

推荐的 RIO10 迁移方案是：

```text
Indoor6 FGPI-style true-global ACE-G baseline
+ RIO10 ACE-format training data
+ WAI/MapAnything ASB single-forward global memory
+ patch_depth_sampling = nearest_valid
+ sparse depth attached only for valid-coordinate guided sampling
+ c1_aux_ref_loss_weight = 0.0
```

在测试 PointRoPE、CIFA、multi-level fusion、view tokens 或 multi-memory routing 之前，这应该作为 RIO10 stabilization 的默认路径。
