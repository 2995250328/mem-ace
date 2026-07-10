# AAAI 2027 OpenReview 取号字段草稿

本文档与 [`aceg_method_framework_config_checked_cn.md`](../paper_experiment_matrix/03_figures/method_architecture/aceg_method_framework_config_checked_cn.md) 对齐。以下版本刻意只陈述已经收敛的 Mem-ACE 方法事实，不把 backbone 分支、开发期 global-to-local fallback、未完成的可视化结论或未核验的结果数字提前写进取号内容。

## 建议直接填写的字段

### Title

```text
Mem-ACE: Compact Scene Memory for Accelerated Coordinate Encoding
```

理由：标题展开 ACE，突出固定尺寸的 scene-memory 表征，且不把 GLACE、DINOv2、可见性路由或未完成实验绑定为方法定义。取号和最终投稿应尽量沿用此标题；除非方法定位发生实质变化，不建议在截稿前大幅改名。

### TL;DR

```text
Mem-ACE equips Accelerated Coordinate Encoding with a compact set of geometry-anchored scene tokens that can be reused across query images for scene coordinate regression.
```

### Abstract

```text
Scene coordinate regression (SCR) relocalizes a camera from predicted 2D--3D correspondences, but its scene-specific knowledge is often absorbed into the parameters of a scene regressor. We introduce Mem-ACE, a compact scene-memory extension of Accelerated Coordinate Encoding. From posed mapping images and an SfM reconstruction, Mem-ACE aligns visual features with 3D points, pools them into a scene memory bank, and compresses the bank into a fixed set of anchored scene tokens. Each token combines learned content with a 3D anchor. A single memory-fusion block conditions query features on token content and anchor geometry expressed relative to the scene center, while a standard SCR head predicts dense scene coordinates in the same coordinate reference. Mem-ACE first geometrically aligns the compressor, fusion module, and coordinate head, then caches the scene tokens while adapting the downstream predictor. When verified SfM tracks are available, they provide cross-view reprojection constraints during this adaptation. We evaluate the resulting representation across indoor and outdoor visual relocalization settings.
```

这个摘要不列数据集名称、具体数值或 “state of the art” 结论。最终完整投稿时，只在最后补入一条由 final matrix 聚合结果直接支持的结果句；其余方法描述应维持不变。

## 可选标题

若最终想进一步强调 anchor geometry，而不是压缩性，可在投稿前二选一；不要在多个地方混用。

1. `Mem-ACE: Compact Scene Memory for Accelerated Coordinate Encoding` （当前推荐）
2. `Mem-ACE: Anchored Scene Memory for Accelerated Coordinate Encoding`
3. `Mem-ACE: Compact Scene Memory for Scene Coordinate Regression`

第 3 个标题不展开 ACE，因此不作为当前取号首选。

## 与方法事实的逐项核对

| 取号表述 | 对应的稳定方法事实 | 不应扩展成的说法 |
|---|---|---|
| compact scene memory | pooled point-feature memory 被压缩为固定的 64 个 tokens | 所有分支恒为 `64 x 1024`，或已经比某一方法快多少。 |
| anchored scene tokens | 每个 token 是 `(z_k,p_k)`；`p_k` 是采样的 3D anchor | `p_k` 是最终预测坐标或有独立坐标监督。 |
| anchor geometry relative to scene center | `value_only_raw` 把 `PE(p_k-c)` 注入 fusion value；head mean 与同一中心对齐 | geometry 同时进入 key，或方法直接回归相对坐标 residual。 |
| single memory-fusion block | query 对 tokens 进行一次 cross-attention read | GLACE bridge、multi-read 或其他历史 fusion 分支是主方法。 |
| caches scene tokens while adapting predictor | compressor 在 Cached Memory Adaptation 中冻结、tokens 被缓存 | compressor 在该步骤继续更新，或这是仅执行一次的静态 two-stage 流程。 |
| verified SfM tracks provide constraints when available | final ACE-FCN/GLACE adaptation protocol 使用 cross-view track reprojection | 所有 backbone、所有阶段均使用 multi-view loss。 |
| indoor and outdoor settings | 当前最终矩阵覆盖 Indoor6、Wayspots、Cambridge | 已在所有数据集上稳定优于所有基线。 |

## 取号版本禁止出现的内容

- `S1`、`S2-G`、`stage2_g`、bridge、FiLM、routing 等代码或历史实验名称；
- DINOv2、MapAnything、ACE-FCN、GLACE 的具体连接方式；
- `$64\times1024$`、`4096\times768` 或任何未经逐项核验的容量比较；
- “latent anchors are coordinate predictions”、“learned visibility policy”、“universal improvement”；
- 未完成的多视角一致性、点云多峰性、稀疏区域精度等分析结论；
- 尚未由最终表格和重复实验支撑的数值、SOTA 或显著性措辞。

固定 global compression 是论文协议。开发期自动 fallback 不会写入取号内容、正文、附录、图注或实验表。

## 最终摘要的结果句位置

最终实验完成后，可把下列句子替换当前摘要最后一句；方括号内容必须从 metric-wise aggregation 和统计分析中填入，不能凭印象写。

```text
Across Indoor6, Wayspots, and Cambridge Landmarks, Mem-ACE improves [metric(s)] over [clearly named comparison setting(s)] while representing each scene with [verified token size / measured resource figure].
```

只有当数据集、比较设置、评测 seed 和资源数字全部固定后，才将这一句写入最终 abstract。若结果是分支依赖的，应改成限定性表述，而不要把局部结果泛化为所有 SCR systems。

## OpenReview 表单核对

根据当前可见表单，至少准备并核对：Title、Authors、Abstract、Primary Topic、Country of Institutions、匿名主 PDF、AAAI reproducibility checklist、reciprocal reviewer nomination/confirmation、profile policy agreement、license/readers/signatures。实际必填项以当日 OpenReview 页面中的红色 `*` 和提交校验为准。

- Primary topic：优先选择 `Computer Vision`；若下拉菜单提供，可添加 `3D Vision`、`Robotics`、`Machine Learning` 或 `Deep Learning` 作为 secondary topics。
- Reciprocal reviewer：只能提名确实满足 AAAI 资格且可完成评审的作者；否则使用系统允许的“不符合资格”声明。不要猜测。
- 作者资料：在提交前确认每位作者的 institution、position、institutional email 和 DBLP URL 已在 OpenReview profile 中完整填写。
- 上传：不要假设系统一定支持 metadata-only 取号。若表单要求 PDF 或 reproducibility checklist，先上传可编译、匿名且与上述标题/摘要一致的版本，再点击 `Submit`。

成功取号的判据是提交后出现 submission detail page 和 submission number/ID，且 Recent Activity 或确认邮件显示创建成功；仅停留在可编辑表单不是完成状态。

## 引用规则

OpenReview 的 Title、TL;DR 和 Abstract 不需要塞入文献引用。完整论文中的 ACE、ACE-G、GLACE、NeuMap 等引文必须分别从官方论文或权威索引获取并核验 BibTeX；不要从标题缩写、记忆或本草稿反推参考文献。
