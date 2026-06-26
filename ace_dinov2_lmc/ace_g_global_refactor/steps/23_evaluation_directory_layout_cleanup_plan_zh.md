# 23. 评估目录治理与后续实验 Run Root 规范

日期：2026-06-26

## 背景

当前 `/data/xwh/ace_dinov2_lmc/04_evaluation` 下历史结果按实验主题平铺，混合了 Wayspots、Indoor6、Cambridge、DSAC deterministic 验证、GPU 并发探针、baseline restore、paper result、memory extraction 等目录。继续平铺会导致：

- baseline / stage1 / stage2 / fusion / diagnostics 互相混杂；
- 同一个数据集的结果难以完整检索；
- 新实验命名依赖临时习惯，后续复盘容易漏掉结果；
- 路径被训练脚本、summary、checkpoint 引用后，不适合随意移动。

## 已执行的非破坏性改动

新增本地 Codex skill：

```text
.codex_skills/lmc-evaluation-directory-layout/SKILL.md
```

并将现有：

```text
.codex_skills/lmc-experiment-standard-workflow/SKILL.md
```

中的新实验 `RUN_ROOT` 规则改为引用该目录规范。

新增辅助脚本：

```text
.codex_skills/lmc-evaluation-directory-layout/scripts/make_lmc_eval_run_root.py
.codex_skills/lmc-evaluation-directory-layout/scripts/inventory_lmc_eval_dirs.py
```

校验已通过：

```text
python -m py_compile ...
quick_validate.py .../lmc-evaluation-directory-layout
```

## 后续新实验强制目录格式

所有新实验应使用：

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/<dataset>/<track>/<method>/<YYYYMMDD>_<scope>_<protocol>_<gpu_tag>/
```

其中：

- dataset: `wayspots`, `cambridge`, `indoor6`, `shared`
- track: `baseline`, `stage1`, `stage2`, `fusion`, `training_efficiency`, `reproduction`, `diagnostics`, `paper_results`, `memory`, `scratch`
- method: 稳定方法名，例如 `single`, `pmrf_v3`, `concat_glace`, `glace_official`, `stage2_r2_fusion`
- scope: 场景集合，例如 `all`, `kings`, `court_stmary`, `bears_sq`
- protocol: 训练/评估协议，例如 `it10_buf10m_final12m_h256`, `official_30k_h64`
- gpu_tag: `gpu0`, `gpu01`, `gpu23`, `cpu`, `nogpu`

示例：

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge/stage2/concat_glace/20260626_kings_it10_buf10m_final12m_h256_gpu0
/data/xwh/ace_dinov2_lmc/04_evaluation/cambridge/baseline/glace_official/20260626_all_official_30k_h64_gpu1
/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/fusion/pmrf_v3/20260623_bears_sq_1iter_h256_gpu01
```

## 创建新 run root 的标准命令

```bash
cd /home/xwh/project/ace_depth
python ace_dinov2_lmc/.codex_skills/lmc-evaluation-directory-layout/scripts/make_lmc_eval_run_root.py \
  --dataset cambridge \
  --track stage2 \
  --method concat_glace \
  --scope kings \
  --protocol it10_buf10m_final12m_h256 \
  --gpus 0 \
  --scenes Cambridge_KingsCollege \
  --create
```

脚本会创建：

```text
manifest.json
logs/
summaries/
artifacts/
scripts/
```

## 历史目录整理原则

当前不要直接移动历史目录，原因：

1. 有些目录仍可能被后续 summary / checkpoint / stage2 local checkpoint 引用；
2. 当前 GPU2/3 仍有实验在跑；
3. 大规模 move 容易破坏已有日志中的绝对路径。

正确流程：

1. 先生成 inventory：

   ```bash
   cd /home/xwh/project/ace_depth
   python ace_dinov2_lmc/.codex_skills/lmc-evaluation-directory-layout/scripts/inventory_lmc_eval_dirs.py \
     --base /data/xwh/ace_dinov2_lmc/04_evaluation \
     --output /tmp/lmc_eval_inventory_20260626.tsv
   ```

2. 人工检查 dataset/track 分类。
3. 等没有活跃训练依赖旧路径后，再分组迁移。
4. 每次只迁移一个 dataset/track 组。
5. 留下迁移表：

   ```text
   /data/xwh/ace_dinov2_lmc/04_evaluation/_migration_YYYYMMDD.tsv
   ```

   字段至少包括 old_path、new_path、action、reason、checked_by。

## 建议的迁移目标

历史平铺目录建议进入对应的：

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/<dataset>/<track>/legacy_flat/<old_name>
```

无法判断的数据先进入：

```text
/data/xwh/ace_dinov2_lmc/04_evaluation/shared/scratch/legacy_flat/<old_name>
```

但只有在确认不被训练引用后再移动。

## 当前状态

- 新目录规范和 skill 已完成。
- 历史目录 inventory 已可生成。
- 尚未移动 `/data` 下任何历史结果。
- 后续所有新训练脚本应优先使用 canonical run root。
