# Indoor6 多场景完整工作流

先在各场景上**提取 memory**（map-anything），再在 **ace_depth** 上跑 **baseline（vanilla）** 与 **ACE-G 训练**。当前 indoor6_ace 含 6 个场景：`scene1`, `scene2a`, `scene3`, `scene4a`, `scene5`, `scene6`。

---

## 一、前置与路径约定

| 项目 | 路径/说明 |
|------|-----------|
| **ACE 场景数据** | `/mnt/storage/xwh/indoor6_ace/<scene>/`，`<scene>` = scene1, scene2a, scene3, scene4a, scene5, scene6 |
| **Memory 提取脚本** | `map-anything/bash_scripts/ace/fps_memory.sh`，需在 **map-anything 项目根目录** 执行 |
| **提取任务入口** | `map-anything/mapanything/tasks/run_memory_extraction.py`（由 fps_memory.sh 调用） |
| **Memory 输出根目录** | `map-anything-experiments/memory_extract/` |
| **单场景输出结构** | `memory_extract/<SCENE_TRAIN>/<N_VIEWS>views/<timestamp>/Indoor6_<SCENE_TRAIN>_pooled_GT.pt` |
| **ACE-G 训练** | 在 **ace_depth** 项目根目录执行 `train_ace_dinov2_lmc.py` |

**场景名对应关系**（ACE 目录名 → 提取用 SCENE_TRAIN）：

- ACE `scene1` → `scene1_train` / `scene1_test`
- ACE `scene2a` → `scene2a_train` / `scene2a_test`
- ACE `scene3` → `scene3_train` / `scene3_test`
- ACE `scene4a` → `scene4a_train` / `scene4a_test`
- ACE `scene5` → `scene5_train` / `scene5_test`
- ACE `scene6` → `scene6_train` / `scene6_test`

---

## 二、Phase 1：提取 Memory（逐场景）

在 **map-anything** 根目录执行。每场景一条，按需改 `SCENE_TRAIN`/`SCENE_TEST` 和 `GPU_ID`。

**环境与默认**：`fps_memory.sh` 内已设 `DATASET=indoor6_dataset`、`N_VIEWS=40`，Indoor6 会用 `PATCH_DEPTH_SAMPLING=nearest_valid`、`DEPTH_VALID_MAX=100.0`。

### 2.1 单条命令模板

```bash
cd /home/xwh/project/map-anything

DATASET=indoor6_dataset \
SCENE_TRAIN=<scene>_train \
SCENE_TEST=<scene>_test \
N_VIEWS=40 \
GPU_ID=2 \
bash bash_scripts/ace/fps_memory.sh
```

将 `<scene>` 替换为：`scene1`, `scene2a`, `scene3`, `scene4a`, `scene5`, `scene6`。

### 2.2 各场景逐条执行（复制即用）

```bash
cd /home/xwh/project/map-anything

# scene1（若已有 memory 可跳过）
DATASET=indoor6_dataset SCENE_TRAIN=scene1_train SCENE_TEST=scene1_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh

# scene2a
DATASET=indoor6_dataset SCENE_TRAIN=scene2a_train SCENE_TEST=scene2a_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh

# scene3
DATASET=indoor6_dataset SCENE_TRAIN=scene3_train SCENE_TEST=scene3_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh

# scene4a
DATASET=indoor6_dataset SCENE_TRAIN=scene4a_train SCENE_TEST=scene4a_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh

# scene5
DATASET=indoor6_dataset SCENE_TRAIN=scene5_train SCENE_TEST=scene5_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh

# scene6
DATASET=indoor6_dataset SCENE_TRAIN=scene6_train SCENE_TEST=scene6_test N_VIEWS=40 GPU_ID=2 bash bash_scripts/ace/fps_memory.sh
```

提取完成后，每个场景会得到目录：

`/home/xwh/project/map-anything-experiments/memory_extract/<SCENE_TRAIN>/40views/<timestamp>/`

其下文件：`Indoor6_<SCENE_TRAIN>_pooled_GT.pt`（以及同名的 .ply、depth_validation 等）。**记下各场景的 `<timestamp>`，或用“该目录下最新一次运行”的 .pt 作为训练用 memory 路径。**

---

## 三、Phase 2：训练（Baseline + ACE-G）

在 **ace_depth** 项目根目录执行。每条命令需把 **memory 路径** 换成该场景刚提取出的 `.pt`（或该场景 40views 下最新一次运行的 .pt）。

### 3.1 查找各场景最新 memory .pt（可选）

```bash
MEM_ROOT=/home/xwh/project/map-anything-experiments/memory_extract
for s in scene1_train scene2a_train scene3_train scene4a_train scene5_train scene6_train; do
  latest=$(ls -td "$MEM_ROOT/$s/40views"/*/Indoor6_${s}_pooled_GT.pt 2>/dev/null | head -1)
  echo "$s -> $latest"
done
```

把输出里的路径填到下面命令的 `--memory_path` 中。

### 3.2 Baseline（vanilla，无 LMC）

与 scene1 已有 baseline 一致：28 轮迭代、2.56M buffer、FP32、hierarchical 输出。仅改场景与 GPU。

```bash
cd /home/xwh/project/ace_depth

# 将 <SCENE> 换为 scene1, scene2a, scene3, scene4a, scene5, scene6
# 将 <CUDA> 换为实际 GPU，如 cuda:2
python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/indoor6_ace/<SCENE> \
    <SCENE>_vanilla_it28.pt \
    --experiment_root output \
    --device <CUDA> \
    --use_lmc False \
    --apply_baseline_contract False \
    --vanilla_iterations 28 \
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

### 3.3 ACE-G（当前最优配置）

与 scene1/chess/heads 的 ACE-G 一致：global、64 tokens、28 iters、full refill、FP16 等。仅改场景、`--memory_path` 和 GPU。

```bash
cd /home/xwh/project/ace_depth

# <SCENE> = scene1, scene2a, scene3, scene4a, scene5, scene6
# <MEMORY_PT> = 该场景 Phase 1 得到的 .pt 完整路径（见 3.1）
# <CUDA> = 如 cuda:3
python train_ace_dinov2_lmc.py \
    /mnt/storage/xwh/indoor6_ace/<SCENE> \
    <SCENE>_aceg_full_refill.pt \
    --device <CUDA> \
    --run_name <SCENE>_aceg_full_refill \
    --use_lmc True \
    --memory_path <MEMORY_PT> \
    --dinov2_path /mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth \
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

显存不足时可将 `--batch_size 5120` 改为 `--batch_size 512`。

---

## 四、一键脚本（提取 + 生成训练命令）

在 **ace_depth** 项目根目录执行（脚本会自行 `cd` 到 map-anything 做提取）：

```bash
cd /home/xwh/project/ace_depth

# 1) 仅提取 6 个场景的 memory（使用 GPU 0）
bash scripts/indoor6_full_workflow.sh extract

# 2) 提取完成后，生成各场景的 baseline 与 ACE-G 训练命令（自动查找最新 .pt）
bash scripts/indoor6_full_workflow.sh train-cmds
```

或一次执行两步：

```bash
bash scripts/indoor6_full_workflow.sh extract train-cmds
```

**可选环境变量**（在命令前设置）：

| 变量 | 含义 | 默认 |
|------|------|------|
| `SCENES` | 空格分隔场景名 | `scene1 scene2a scene3 scene4a scene5 scene6` |
| `GPU_EXTRACT` | 提取 memory 的 GPU 编号 | `0` |
| `GPU_BASELINE` | 生成命令里 baseline 的 `--device cuda:X` | `2` |
| `GPU_ACEG` | 生成命令里 ACE-G 的 `--device cuda:X` | `3` |
| `N_VIEWS` | 提取视角数 | `40` |

示例：只处理 scene1 和 scene2a，且用 1 号卡提取：

```bash
SCENES="scene1 scene2a" GPU_EXTRACT=1 bash scripts/indoor6_full_workflow.sh extract train-cmds
```

脚本会：

1. **Phase 1 (extract)**：在 map-anything 下对每个 `SCENE_TRAIN=<scene>_train` 执行 `fps_memory.sh`。
2. **Phase 2 (train-cmds)**：在 `memory_extract/<scene>_train/40views/` 下找**最新 timestamp** 目录中的 `Indoor6_<scene>_train_pooled_GT.pt`，为该场景打印 baseline 与 ACE-G 的完整命令，可直接复制到终端或不同 GPU 上并行跑。

使用前请确认：

- `map-anything` 与 `ace_depth` 均在 `/home/xwh/project/` 下（或通过 `MAPANYTHING_ROOT` 指定）；
- `map-anything-experiments/memory_extract` 可写；
- 各场景 WAI 数据已就绪（与 `indoor6_dataset` 使用的 ROOT / scene 名一致）。

---

## 五、参考：scene1 已有结果

- **Baseline**：`output/indoor6_ace/scene1/dino_ace_baseline/20260314_183743_scene1_vanilla_buf2.6M_K64_it28_bs5120_onecycle/`
- **ACE-G**：`output/indoor6_ace/scene1/dino_ace_lmc_ace_g/scene1_aceg_full_refill/`  
两者配置与上文 3.2、3.3 一致，仅场景与 memory 路径不同。
