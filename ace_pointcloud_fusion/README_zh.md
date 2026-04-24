# ACE Point-Cloud Fusion

这是一个独立子项目，用来验证“场景点云编码器 + ACE/DINO 图像特征”的跨模态融合路线。它不改动现有 `ace_fcn_lmc`、`ace_dinov2_lmc` 分支，第一版目标是把训练链路跑通，后续再补 dense-depth 点云恢复和 PointNet/Utonia 互换实验。

## 当前流程

1. 从场景点云生成点云特征库 `point_features.pt`。
2. 训练时仍按 ACE 流程先构建 buffer，但 buffer 存的是原始 FCN 图像特征、像素位置、相机位姿/内参，以及每个采样点对应的 camera center 和 world ray。
3. 每个训练 batch 从 buffer 取样后，通过 `RayPointCrossAttentionAdaptor` 用 ray geometry 查询全局点云特征库，得到融合后的 512-d feature。
4. 融合 feature 输入 ACE 原版 head，继续使用标准 reprojection loss 做场景坐标回归。

## 主要文件

- `pointcloud_io.py`：读取 COLMAP `sparse/0/points3D.bin`、`.npy/.npz`、`.txt/.xyz` 点云，支持 `colmap_to_wai.npy` 这类 4x4 坐标变换。
- `extract_pointcloud_features.py`：调用 `/home/xwh/project/Utonia` 生成可复用的点云 feature bank。
- `adaptor.py`：跨模态 adaptor。当前实现是 ray-aware cross-attention，避免在还没有场景坐标预测时依赖 2D-3D 最近邻。
- `ace_network_pointcloud.py`：ACE FCN encoder + adaptor + ACE head。
- `trainer_pointcloud_fusion.py`：复用 ACE buffer/head 训练逻辑，扩展 buffer schema 并在 batch 内执行 fusion。
- `train_pointcloud_fusion.py`：训练入口。
- `test_pointcloud_fusion.py`：point-fusion checkpoint 专用测试入口。

## Indoor6 COLMAP 示例

以下命令默认在仓库根目录运行：

```bash
cd /home/xwh/project/ace_depth
conda activate mapanything_new
```

先提取点云特征：

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.extract_pointcloud_features \
  --input /mnt/storage/xwh/indoor6/indoor6-colmap/scene1 \
  --output /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --voxel_size 0.02 \
  --max_points 200000 \
  --utonia_root /home/xwh/project/Utonia \
  --device cuda \
  --disable_flash
```

如果训练数据使用 WAI/Indoor6 坐标而 COLMAP 点云在 COLMAP 坐标系，加入对应变换：

```bash
python -m ace_pointcloud_fusion.extract_pointcloud_features \
  --input /mnt/storage/xwh/indoor6/indoor6-colmap/scene3 \
  --transform_path /mnt/storage/xwh/indoor6/indoor6-colmap/scene3/colmap_to_wai.npy \
  --output /mnt/storage/xwh/indoor6/point_features/scene3_utonia_wai.pt
```

### ACE-FCN 训练

`--max_context_points` 是每次 fusion 最多 attend 的场景点云特征数。训练默认是 `4096`；OOM 时可降到 `2048`。

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.train_pointcloud_fusion \
  /mnt/storage/xwh/indoor6_ace/scene1 \
  scene1_point_fusion.pt \
  --image_backbone ace_fcn \
  --encoder_path /home/xwh/project/ace_depth/ace_encoder_pretrained.pt \
  --point_feature_path /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --use_lmc False \
  --training_buffer_size 2560000 \
  --buffer_batch_size 1 \
  --samples_per_image 512 \
  --batch_size 5120 \
  --epochs 24 \
  --buffer_on_cpu True \
  --eval_after_train False \
  --max_context_points 4096
```

### ACE-FCN 测试

第二个参数必须是具体 `.pt` checkpoint 文件，不是目录。可以用 `ls -lt` 找最新权重。

```bash
ls -lt /home/xwh/project/ace_depth/ace_pointcloud_fusion/04_evaluation/ace_pointcloud_fusion/*/best_*.pt | head
```

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.test_pointcloud_fusion \
  /mnt/storage/xwh/indoor6_ace/scene1 \
  /home/xwh/project/ace_depth/ace_pointcloud_fusion/04_evaluation/ace_pointcloud_fusion/20260423_113928_scene1_pcfusion/best_scene1_point_fusion.pt \
  --encoder_path /home/xwh/project/ace_depth/ace_encoder_pretrained.pt \
  --point_feature_path /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --device cuda:0 \
  --session scene1_acefcn_pointfusion \
  --image_resolution 480 \
  --max_context_points 4096
```

### DINOv2 训练

DINOv2 权重可用以下任一路径，仓库下两个文件是软链接：

- `/home/xwh/project/ace_depth/checkpoints/dinov2_vitl14_pretrain.pt`
- `/home/xwh/project/ace_depth/checkpoints/dinov2_vitl14_pretrain.pth`
- `/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth`

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.train_pointcloud_fusion \
  /mnt/storage/xwh/indoor6_ace/scene1 \
  scene1_dino_point_fusion.pt \
  --image_backbone dinov2 \
  --dinov2_path /home/xwh/project/ace_depth/checkpoints/dinov2_vitl14_pretrain.pth \
  --point_feature_path /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --use_lmc False \
  --training_buffer_size 2560000 \
  --buffer_batch_size 1 \
  --samples_per_image 512 \
  --batch_size 5120 \
  --epochs 24 \
  --buffer_on_cpu True \
  --eval_after_train False \
  --max_context_points 4096
```

### DINOv2 测试

测试脚本会根据 checkpoint 中的 `image_backbone/model_type` 自动选择 ACE-FCN 或 DINOv2。DINO checkpoint 可以显式传 `--dinov2_path` 覆盖权重路径。

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.test_pointcloud_fusion \
  /mnt/storage/xwh/indoor6_ace/scene1 \
  /home/xwh/project/ace_depth/ace_pointcloud_fusion/04_evaluation/ace_pointcloud_fusion/<RUN_DIR>/best_scene1_dino_point_fusion.pt \
  --dinov2_path /home/xwh/project/ace_depth/checkpoints/dinov2_vitl14_pretrain.pth \
  --point_feature_path /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --device cuda:0 \
  --session scene1_dino_pointfusion \
  --image_resolution 480 \
  --max_context_points 4096
```

### 快速 Smoke Run

如果只想检查流程是否能启动，不完整训练：

```bash
CUDA_VISIBLE_DEVICES=1 \
python -m ace_pointcloud_fusion.train_pointcloud_fusion \
  /mnt/storage/xwh/indoor6_ace/scene1 \
  smoke_point_fusion.pt \
  --image_backbone ace_fcn \
  --encoder_path /home/xwh/project/ace_depth/ace_encoder_pretrained.pt \
  --point_feature_path /mnt/storage/xwh/indoor6/point_features/scene1_utonia.pt \
  --use_lmc False \
  --training_buffer_size 8192 \
  --buffer_batch_size 1 \
  --samples_per_image 128 \
  --batch_size 1024 \
  --epochs 1 \
  --buffer_on_cpu True \
  --eval_after_train False \
  --use_aug False \
  --max_context_points 2048
```

## 设计约束

- 第一版训练依赖预提取点云特征，不在 ACE 训练循环内调用 Utonia，避免每次 refill buffer 都重复跑大型点云 encoder。
- 当前 adaptor 使用 camera center + ray direction 作为 query geometry，因此既能用于稀疏 COLMAP 点云，也能用于未来从 dense depth 恢复出的点云。
- 当前 checkpoint 保存 `heads_state_dict` 和 `adaptor_state_dict`，需要使用 `test_pointcloud_fusion.py` 评估，不能直接用 `test_ace.py` 或 `test_ace_lmc.py`。
- PointFusion 原型阶段强制 `buffer_batch_size=1`，避免 batched buffer 中图像特征、ray metadata 与相机参数展平顺序引入额外变量。
- `--max_context_points` 不改变点云特征文件，只控制训练/测试时临时抽取多少点参与 cross-attention。默认 `4096`，显存紧张用 `2048`。
- 如果点云来自 COLMAP，而训练/测试 pose 在另一个坐标系，必须先确认 `colmap_to_wai.npy` 或其他 4x4 transform 已正确应用，否则 fusion 会学习到错误几何关系。
