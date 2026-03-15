# LMC scene_center 与 ACE mean_cam_center 偏差分析

## 1. 偏差现象

训练 ACE-G 时出现：

```text
ValueError: [LMC] Memory center mismatch is too large (3.1748 m > 3.0000 m).
Likely using memory from a different scene or coordinate system.
```

即 **memory 文件里的 `scene_center`** 与 **ACE 数据集的 `mean_cam_center`** 的欧氏距离为 3.17 m，超过默认阈值 3.0 m。

---

## 2. 两边的“中心”分别怎么来的

### 2.1 fps_memory（map-anything）里的 scene_center

**代码位置**：`map-anything/mapanything/tasks/run_memory_extraction.py`

- 对选中的 N 帧，用 **深度 + 内参 + 位姿** 反投影得到世界坐标系下的点云 `pts_world`，逐帧拼成 `merged_pts`。
- **scene_center 在 SOR 与 voxel pooling 之前** 计算（约 1566–1574 行）：

```python
merged_pts = torch.cat(all_points, dim=0)   # 所有有效深度反投影后的世界坐标点
scene_center = torch.mean(merged_pts, dim=0).cpu()
```

也就是说：

- **fps_memory 的 scene_center = 反投影点云的几何中心（质心）**
- 参与平均的是“所有有效深度点”，不是相机位置；
- 点云分布受以下影响：
  - 哪些视角被选（FPS 选出的 N_VIEWS 帧）
  - 每帧里哪些像素深度有效（Indoor6 稀疏深度下通常只有很少比例有效）
  - 有效点更集中在某些区域（如墙面、近处物体）时，质心会往这些区域偏

### 2.2 ACE 里的 mean_cam_center

**代码位置**：`ace_depth/dataset_wai_dinov2.py`（及同类 dataset）

- 对当前数据集用到的**所有帧**的位姿，取**相机位置**的平均：

```python
def _compute_mean_camera_center(self):
    mean_cam_center = torch.zeros((3,), dtype=torch.float32)
    for idx in self.valid_file_indices:
        frame_name = self.frame_names[int(idx)]
        pose = self._get_pose_from_meta(frame_name)  # 4x4 c2w
        mean_cam_center += pose[0:3, 3]              # 相机在世界系下的位置
    mean_cam_center /= len(self)
    return mean_cam_center
```

也就是说：

- **ACE 的 mean_cam_center = 相机轨迹的几何中心**
- 每个视角贡献一个点（相机光心），求平均；
- 与“场景几何长什么样、哪里深度多”无关，只和“相机站在哪”有关。

---

## 3. 为什么会有 ~3.17 m 的偏差？

本质原因：**两边定义的不是同一个量**。

| 项目       | fps_memory (memory 文件)     | ACE (训练数据集)        |
|------------|-----------------------------|---------------------------|
| 含义       | 反投影点云的质心             | 相机位置的质心            |
| 依赖       | 深度覆盖、有效像素分布        | 仅位姿（相机轨迹）        |
| 点数/来源  | 大量 3D 点（每像素/每 patch）| 每帧 1 个点，共 N 帧      |

因此：

1. **定义不同**
   - 点云质心 ≠ 相机质心。相机可以围着一块区域转，相机质心在“轨迹中心”；点云质心在“被观测到的几何”的中心，会偏向深度多、观测密的区域（例如某面墙、某块家具）。
2. **Indoor6 稀疏深度放大了差异**
   - 有效深度像素很少（约 0.36%），有效点往往集中在少数区域（近处、平面等），点云质心容易明显偏离相机轨迹中心，几米的偏差很常见。
3. **视角集合可能不完全一致**
   - 若 memory 用的帧集合（FPS 选出的 40 帧）与 ACE 训练用的帧集合不完全一致，两边平均的“中心”也会略有不同；但主要差异仍来自“点云质心 vs 相机质心”。

坐标系上，两边都使用 **c2w 位姿** 和 `pts_world = pts_cam @ R.T + t`，世界系一致；偏差主要来自**用谁做平均**（点 vs 相机），而不是来自不同的坐标定义。

---

## 4. fps_memory 里“中心”的选择方式小结

- **没有**单独“选一个中心位置”的步骤；
- 中心就是 **整段流程里得到的反投影点云的均值**：
  1. 按 FPS 选出 N_VIEWS 帧；
  2. 每帧用深度+内参+位姿反投影得到 `pts_world`，只保留有效深度对应的点；
  3. 所有帧的有效点拼成 `merged_pts`；
  4. `scene_center = merged_pts.mean(dim=0)`（在 SOR/voxel 之前）。

因此，fps_memory 的 scene_center 完全由“当前深度覆盖和有效点分布”决定，和 ACE 的 mean_cam_center（相机轨迹中心）没有必然一致性。

---

## 5. 实际使用建议

- 这是**预期内的几何差异**，不是配错场景或坐标系错误；用 `--lmc_scene_center_max_distance 4.0`（或 5.0）放宽阈值即可正常训练。
- 若希望两边“中心”更一致，可以考虑（需改代码）：
  - 在 map-anything 保存 memory 时，用 **all_poses 的相机位置均值** 作为 scene_center 写入，与 ACE 的 mean_cam_center 定义一致；或
  - 在 ACE 侧用 memory 里已有的 scene_center 作为回归的参考中心，而不是用 dataset 的 mean_cam_center（需核对 head 的 mean 与 loss 是否依赖 mean_cam_center）。

当前推荐做法仍是：**放宽 `lmc_scene_center_max_distance`**，在 Indoor6 多场景下统一使用（例如 4.0 或 5.0）。
