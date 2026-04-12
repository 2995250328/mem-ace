# ACE / WAI 数据加载路径对比说明

## 1. 总体结构概览

本工程中，相机定位 / 3D memory 提取共有两条主要数据加载路径：

- **ACE 路径（兼容 ACE / CamLocDatasetDINOv2 数据）**
  - 入口：`DATASET_LOADER=ace`
  - 数据源：原始 7Scenes / ACE 风格目录（`rgb/`, `poses/`, `calibration/`, `depth/` 等）
  - 核心组件：
    - `ace_depth/dataset_dinov2.py` 中的 `CamLocDatasetDINOv2`
    - `ace_depth/ace_dinov2_lmc/memory_extraction/run_memory_extraction.py` 中的：
      - `load_dataset(..., dataset_loader="ace")`
      - `ACEDatasetWithDepth`
      - `convert_ace_tuple_to_dict(...)`（tuple -> dict 适配）

- **WAI 路径（MapAnything 官方 WAI 数据流）**
  - 入口：`DATASET_LOADER=wai`
  - 数据源：MapAnything 预处理好的 WAI 格式 7Scenes 数据（包含 `scene_meta.json`、`depth_along_ray` 等）
  - 核心组件：
    - `map-anything/mapanything/datasets/wai/seven_scenes.py` 中的 `SevenScenesWAI`
    - `ace_depth/ace_dinov2_lmc/memory_extraction/run_memory_extraction.py` 中的 `load_dataset(..., dataset_loader="wai")`
    - `map-anything/mapanything/utils/wai/core.py` 中的 `load_data(...)`、`load_frame(...)`

两条路径最终都被适配为 MapAnything 模型一致的输入字典结构：  
`img` / `depth_z` / `intrinsics` / `camera_poses` 等，然后进入统一的 memory 提取 / PoseEval pipeline。

## 2. 顶层入口：`run_memory_extraction.py` 中的 loader 分支

文件：`ace_depth/ace_dinov2_lmc/memory_extraction/run_memory_extraction.py`

### 2.1 `load_dataset(...)` 总览

在 `run_memory_extraction.py` 里，`load_dataset(...)` 根据 `dataset_loader` 参数分派：

- **ACE 分支（CamLocDatasetDINOv2）**
  - 条件：`dataset_loader == "ace"`
  - 行为（概念）：
    - 打印标记：`[Dataset] Using ACE loader (CamLocDatasetDINOv2)`
    - 把上级目录加入 `sys.path`，以便导入 `dataset_dinov2`
    - 导入并实例化：
      - `from dataset_dinov2 import CamLocDatasetDINOv2`
      - `base_ds = CamLocDatasetDINOv2(dataset_path, mode=2)`
    - 用 `ACEDatasetWithDepth(base_ds)` 包一层，输出给后续 pipeline 使用

- **WAI 分支（SevenScenesWAI / Indoor6WAI）**
  - 条件：`dataset_loader == "wai"`（否则报错）
  - 行为（概念）：
    - 导入：
      - `from mapanything.datasets.wai.seven_scenes import SevenScenesWAI`
      - `from mapanything.datasets.wai.indoor6 import Indoor6WAI`
    - 从环境变量或默认路径获取 `dataset_metadata_dir`：
      - `MAPANYTHING_DATASET_METADATA_DIR` 或 `/mnt/storage/xwh/map-anything/mapanything_dataset_metadata`
    - 定义公共参数：
      - `base_kwargs = dict(resolution=518, data_norm_type='dinov2', transform='imgnorm')`
    - 当 `dataset_type == "7scenes"` 时：
      - 返回 `SevenScenesWAI(...)`，其中：
        - `ROOT=dataset_path`
        - `dataset_metadata_dir=metadata_dir`
        - `split='train'`
        - `sample_specific_scene=True`
        - `specific_scene_name=scene_name`（如 `chess_train`）
        - `sequential_view_mode=True`
        - `num_views=n_views`
        - 以及 `**base_kwargs`

## 3. ACE 路径细节：`CamLocDatasetDINOv2` -> `convert_ace_tuple_to_dict`

### 3.1 `CamLocDatasetDINOv2`：从 7Scenes / ACE 目录读原始数据

文件：`ace_depth/dataset_dinov2.py`

- **类定义**：`class CamLocDatasetDINOv2(Dataset):`

- **构造函数 `__init__(...)` 要点**：
  - 关键参数：
    - `root_dir`：场景根目录（包含 `rgb`, `poses`, `calibration`, `depth`/`eye` 等）
    - `mode`：
      - `0`：RGB only
      - `1`：RGB + scene coords
      - `2`：RGB-D（当前 memory_extract 中采用）
    - `image_height` 默认 `518`，`image_width=None`
    - `use_half=True`
  - 路径设定：
    - `rgb_dir = root_dir / 'rgb'`
    - `pose_dir = root_dir / 'poses'`
    - `calibration_dir = root_dir / 'calibration'`
    - `coord_dir` 根据 `mode` 和 `sparse` 选择 `depth` / `init` / `eye`
  - **DINOv2 Patch 约束**：
    - `self.patch_size = 14`
    - `self.image_height` 被 `_round_to_patch_size` 调整为 14 的倍数
    - 若设定的 `image_width` 不为 `None`，也会被调整为 14 的倍数
  - 读取与校验：
    - 列出 `self.rgb_files`、`self.pose_files`、`self.calibration_files`
    - 校验长度一致
  - 若启用 clustering，会基于相机中心做 k-means 聚类，筛选子集
  - 计算 `self.mean_cam_center` 以匹配 ACE 原始实现习惯

- **`_get_single_item(self, idx, image_height)` 要点**：
  这是 ACE 路径的“单帧读入 + 几何缩放”的关键函数。

  1. 按 `self.valid_file_indices[idx]` 重新映射下标
  2. 读取 RGB：
     - `image = io.imread(self.rgb_files[idx])`
     - 灰度 -> RGB：`color.gray2rgb`
  3. 读取 intrinsics：
     - `k = np.loadtxt(self.calibration_files[idx])`
     - 若 `k` 为 `3x3`：
       - `focal_length = [k[0][0], k[1][1]]`
       - `centre_point = [k[0][2], k[1][2]]`
  4. 保证 `image_height` 为 14 的倍数，并计算尺度：
     - `image_height = self._round_to_patch_size(image_height)`
     - `f_scale_factor = image_height / image.shape[0]`
     - `centre_point` 与 `focal_length` x `f_scale_factor`
  5. Resize 图像：
     - `_resize_image(image, image_height)` 保持纵横比
     - 若 `self.image_width` 不为 `None` 或当前宽度不是 14 的倍数，则再做一次宽度调整，并对 intrinsics 再次乘以 `w_scale`
  6. 构造 `image_mask = torch.ones((1,H,W))`
  7. 图像变换：
     - 使用 DINOv2 风格的 `transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
  8. 读取 pose：
     - `pose = np.loadtxt(self.pose_files[idx])` -> `torch.float` 的 4x4
  9. 按 `mode` 加载 `coords` / `depth`：
     - 对于 memory_extract 中使用的 `mode=2`（eye 模式），“coords” 通常对应 `eye/` 中的预计算坐标或 `depth` 信息，这部分在 ACE 原始实现中用于坐标回归。
  10. 若启用数据增强，会在图像、mask、coords/depth、pose 上同步施加旋转，并在非 sparse init 模式下从 depth 生成 3D 坐标。
  11. 构造 `intrinsics`：
      - 若 `centre_point` 存在：
        - `K[0,2] = centre_point[0]`
        - `K[1,2] = centre_point[1]`
        - `K[0,0] = focal_length[0]`
        - `K[1,1] = focal_length[1]`
      - 否则使用图像中心和单一焦距
      - 同时计算 `intrinsics_inv = intrinsics.inverse()`
  12. 返回 8 元组：
      - `(image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, str(self.rgb_files[idx]))`

- **`__getitem__`**：
  - 对单 index：直接调用 `_get_single_item(idx, self.image_height)`
  - 对 list/tuple index：批量调用 `_get_single_item` 后，按通道堆叠，并整理 `coords` 和 `filename` 列表。

### 3.2 `convert_ace_tuple_to_dict(...)`：ACE -> MapAnything 风格 dict

文件：`ace_depth/ace_dinov2_lmc/memory_extraction/run_memory_extraction.py`（靠近文件末尾）

- **函数签名**（概念）：
  - `def convert_ace_tuple_to_dict(sample, dataset_path: str) -> Dict[str, Any]:`
  - `sample`：来自 `CamLocDatasetDINOv2.__getitem__` 的 8 元组

- **主要逻辑**：
  1. 解包 8 元组：`image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, filename`
  2. 从 `filename` 推导 depth 路径：
     - `.../rgb/xxxxx.color.png` -> 替换为 `.../depth/xxxxx.depth.png`
     - 并确保相对于 `dataset_path` 的路径一致性
  3. 使用 `skimage.io.imread(depth_path)` 读取深度，单位 mm -> m
  4. 检查 depth 尺寸与 `image` 输出尺寸是否一致：
     - 若 `depth.shape[:2] != image.shape[-2:]`：
       - 使用 `torch.nn.functional.interpolate(..., mode="nearest")` 重采样至与 `image` 相同的高宽
  5. 整理输出结构：
     - `img`：形状 `[1, 3, H, W]`，数据类型多为 `float16` 或 `float32`，已按 DINOv2 方式归一化
     - `depth_z`：形状 `[1, H, W, 1]`，单位为米，最近邻对齐到 `img` 分辨率
     - `intrinsics`：形状 `[1, 3, 3]`
     - `camera_poses`：形状 `[1, 4, 4]`
     - 以及辅助信息如 `filename`、`image_mask` 等，下游关键字段与 MapAnything 期望一致

- **调用点**：
  - 在 `run_memory_extraction.py` 中构造 memory 视角时：
    - `if config.dataset_loader == "ace":`
      - `view = convert_ace_tuple_to_dict(raw_data, config.dataset_path)`
    - 并明确注释：**WAI loader 返回的是 `list[dict]`，不能传给这个函数**。

## 4. WAI 路径细节：`SevenScenesWAI` 与 `load_frame(...)`

### 4.1 `SevenScenesWAI`：官方 WAI 数据集包装

文件：`map-anything/mapanything/datasets/wai/seven_scenes.py`

- **类定义**：`class SevenScenesWAI(BaseDataset):`
- **构造函数 `__init__(..., ROOT, dataset_metadata_dir, split="test", ..., sequential_view_mode=False, **kwargs)` 要点**：
  1. 调用 `super().__init__`，完成基础归一化 / transform 设置。
  2. 保存：
     - `self.ROOT = ROOT`
     - `self.dataset_metadata_dir = dataset_metadata_dir`
     - `self.split = split`
     - `self.sample_specific_scene`, `self.specific_scene_name`
     - `self.sequential_view_mode`
  3. 调用 `_load_data()`：
     - 加载 `seven_scenes_scene_list_{split}.npy`，或 fallback 为扫描根目录下以 `_{split}` 结尾的子目录
     - 若 `sample_specific_scene=True`，限制为单一场景
     - 在 `sequential_view_mode=True` 下：
       - 为每个场景构建 `self.flat_view_list = [(scene_name, frame_name), ...]`
       - 预先加载并缓存每个场景的 `scene_meta.json` 到 `self.scene_meta_cache`
  4. 计算 `self.mean_cam_center = self._compute_mean_camera_center()`：
     - 遍历所有场景，读取 `scene_meta["frames"]` 中的 `transform_matrix` 或 `extrinsics`，累加平移部分。

- **`_get_views(self, sampled_idx, num_views_to_sample, resolution)` 要点**：
  - 在 `sequential_view_mode=True` 模式下：
    1. `sampled_idx` 映射到 `self.flat_view_list` 中的某个 `(scene_name, target_frame_name)`
    2. 根据 covisibility（若可用）选出额外视角，得到一个视角列表（长度为 `num_views_to_sample`）
    3. 对每个 `(scene_name, frame_name)`：
       - 调用 `load_frame(scene_root, scene_meta, frame_name, resolution, ...)`
       - `load_frame` 会：
         - 从 WAI 预处理结果中加载：
           - `img`（分辨率统一为 `resolution=518`）
           - `depth_along_ray` / `depth_z` / `depthmap`（MapAnything 训练域约定的深度语义）
           - `camera_pose`（4x4）
           - `camera_intrinsics`（3x3），并考虑任何裁剪 / 缩放的补偿
  - 返回：**`list[dict]`**，每个 dict 都是 MapAnything 官方设计的单视角结构。

- **`__getitem__(self, idx)`**：
  - 一般形式：调用 `_get_views(idx, num_views_to_sample=self.num_views, resolution=518)`，再进行适当打包。

### 4.2 `load_data(...)` / `load_frame(...)`：帧级 I/O

文件：`map-anything/mapanything/utils/wai/core.py`

- **`load_data(path, type)`**：
  - 负责从磁盘读取 json / npy / mmap 等多种格式，并做基本的缓存/转换。
- **`load_frame(scene_root, scene_meta, frame_name, resolution, ...)`**（概念说明）：
  - 使用 `scene_meta` 中关于该 `frame_name` 的记录：
    - 读取对应的 RGB 图像（通常已是 WAI 的预裁剪结果）
    - 根据 `resolution` 做统一重采样
    - 读取预计算的 `depth_along_ray` 或 `depth_z`（保证语义与 MapAnything 训练时一致）
    - 按照 WAI 预处理约定构造：
      - `camera_intrinsics`（考虑任何缩放/裁剪）
      - `camera_pose`（c2w / w2c 的一致性）
  - 返回一个包含 `img` / `depth_along_ray` / `depth_z` / `camera_pose` / `camera_intrinsics` 等字段的 dict。

## 5. 对齐 / 调试时可参考的阅读顺序

当你希望进一步理解 ACE 和 WAI 路径在“内参缩放、深度语义、坐标系”方面的差异时，可以按如下顺序阅读代码：

1. **图像与 intrinsics 缩放逻辑**
   - ACE：
     - `ace_depth/dataset_dinov2.py` -> `CamLocDatasetDINOv2._get_single_item`
     - 重点：`image_height` / `image_width` 调整、`f_scale_factor` / `w_scale` 以及 `focal_length` / `centre_point` 的缩放方式。
   - WAI：
     - `map-anything/mapanything/utils/wai/core.py` -> `load_frame`
     - `map-anything/mapanything/datasets/wai/seven_scenes.py` -> `_get_views`
     - 重点：如何在 WAI 预处理结果基础上，保持 intrinsics 与重采样后分辨率的一致性。

2. **深度语义 / 深度图生成**
   - ACE：
     - `convert_ace_tuple_to_dict` 中：
       - 从 `/depth/*.depth.png` 读取深度（mm -> m）
       - 与 `image` 分辨率对齐（最近邻插值）
       - 整理为 `[1, H, W, 1]` 的 `depth_z`
   - WAI：
     - `load_frame` 中：
       - 如何读取 `depth_along_ray` / `depth_z`
       - 是否有范围截断 / 无效值填充 / 单位缩放
     - 这部分直接决定 WAI 深度分布与 MapAnything 训练域的一致性。

3. **pose / 坐标系一致性**
   - ACE：
     - `CamLocDatasetDINOv2._load_pose`（4x4 矩阵）
     - `_get_single_item` 中的 `pose` / `pose_inv` / `mean_cam_center` 计算方式
   - WAI：
     - `scene_meta["frames"][i]["transform_matrix"]` 或 `extrinsics`
     - `_compute_mean_camera_center` 中对 pose 的解析方式
     - `load_frame` 返回的 `camera_pose` 与 `scene_meta` 中矩阵的对应关系

---

这份文档只做代码位置和职责梳理，没有对任何路径的实现逻辑做改写建议，因此不会影响当前 WAI / MapAnything 官方数据流。后续可以基于这份文档继续做“只作用于 ACE 分支”的差异分析与改进设计。
