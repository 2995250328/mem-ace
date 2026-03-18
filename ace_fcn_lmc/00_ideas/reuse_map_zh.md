# 复用映射

> 以下所有文件均已在实现中复用，不应重新实现。

| 现有文件 | 相关类/函数 | 用途 |
|---|---|---|
| `../ace_network.py` | `Encoder` | ACE FCN backbone（被 ACEEncoder 包装） |
| `../ace_network_dinov2.py` | `Head` | 坐标回归头（直接复用） |
| `../trainer_dinov2.py` | `TrainerACEDINOv2` | Vanilla 训练逻辑基类 |
| `../trainer_dinov2_lmc.py` | `TrainerACEDINOv2LMC` | LMC 两阶段训练逻辑基类 |
| `../train_ace_dinov2_lmc.py` | `run_vanilla_iterative_baseline`, `run_post_train_eval` | 训练入口逻辑模板 |
| `../options_dinov2_lmc.py` | `get_lmc_train_parser` | 参数解析器基础（被 options_ace_lmc.py 扩展） |
| `../dataset.py` | `CamLocDataset` | 标准 ACE 数据集加载（灰度图） |
| `../dataset_wai_dinov2.py` | `WAIDINOv2Dataset` | WAI 数据后端 |
| `../ace_loss.py` | `ReproLoss` | 重投影损失函数 |
| `../ace_util.py` | `get_pixel_grid`, `to_homogeneous` | 几何工具函数 |
| `../result_manager.py` | `ResultManager` | 层级化结果管理 |
| `../utils_lmc.py` | `build_lmc_run_folder_config_tag`, `setup_cuda_environment` | 共享工具函数 |
| `../eval_poses.py` | `eval_poses` | 姿态评估指标计算 |
| `../dsacstar/` | C++ RANSAC bindings | DSAC* 姿态估计 |

## ace_fcn_lmc 新增文件

| 新文件 | 用途 |
|---|---|
| `../ace_network_ace.py` | ACEEncoder 包装器 + RegressorACE（适配 LMC 接口） |
| `../options_ace_lmc.py` | ACE FCN 专属参数（encoder_path、无 patch-size 约束等） |
| `../trainer_ace_fcn.py` | TrainerACEFCN / TrainerACEFCNLMC（仅覆盖 _create_regressor） |
| `../train_ace_lmc.py` | 训练入口（与 train_ace_dinov2_lmc.py 结构一致） |
| `../test_ace_lmc.py` | 评估脚本（与 test_ace_dinov2_lmc.py 结构一致） |
