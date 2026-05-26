**Scope**
只记录数据处理、下载、解压、格式转换、稀疏深度和目录规范。
不涉及 `GLACE-LMC / ACE-G / fusion / residual / model_backend`。

**Current Layout**
原始压缩包与大文件:
`/data/xwh/dataset_staging/naver_indoor`
`/data/xwh/dataset_staging/inloc`
`/data/xwh/Wayspots`
`/data/xwh/MuSHRoom`

已转换的 ACE/WAI:
`/data/xwh/Wayspots_wai`
`/data/xwh/MuSHRoom_ace`
`/data/xwh/MuSHRoom_wai`

建议给 NAVER 统一新增:
`/data/xwh/dataset_staging/naver_indoor_extracted`
`/data/xwh/NAVER_ace`
`/data/xwh/NAVER_wai`

**NAVER**
当前已发现 `HyundaiDepartmentStore_4F` 目录下已有:
`HyundaiDepartmentStore_4F_release_mapping.tar.gz`
`HyundaiDepartmentStore_4F_release_mapping_lidar_only.tar.gz`
`HyundaiDepartmentStore_4F_release_test.tar.gz`
`HyundaiDepartmentStore_4F_release_validation.tar.gz`

archive 内部真实根路径不是平铺结构，而是:
`HyundaiDepartmentStore/4F/release/<split>/...`

最小可验证链路:
1. 先用 `mapping` 做 train，`validation` 做 test。
2. 自动选择 mapping/validation 共有的一个 camera sensor。
3. 只生成 `rgb + poses + calibration`，先打通 ACE/WAI 消费链。
4. depth/lidar 留在原始解压目录，后续再做稀疏深度和投影可视化。

执行命令:
```bash
python tools/prepare_naver_indoor_scene.py --write-wai
```

输出:
`/data/xwh/NAVER_ace/naver_hyundai_4f_minival`
`/data/xwh/NAVER_wai/naver_hyundai_4f_minival_train`
`/data/xwh/NAVER_wai/naver_hyundai_4f_minival_test`

说明:
- 原始数据: `/data/xwh/dataset_staging/naver_indoor/...`
- 解压产物: `/data/xwh/dataset_staging/naver_indoor_extracted/...`
- 转换产物: `/data/xwh/NAVER_ace/...` 与 `/data/xwh/NAVER_wai/...`
- 清单产物: `.../manifests/summary.json`

**InLoc**
当前本地已有:
`/data/xwh/dataset_staging/inloc/InLoc_dataset`
`/data/xwh/dataset_staging/inloc/raw`

注意:
`raw` 目录里除 `cutouts.tar.gz` 外，许多文件目前只有约 `95 KB`，高度疑似是 HTML/redirect，而不是真实压缩包。

执行验包:
```bash
python tools/check_inloc_downloads.py \
  --write-json /data/xwh/dataset_staging/inloc/inloc_download_audit.json
```

判定规则:
- `gzip / bzip2 / zip` magic 正常: `ok`
- 文件缺失: `missing`
- 开头是 HTML: `html`
- 其余异常或极小文件: `suspect`

这一步只负责把“能用 / 失效 / 需重下”的状态固定下来，不直接进入训练。

**Wayspots / MuSHRoom**
现有相关脚本:
`tools/wayspots_known_pose_sparse_depth.py`
`tools/project_colmap_sparse_depth.py`
`tools/visualize_sparse_depth_overlay.py`
`tools/convert_ace_to_wai.py`
`tools/convert_mushroom_kinect_to_ace.py`
`tools/convert_mushroom_kinect_to_wai.py`

后续如果继续做数据处理，优先顺序建议:
1. `overlay` 可视化检查
2. 点云/深度过滤
3. sparse depth 采样/NMS 统一
4. 命名统一
5. `ply` 导出兼容 Mac 预览

**Loader Contract**
ACE 训练侧当前只要求:
`train/rgb`
`train/poses`
`train/calibration`
以及对应 `test/...`

WAI 训练侧当前只要求:
`scene_meta.json`
每帧包含:
- `image`
- `transform_matrix`
- `fl_x / fl_y / cx / cy`

这意味着新数据集第一阶段不必先实现完整 depth/lidar 消费，只要先把图像、位姿、内参规范化即可。
