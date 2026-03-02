#!/usr/bin/env python3
"""
检查场景目录是否符合 DINOv2-ACE 所需的数据格式（与 dataset_dinov2.py 一致）。
用于在训练前确认 Cambridge / 7-Scenes 等数据集能否直接使用。

所需结构（与 DSAC* / ACE 一致）：
  <scene>/
    train/
      rgb/          # 训练图像（.png/.jpg 等）
      poses/        # 每帧 4x4 位姿矩阵，与 rgb 数量、排序一致
      calibration/  # 每帧内参：一个数（焦距）或 3x3 矩阵
    test/
      rgb/
      poses/
      calibration/

注意：rgb、poses、calibration 三个目录下文件按 sorted() 顺序一一对应，建议使用相同基名（如 1000.png / 1000.pose / 1000.calibration）。
"""
import argparse
import sys
from pathlib import Path


def check_scene(scene_path: Path) -> bool:
    scene_path = Path(scene_path).resolve()
    if not scene_path.is_dir():
        print(f"错误: 不是目录: {scene_path}")
        return False

    ok = True
    for split in ("train", "test"):
        root = scene_path / split
        if not root.is_dir():
            print(f"错误: 缺少 {split}/ 目录: {root}")
            ok = False
            continue

        rgb_dir = root / "rgb"
        pose_dir = root / "poses"
        calib_dir = root / "calibration"

        for name, d in (("rgb", rgb_dir), ("poses", pose_dir), ("calibration", calib_dir)):
            if not d.is_dir():
                print(f"错误: 缺少 {split}/{name}/ 目录: {d}")
                ok = False
                continue

        if not ok:
            continue

        rgb_files = sorted(rgb_dir.iterdir())
        pose_files = sorted(pose_dir.iterdir())
        calib_files = sorted(calib_dir.iterdir())

        nr, np_, nc = len(rgb_files), len(pose_files), len(calib_files)
        if nr != np_ or nr != nc:
            print(f"错误: {split}/ 下数量不一致: rgb={nr}, poses={np_}, calibration={nc}")
            ok = False
        else:
            print(f"  {split}: rgb={nr}, poses={np_}, calibration={nc}")

        # 抽查第一帧：pose 4x4，calibration 1 或 3x3（需要 numpy）
        if rgb_files and pose_files and calib_files:
            try:
                import numpy as np
                p = np.loadtxt(pose_files[0])
                c = np.loadtxt(calib_files[0])
                pshape = getattr(p, "shape", ())
                cshape = getattr(c, "shape", ())
                csize = getattr(c, "size", 1)
                if pshape != (4, 4):
                    print(f"错误: {split}/poses 首文件应为 4x4 矩阵，当前 shape={pshape}")
                    ok = False
                if csize != 1 and cshape != (3, 3):
                    print(f"错误: {split}/calibration 首文件应为 1 个数或 3x3 矩阵，当前 shape={cshape}")
                    ok = False
            except ImportError:
                print(f"  提示: 未安装 numpy，跳过 pose/calibration 格式抽查")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Check if a scene folder is valid for DINOv2-ACE (train/test, rgb/poses/calibration)."
    )
    parser.add_argument("scene", type=Path, help="Scene root (e.g. /data/xwh/Cambridge/GreatCourt)")
    args = parser.parse_args()

    print(f"检查场景: {args.scene}")
    if check_scene(args.scene):
        print("检查通过，可直接用于 train_ace_dinov2.py / test_ace_dinov2.py")
        return 0
    print("检查未通过，请按上述提示补全或调整目录与文件。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
