#!/usr/bin/env python3
"""
将 indoor6 数据集转换为 ACE DINOv2 / train_ace_dinov2_lmc 所需格式。

indoor6 每场景结构（与 ace-g/datasets/setup_indoor6.py 一致）：
  <scene>/
    images/
      image-XXXXXX.color.jpg
      image-XXXXXX.pose.txt      # 3x4 矩阵，world-to-camera (w2c)，需取逆得 c2w
      image-XXXXXX.intrinsics.txt  # 一行: width height fx cx cy [其余忽略]
    train_test_val.pkl          # 含 'train', 'test', 'val' 列表，元素为文件名如 image-000000.color.jpg

输出结构：
  <out_root>/<scene>/
    train/  (rgb, poses, calibration)
    test/   (rgb, poses, calibration)
文件按 sorted() 顺序一一对应，命名为 000000.jpg / 000000.pose / 000000.calibration 等。
"""
import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


# 默认场景名（排除 indoor6-colmap 与 *-tr）
DEFAULT_SCENES = ("scene1", "scene2a", "scene3", "scene4a", "scene5", "scene6")


def parse_pose_3x4(path: Path) -> np.ndarray:
    """读取 indoor6 3x4 pose (world-to-camera)，取逆返回 c2w 4x4。

    与 ace-g/datasets/setup_indoor6.py 一致：indoor6 pose.txt 是 w2c，需 inv 得 c2w。
    """
    p = np.loadtxt(path)
    if p.shape == (3, 4):
        P_w2c = np.eye(4)
        P_w2c[:3, :] = p
        P_c2w = np.linalg.inv(P_w2c)
        return P_c2w
    if p.shape == (4, 4):
        # 若已是 4x4，假定为 w2c（与 ace-g 保持一致）
        return np.linalg.inv(p)
    raise ValueError(f"Unexpected pose shape {p.shape} at {path}")


def parse_intrinsics(path: Path) -> np.ndarray:
    """解析一行: width height fx cx cy ...，返回 3x3 K。"""
    with open(path) as f:
        parts = f.read().strip().split()
    if len(parts) < 5:
        raise ValueError(f"Need at least 5 numbers in intrinsics, got {path}")
    w, h = int(parts[0]), int(parts[1])
    fx = float(parts[2])
    cx = float(parts[3])
    cy = float(parts[4])
    # fy 未给出时用 fx
    fy = fx
    K = np.array([[fx, 0, cx], [0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def convert_scene(
    scene_dir: Path,
    out_scene_dir: Path,
    symlink: bool = True,
) -> None:
    images_dir = scene_dir / "images"
    pkl_path = scene_dir / "train_test_val.pkl"
    if not images_dir.is_dir():
        print(f"  跳过（无 images/）: {scene_dir}")
        return
    if not pkl_path.is_file():
        print(f"  跳过（无 train_test_val.pkl）: {scene_dir}")
        return

    with open(pkl_path, "rb") as f:
        split = pickle.load(f)
    train_list = split.get("train", [])
    test_list = split.get("test", [])
    if not train_list or not test_list:
        print(f"  跳过（train 或 test 为空）: {scene_dir}")
        return

    for split_name, file_list in (("train", train_list), ("test", test_list)):
        rgb_dir = out_scene_dir / split_name / "rgb"
        poses_dir = out_scene_dir / split_name / "poses"
        calib_dir = out_scene_dir / split_name / "calibration"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)
        calib_dir.mkdir(parents=True, exist_ok=True)

        for idx, fname in enumerate(file_list):
            # fname 如 image-000000.color.jpg
            base = fname.replace(".color.jpg", "").replace(".color.png", "")
            stem = base.replace("image-", "")
            # 统一输出基名：六位数字，保证 sorted() 顺序一致
            out_base = f"{idx:06d}"

            # 图像：复制或软链
            src_img = images_dir / fname
            if not src_img.is_file():
                src_img = images_dir / (base + ".color.png")
            if not src_img.is_file():
                print(f"  警告: 未找到 {src_img}")
                continue
            ext = src_img.suffix.lower()
            dst_img = rgb_dir / f"{out_base}{ext}"
            if symlink:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)

            # 位姿：3x4 -> 4x4
            pose_src = images_dir / f"{base}.pose.txt"
            if not pose_src.is_file():
                pose_src = images_dir / (base + ".pose.txt")
            if pose_src.is_file():
                P = parse_pose_3x4(pose_src)
                np.savetxt(poses_dir / f"{out_base}.pose", P, fmt="%.6f")
            else:
                print(f"  警告: 未找到 {pose_src}")

            # 内参：3x3
            intr_src = images_dir / f"{base}.intrinsics.txt"
            if not intr_src.is_file():
                intr_src = images_dir / (base + ".intrinsics.txt")
            if intr_src.is_file():
                K = parse_intrinsics(intr_src)
                np.savetxt(calib_dir / f"{out_base}.calibration", K, fmt="%.6f")
            else:
                print(f"  警告: 未找到 {intr_src}")

    print(f"  {scene_dir.name}: train={len(train_list)}, test={len(test_list)} -> {out_scene_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert indoor6 per-scene layout to ACE train/test rgb/poses/calibration."
    )
    parser.add_argument(
        "indoor6_root",
        type=Path,
        help="Root of indoor6 dataset (e.g. /data/xwh/indoor6)",
    )
    parser.add_argument(
        "out_root",
        type=Path,
        help="Output root (e.g. /data/xwh/indoor6_ace). Each scene will be out_root/<scene>/train|test/...",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=list(DEFAULT_SCENES),
        help=f"Scene names to convert. Default: {DEFAULT_SCENES}",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Copy images instead of symlinking",
    )
    args = parser.parse_args()

    indoor6_root = args.indoor6_root.resolve()
    out_root = args.out_root.resolve()
    if not indoor6_root.is_dir():
        print(f"错误: 不是目录: {indoor6_root}", file=sys.stderr)
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    for scene_name in args.scenes:
        scene_dir = indoor6_root / scene_name
        if not scene_dir.is_dir():
            print(f"跳过（不存在）: {scene_dir}")
            continue
        convert_scene(scene_dir, out_root / scene_name, symlink=not args.no_symlink)

    print("完成。校验: python check_dataset_dinov2.py <out_root>/<scene>")


if __name__ == "__main__":
    main()
