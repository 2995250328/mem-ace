#!/usr/bin/env python3
"""
indoor6 位姿坐标系验证脚本。

目的：检查 pose.txt 与 COLMAP 坐标系是否一致，是否存在 scale/origin/rotation 差异。
现象：旋转误差较小（~3°）但平移误差巨大（~138cm）时，常见于 scale 或 origin 不一致。

检查项：
1. pose.txt 相机中心统计（范围、单位合理性）
2. pose.txt vs COLMAP（若 indoor6-colmap 可用）：Umeyama 相似变换（scale/R/t）
3. 可选：ACE 预测 pose vs GT pose，对齐后误差是否骤降（验证假设）
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_pose(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        raw = np.loadtxt(path, dtype=np.float64)
        flat = raw.ravel()
        if flat.size == 16:
            return flat.reshape(4, 4)
        if flat.size == 12:
            P = np.eye(4, dtype=np.float64)
            P[:3, :] = flat.reshape(3, 4)
            return P
    except Exception:
        pass
    return None


def _camera_center_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return np.array(c2w[:3, 3], dtype=np.float64)


def _umeyama_similarity(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Umeyama: dst ≈ s * R @ src + t. Returns (T_4x4, scale, R, t)."""
    n = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_c = src_pts - src_mean
    dst_c = dst_pts - dst_mean
    var_src = (src_c**2).sum() / n
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    s = np.trace(np.diag(S)) / (var_src * n) if var_src > 1e-12 else 1.0
    t = dst_mean - s * R @ src_mean
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T, s, R, t


def _qvec2rotmat(q):
    """COLMAP qvec (qw,qx,qy,qz) -> 3x3 R (camera-to-world)."""
    q = np.asarray(q, dtype=np.float64)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)


def load_colmap_images_from_txt(images_txt: Path) -> dict:
    """Parse COLMAP images.txt. Returns dict: image_name -> (qvec, tvec)."""
    out = {}
    with open(images_txt) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = " ".join(parts[9:]).strip()
        i += 1
        if i < len(lines) and not lines[i].startswith("#"):
            i += 1
        out[name] = (np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]))
    return out


def load_colmap_images(colmap_sparse: Path) -> dict | None:
    """Load COLMAP images (binary via pycolmap, or txt). Returns dict: image_name -> (qvec, tvec)."""
    images_txt = colmap_sparse / "images.txt"
    if images_txt.exists():
        return load_colmap_images_from_txt(images_txt)
    # 优先用 pycolmap 读 binary
    try:
        import pycolmap
        model = pycolmap.Reconstruction(colmap_sparse)
        out = {}
        for img in model.images.values():
            name = img.name
            q = img.cam_from_world.rotation.quat  # (qw,qx,qy,qz)
            t = img.cam_from_world.translation
            out[name] = (np.array([q[0], q[1], q[2], q[3]]), np.array(t))
        return out if out else None
    except ImportError:
        pass
    except Exception as e:
        print(f"  pycolmap 读取失败: {e}")
    # 备选：导出为 txt 再读（需 colmap model_converter）
    print("  提示: 安装 pycolmap 或运行 model_converter 导出 images.txt 以比对 COLMAP")
    return None


def colmap_pose_to_c2w(qvec, tvec):
    """COLMAP qvec/tvec (w2c) -> c2w 4x4."""
    R_w2c = _qvec2rotmat(qvec)
    t_w2c = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R_c2w = R_w2c.T
    t_c2w = (-R_c2w @ t_w2c).ravel()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w
    return c2w


def check_pose_statistics(ace_scene_dir: Path, scene_name: str, indoor6_root: Path):
    """1. 统计 pose.txt 相机中心范围。"""
    print("\n" + "=" * 60)
    print("1. pose.txt 相机中心统计 (ACE 格式)")
    print("=" * 60)
    pose_dir = ace_scene_dir / "test" / "poses"
    if not pose_dir.is_dir():
        print(f"  无 test/poses: {pose_dir}")
        return None
    pose_files = sorted(pose_dir.iterdir())
    centers = []
    for pf in pose_files[:min(500, len(pose_files))]:  # 采样最多 500 帧
        P = np.loadtxt(pf)
        c = _camera_center_from_c2w(P)
        centers.append(c)
    centers = np.array(centers)
    extent = centers.max(axis=0) - centers.min(axis=0)
    span = np.linalg.norm(extent)
    print(f"  帧数: {len(centers)}")
    print(f"  中心范围 X: [{centers[:, 0].min():.3f}, {centers[:, 0].max():.3f}] m")
    print(f"  中心范围 Y: [{centers[:, 1].min():.3f}, {centers[:, 1].max():.3f}] m")
    print(f"  中心范围 Z: [{centers[:, 2].min():.3f}, {centers[:, 2].max():.3f}] m")
    print(f"  轨迹跨度 (L2): {span:.3f} m")
    if span < 0.5:
        print("  ⚠ 轨迹跨度 < 0.5m，可能单位是米但场景很小，或 pose 有误")
    elif span > 100:
        print("  ⚠ 轨迹跨度 > 100m，可能单位是 cm/mm 被误读为 m")
    return centers


def check_pose_vs_colmap(ace_scene_dir: Path, scene_name: str, indoor6_root: Path, colmap_root: Path):
    """2. 比对 pose.txt 与 COLMAP 坐标系（需有 indoor6-colmap）。"""
    print("\n" + "=" * 60)
    print("2. pose.txt vs COLMAP 坐标系")
    print("=" * 60)
    colmap_sparse = colmap_root / scene_name / "sparse" / "0"
    if not colmap_sparse.is_dir():
        colmap_sparse = colmap_root / scene_name / "sparse"
    if not colmap_sparse.is_dir():
        print(f"  无 COLMAP sparse 目录: {colmap_sparse}")
        return
    colmap_poses = load_colmap_images(colmap_sparse)
    if colmap_poses is None:
        print(f"  无法加载 COLMAP 模型 (images.txt/images.bin): {colmap_sparse}")
        return
    ace_pose_dir = ace_scene_dir / "test" / "poses"
    ace_rgb_dir = ace_scene_dir / "test" / "rgb"
    if not ace_pose_dir.is_dir() or not ace_rgb_dir.is_dir():
        print("  无 ACE test/poses 或 test/rgb")
        return
    # 建立 ACE 基名 -> 文件名映射（ACE 用 000000.pose 等）
    ace_files = {p.stem: p for p in ace_pose_dir.iterdir() if p.suffix in (".pose", ".txt")}
    # 建立原始 indoor6 名 -> ACE 基名（需从 train_test_val 或 scene*_test.txt 推导）
    # ACE adapt 时 test 列表是 pkl["test"]，顺序 0,1,2... 对应 000000, 000001
    pkl_path = indoor6_root / scene_name / "train_test_val.pkl"
    if not pkl_path.exists():
        print("  无 train_test_val.pkl，无法建立 ACE 与原始帧名对应")
        return
    import pickle
    with open(pkl_path, "rb") as f:
        split = pickle.load(f)
    test_list = split.get("test", [])
    # test_list 元素如 "image-000633.color.jpg"
    gt_centers = []
    colmap_centers = []
    matched = []
    for idx, fname in enumerate(test_list[:min(300, len(test_list))]):
        base = fname.replace(".color.jpg", "").replace(".color.png", "")  # image-000633
        stem = f"{idx:06d}"
        ace_pose = ace_pose_dir / f"{stem}.pose"
        if not ace_pose.exists():
            continue
        P_ace = np.loadtxt(ace_pose)
        c_gt = _camera_center_from_c2w(P_ace)
        gt_centers.append(c_gt)
        # COLMAP 中图像名可能是 image-000633.color.jpg 或类似
        colmap_name = None
        for k in colmap_poses:
            if base in k or k.replace(".jpg", "").replace(".png", "") == base:
                colmap_name = k
                break
        if colmap_name is None:
            # 尝试直接匹配
            colmap_name = fname if fname in colmap_poses else None
        if colmap_name is not None:
            q, t = colmap_poses[colmap_name]
            c2w_colmap = colmap_pose_to_c2w(q, t)
            c_colmap = _camera_center_from_c2w(c2w_colmap)
            colmap_centers.append(c_colmap)
            matched.append((c_gt, c_colmap))
    gt_centers = np.array(gt_centers)
    if len(matched) < 3:
        print(f"  可匹配帧数不足: {len(matched)} (需>=3)")
        return
    colmap_centers = np.array([m[1] for m in matched])
    gt_matched = np.array([m[0] for m in matched])
    T, scale, R, t = _umeyama_similarity(colmap_centers, gt_matched)
    print(f"  匹配帧数: {len(matched)}")
    print(f"  Umeyama scale (COLMAP->pose.txt): {scale:.6f}")
    print(f"  Umeyama 平移 t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    if abs(scale - 1.0) > 0.01:
        print("  ⚠ scale 明显偏离 1.0 → COLMAP 与 pose.txt 单位或尺度不一致")
    elif abs(scale - 1.0) > 0.001:
        print("  scale 略有偏差，可能在数值误差内")
    # 对齐后误差
    colmap_aligned = (scale * (R @ colmap_centers.T).T + t)
    errs = np.linalg.norm(colmap_aligned - gt_matched, axis=1)
    print(f"  对齐后平移误差 (m): median={np.median(errs):.4f}, mean={np.mean(errs):.4f}, max={np.max(errs):.4f}")


def check_ace_pred_vs_gt(poses_txt: Path | None, ace_scene_dir: Path, indoor6_root: Path):
    """3. 若有 ACE 测试输出的 poses_*.txt，检查 GT vs 预测，对齐前后误差。

    ACE poses 格式: frame_name q_w qx qy qz tx ty tz r_err t_err inlier_count
    其中 q,t 是 pose_inv (w2c)，c2w 相机中心 C = -R_w2c^T @ t_w2c
    """
    print("\n" + "=" * 60)
    print("3. ACE 预测 vs GT（对齐前后）")
    print("=" * 60)
    if poses_txt is None or not poses_txt.exists():
        print("  未指定 poses_*.txt（ACE 测试输出的位姿日志），跳过")
        print("  用法: 加上 --poses_log <path/to/poses_scene1_test.txt>")
        return
    lines = poses_txt.read_text().strip().split("\n")
    pred_centers = []
    gt_from_ace = []
    ace_pose_dir = ace_scene_dir / "test" / "poses"
    stem_to_pose = {}
    if ace_pose_dir.exists():
        for pf in ace_pose_dir.iterdir():
            if pf.suffix in (".pose", ".txt"):
                stem_to_pose[pf.stem] = np.loadtxt(pf)
    scene_name = ace_scene_dir.name
    pkl_path = indoor6_root / scene_name / "train_test_val.pkl"
    name_to_stem = {}
    if pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            split = pickle.load(f)
        for idx, fname in enumerate(split.get("test", [])):
            stem = f"{idx:06d}"
            base = fname.replace(".color.jpg", "").replace(".color.png", "")
            name_to_stem[base] = stem
            name_to_stem[Path(fname).name] = stem
    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        fname = parts[0]
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        q = np.array([qw, qx, qy, qz])
        t = np.array([tx, ty, tz])
        R_w2c = _qvec2rotmat(q)
        C = -R_w2c.T @ t
        pred_centers.append(C)
        base = Path(fname).stem.replace(".color", "").replace(".jpg", "").replace(".png", "")
        stem = name_to_stem.get(base) or name_to_stem.get(fname) or (base.replace("image-", "").zfill(6) if "image-" in base else base)
        if stem in stem_to_pose:
            P = stem_to_pose[stem]
            c_gt = _camera_center_from_c2w(P)
            gt_from_ace.append(c_gt)
        else:
            gt_from_ace.append(None)
    pred_centers = np.array(pred_centers)
    valid = [i for i, g in enumerate(gt_from_ace) if g is not None]
    if len(valid) < 3:
        print(f"  有效匹配不足: {len(valid)}，需 ACE test/poses 与 pkl 对应")
        return
    pred_valid = pred_centers[valid]
    gt_valid = np.array([gt_from_ace[i] for i in valid])
    err_raw = np.linalg.norm(pred_valid - gt_valid, axis=1) * 100  # cm
    T, scale, R, t = _umeyama_similarity(pred_valid, gt_valid)
    pred_aligned = (scale * (R @ pred_valid.T).T + t)
    err_aligned = np.linalg.norm(pred_aligned - gt_valid, axis=1) * 100  # cm
    print(f"  有效帧数: {len(valid)}")
    print(f"  未对齐平移误差 (cm): median={np.median(err_raw):.1f}, mean={np.mean(err_raw):.1f}")
    print(f"  Umeyama scale (pred->GT): {scale:.6f}")
    print(f"  对齐后平移误差 (cm): median={np.median(err_aligned):.1f}, mean={np.mean(err_aligned):.1f}")
    if abs(scale - 1.0) > 0.05:
        print("  ⚠ scale 明显偏离 1.0 → 预测与 GT 存在尺度差异")
    if np.median(err_aligned) < np.median(err_raw) * 0.5:
        print("  ✓ 对齐后误差明显下降 → 存在坐标系（scale/origin/rotation）不一致")
    else:
        print("  (对齐后误差未明显改善，可能非坐标系问题)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify indoor6 pose coordinate system vs COLMAP / ACE prediction."
    )
    parser.add_argument(
        "ace_scene_dir",
        type=Path,
        help="ACE 场景目录，如 /data/xwh/indoor6_ace/scene1",
    )
    parser.add_argument(
        "--indoor6_root",
        type=Path,
        default=Path("/data/xwh/indoor6"),
        help="indoor6 原始根目录",
    )
    parser.add_argument(
        "--colmap_root",
        type=Path,
        default=Path("/data/xwh/indoor6/indoor6-colmap"),
        help="COLMAP 根目录 (indoor6-colmap)",
    )
    parser.add_argument(
        "--poses_log",
        type=Path,
        default=None,
        help="ACE 测试输出的 poses_*.txt，用于对比预测 vs GT",
    )
    args = parser.parse_args()

    ace_scene_dir = args.ace_scene_dir.resolve()
    scene_name = ace_scene_dir.name

    print(f"场景: {scene_name}")
    print(f"ACE 目录: {ace_scene_dir}")
    print(f"indoor6 根: {args.indoor6_root}")
    print(f"COLMAP 根: {args.colmap_root}")

    check_pose_statistics(ace_scene_dir, scene_name, args.indoor6_root)
    check_pose_vs_colmap(ace_scene_dir, scene_name, args.indoor6_root, args.colmap_root)
    check_ace_pred_vs_gt(args.poses_log, ace_scene_dir)

    print("\n" + "=" * 60)
    print("完成。若 scale 偏离 1.0 或对齐后误差骤降，需在转换时做坐标系对齐。")
    print("=" * 60)


if __name__ == "__main__":
    main()
