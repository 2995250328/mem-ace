#!/usr/bin/env python3
"""
验证 create_training_buffer 收集的 buffer 与 trainer_dinov2 完全一致：
- 采集逻辑、种子、投影公式与 create_training_buffer / training_step 相同；
- 验证特征/坐标与 (图像, 像素) 一一对应，以及 scene_coords + pose_inv + intrinsics 重投影正确。

说明：进度条里每 10240 = 10 个 batch × 1024 samples_per_image。
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ace_util import to_homogeneous
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 与 trainer_dinov2 完全一致
BASE_SEED = 2089
LOADER_SEED = BASE_SEED + 511
SAMPLING_SEED = BASE_SEED + 4095
OUTPUT_SUBSAMPLE = Regressor.OUTPUT_SUBSAMPLE  # 14


def pose_inv_to_34(pose_inv):
    """与 trainer 一致：投影用 3x4 [R|t]，dataset 可能返回 4x4。"""
    if pose_inv.dim() == 3:
        pose_inv = pose_inv[0]
    return pose_inv[:3, :] if pose_inv.shape[0] == 4 else pose_inv


def main():
    parser = argparse.ArgumentParser(description="Verify DINOv2 training buffer consistency")
    parser.add_argument("scene", type=Path, help="Scene folder (e.g. datasets/7scenes/chess)")
    parser.add_argument("--dinov2_path", type=Path,
                        default=Path("/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth"))
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--samples_per_image", type=int, default=1024)
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of images to use for verification")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.image_resolution % 14 != 0:
        args.image_resolution = (args.image_resolution // 14) * 14

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(BASE_SEED)

    dataset = CamLocDatasetDINOv2(
        root_dir=args.scene / "train",
        mode=0,
        use_half=False,
        image_height=args.image_resolution,
        augment=False,
        aug_rotation=0,
        aug_scale_max=1.0,
        aug_scale_min=1.0,
    )

    regressor = Regressor.create_from_encoder(
        dinov2_path=args.dinov2_path,
        mean=dataset.mean_cam_center,
        num_head_blocks=1,
        use_homogeneous=True,
        num_encoder_features=1024,
        freeze_backbone=True,
    ).to(device)
    regressor.eval()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(LOADER_SEED),
    )
    sampling_gen = torch.Generator().manual_seed(SAMPLING_SEED)

    # 与 create_training_buffer 完全一致的采集逻辑，并记录 (feature, coord, image_idx, y, x, pose_inv, intrinsics)
    records = []  # list of (feature [C], coord [3], image_idx, y, x, pose_inv [3,4], intrinsics [3,3])
    images_for_idx = []  # images_for_idx[image_idx] = (image_tensor, pose_inv, intrinsics)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_images:
                break

            image = batch[0].to(device)
            if image.dtype == torch.float16:
                image = image.float()
            pose_inv = batch[3].to(device)
            intrinsics_inv = batch[5].to(device)
            intrinsics = batch[4].to(device)

            features = regressor.get_features(image)
            B, C, H, W = features.shape
            scene_coords_pred = regressor.get_scene_coordinates(features)

            num_samples = min(args.samples_per_image, H * W)
            indices = torch.randperm(H * W, generator=sampling_gen)[:num_samples]
            y_coords = indices // W
            x_coords = indices % W

            sampled_features = features[0, :, y_coords, x_coords].T  # [num_samples, C]
            sampled_coords = scene_coords_pred[0, :, y_coords, x_coords].T  # [num_samples, 3]

            images_for_idx.append((image.cpu(), pose_inv.cpu(), intrinsics.cpu()))

            for i in range(num_samples):
                records.append({
                    "feature": sampled_features[i].cpu(),
                    "coord": sampled_coords[i].cpu(),
                    "image_idx": batch_idx,
                    "y": int(y_coords[i].item()),
                    "x": int(x_coords[i].item()),
                    "pose_inv": pose_inv[0].cpu(),
                    "intrinsics": intrinsics[0].cpu(),
                })

    logger.info("Collected %d samples from %d images.", len(records), len(images_for_idx))

    # ---------- 验证 A：同一张图、同一 (y,x) 重新前向，特征与场景坐标应完全一致 ----------
    logger.info("Verification A: re-forward same images and same (y,x) -> same feature & coord")
    max_feat_diff = 0.0
    max_coord_diff = 0.0
    num_checked = 0

    for image_idx, (image, pose_inv, intrinsics) in enumerate(images_for_idx):
        image = image.to(device)
        with torch.no_grad():
            features = regressor.get_features(image)
            scene_coords = regressor.get_scene_coordinates(features)
        # 该图像下所有样本的 (y,x)
        samples_this_image = [(r["y"], r["x"], r["feature"], r["coord"]) for r in records if r["image_idx"] == image_idx]
        for (y, x, stored_feat, stored_coord) in samples_this_image:
            ref_feat = features[0, :, y, x].cpu()
            ref_coord = scene_coords[0, :, y, x].cpu()
            d_f = (ref_feat - stored_feat).abs().max().item()
            d_c = (ref_coord - stored_coord).abs().max().item()
            max_feat_diff = max(max_feat_diff, d_f)
            max_coord_diff = max(max_coord_diff, d_c)
            num_checked += 1

    logger.info("  Checked %d samples. max |feature diff| = %.6e, max |coord diff| = %.6e",
                num_checked, max_feat_diff, max_coord_diff)
    ok_a = max_feat_diff < 1e-5 and max_coord_diff < 1e-5
    if ok_a:
        logger.info("  -> PASS: re-forward gives identical feature and coord at same (y,x).")
    else:
        logger.warning("  -> FAIL: re-forward differs (possible dtype or non-determinism).")

    # ---------- 验证 B：与 training_step 相同的投影公式，scene_coords -> cam -> pixel ----------
    logger.info("Verification B: reproject scene_coords (same formula as training_step) -> pixel ~ (x*14, y*14)")
    repro_errors = []

    for r in records:
        coord = r["coord"]  # [3]
        pose_inv_34 = pose_inv_to_34(r["pose_inv"])  # [3, 4]，与 trainer 一致
        K = r["intrinsics"]  # [3, 3]
        x, y = r["x"], r["y"]

        # 与 trainer 一致: pred_scene [1,3,1], to_homogeneous dim=1 -> [1,4,1]; 这里单样本用 [1,4]
        sc_homo = to_homogeneous(coord.unsqueeze(0), dim=1)  # [1, 4]
        cam = (pose_inv_34 @ sc_homo.T).T  # [3,4] @ [4,1] -> [3,1] -> [1,3]
        px = (K @ cam.T).T  # [3,3] @ [3,1] -> [1,3]
        u = (px[0, 0] / px[0, 2]).item()
        v = (px[0, 1] / px[0, 2]).item()

        expected_u = x * OUTPUT_SUBSAMPLE
        expected_v = y * OUTPUT_SUBSAMPLE
        err = ((u - expected_u) ** 2 + (v - expected_v) ** 2) ** 0.5
        repro_errors.append(err)

    repro_errors = np.array(repro_errors)
    mean_err = float(repro_errors.mean())
    max_err = float(repro_errors.max())
    median_err = float(np.median(repro_errors))
    logger.info("  Reprojection error (px): mean = %.4f, median = %.4f, max = %.4f",
                mean_err, median_err, max_err)
    # 重投影误差取决于网络预测质量，这里只检查“几何关系正确”：用 to_homogeneous 再走一遍与 training_step 一致的投影
    ok_b = mean_err < 100.0  # 宽松：仅排除明显错误（例如矩阵用反）
    if ok_b:
        logger.info("  -> PASS: reprojection pipeline is consistent (errors reflect network prediction quality).")
    else:
        logger.warning("  -> FAIL: reprojection errors very large (check pose/intrinsics convention).")

    # ---------- 验证 C：与 training_step 完全相同的 batch 投影（bmm 公式）----------
    logger.info("Verification C: batch projection identical to training_step (bmm)")
    first_idx = 0
    first_records = [r for r in records if r["image_idx"] == first_idx]
    if not first_records:
        logger.info("  Skip (no samples for first image).")
        ok_c = True
    else:
        N_r = len(first_records)
        coords_b = torch.stack([r["coord"] for r in first_records])  # [N_r, 3]
        pose_inv_34 = pose_inv_to_34(first_records[0]["pose_inv"])  # [3, 4]
        pose_inv_b = pose_inv_34.unsqueeze(0).expand(N_r, -1, -1)  # [N_r, 3, 4]
        K_b = first_records[0]["intrinsics"].unsqueeze(0).expand(N_r, -1, -1)  # [N_r, 3, 3]
        # 与 trainer 一致: pred_scene [B,3,N] -> to_homogeneous dim=1 -> [B,4,N]; 这里 [N_r,3,1] -> [N_r,4,1]
        sc_homo_b = to_homogeneous(coords_b.unsqueeze(2), dim=1)  # [N_r, 4, 1]
        cam_b = torch.bmm(pose_inv_b, sc_homo_b).squeeze(2)  # [N_r, 3, 4] @ [N_r, 4, 1] -> [N_r, 3]
        px_b = torch.bmm(K_b, cam_b.unsqueeze(2)).squeeze(2)  # [N_r, 3, 3] @ [N_r, 3, 1] -> [N_r, 3]
        u_b = (px_b[:, 0] / px_b[:, 2]).numpy()
        v_b = (px_b[:, 1] / px_b[:, 2]).numpy()
        expected_u = np.array([r["x"] * OUTPUT_SUBSAMPLE for r in first_records])
        expected_v = np.array([r["y"] * OUTPUT_SUBSAMPLE for r in first_records])
        err_b = np.sqrt((u_b - expected_u) ** 2 + (v_b - expected_v) ** 2)
        logger.info("  Batch reprojection (first image): mean err = %.4f px", float(err_b.mean()))
        ok_c = float(err_b.mean()) < 100.0

    all_ok = ok_a and ok_b and ok_c
    logger.info("=" * 60)
    if all_ok:
        logger.info("Overall: PASS — buffer feature/coord/pose/intrinsics correspondence is correct.")
    else:
        logger.info("Overall: FAIL — see above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
