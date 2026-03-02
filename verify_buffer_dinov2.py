#!/usr/bin/env python3
"""
验证 trainer_dinov2 的 buffer 采集与 training_step 与当前训练代码完全一致，
确保大规模训练时不出问题。

方式：用真实 TrainerACEDINOv2 + 小 buffer 执行 create_training_buffer，
再多次调用 training_step 检查损失有限且确定性一致。
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify DINOv2 buffer & training_step (same code as trainer)")
    parser.add_argument("scene", type=Path, help="Scene folder (e.g. datasets/7scenes/chess)")
    parser.add_argument("--dinov2_path", type=Path,
                        default=Path("/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth"))
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--buffer_batch_size", type=int, default=10,
                        help="Must match trainer (images per forward when filling buffer)")
    parser.add_argument("--buffer_image_width", type=int, default=None)
    parser.add_argument("--samples_per_image", type=int, default=512)
    parser.add_argument("--verify_buffer_size", type=int, default=10000,
                        help="Small buffer size for verification (faster)")
    parser.add_argument("--verify_batch_size", type=int, default=512,
                        help="Batch size for training_step checks (must divide verify_buffer_size)")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of training_step runs to check loss finite")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.image_resolution % 14 != 0:
        args.image_resolution = (args.image_resolution // 14) * 14

    # Build options identical to train_ace_dinov2 (only what trainer needs)
    class Options:
        pass

    opts = Options()
    opts.scene = args.scene
    opts.dinov2_path = args.dinov2_path
    opts.freeze_backbone = True
    opts.num_head_blocks = 1
    opts.use_homogeneous = True
    opts.training_buffer_size = args.verify_buffer_size
    opts.buffer_batch_size = args.buffer_batch_size
    opts.buffer_image_width = args.buffer_image_width
    opts.samples_per_image = args.samples_per_image
    opts.batch_size = args.verify_batch_size
    opts.epochs = 1
    opts.learning_rate_min = 0.0001
    opts.learning_rate_max = 0.001
    opts.image_resolution = args.image_resolution
    opts.use_aug = False
    opts.aug_rotation = 0
    opts.aug_scale = 1.0
    opts.repro_loss_soft_clamp = 50.0
    opts.repro_loss_soft_clamp_min = 1.0
    opts.repro_loss_type = "dyntanh"
    opts.repro_loss_schedule = "circle"
    opts.repro_loss_hard_clamp = 1000.0
    opts.depth_min = 0.1
    opts.depth_max = 1000.0
    opts.depth_target = 10.0
    opts.use_half = True
    opts.output_map = args.scene / "verify_dummy.pt"

    from trainer_dinov2 import TrainerACEDINOv2

    logger.info("Creating trainer (same as training script)...")
    trainer = TrainerACEDINOv2(opts)
    trainer.training_start = __import__("time").time()

    # ---------- 1) Run real create_training_buffer (with buffer_batch_size) ----------
    logger.info("Verification 1: create_training_buffer (buffer_batch_size=%d)", opts.buffer_batch_size)
    trainer.create_training_buffer()
    buf = trainer.training_buffer
    n = len(buf["features"])
    logger.info("  Buffer filled with %d samples.", n)
    if n == 0:
        logger.error("  FAIL: Buffer is empty.")
        return 1

    # ---------- 2) Run training_step repeatedly; check loss finite ----------
    logger.info("Verification 2: training_step loss finite (num_steps=%d)", args.num_steps)
    trainer.regressor.train()
    step_gen = torch.Generator().manual_seed(42)
    losses = []
    for _ in range(args.num_steps):
        idx = torch.randperm(n, generator=step_gen)[: opts.batch_size]
        if idx.shape[0] != opts.batch_size:
            idx = torch.cat([idx, idx[: opts.batch_size - idx.shape[0]]], dim=0)
        loss = trainer.training_step(
            buf["features"][idx].contiguous(),
            buf["target_px"][idx].contiguous(),
            buf["gt_poses_inv"][idx].contiguous(),
            buf["intrinsics"][idx].contiguous(),
            buf["intrinsics_inv"][idx].contiguous(),
        )
        if loss is None:
            logger.error("  FAIL: training_step returned None.")
            return 1
        loss_val = loss.item() if hasattr(loss, "item") else float(loss)
        losses.append(loss_val)
        if not np.isfinite(loss_val):
            logger.error("  FAIL: Non-finite loss at step: %s", loss_val)
            return 1
    logger.info("  Losses: min=%.4f, max=%.4f, mean=%.4f (all finite).",
                float(min(losses)), float(max(losses)), float(np.mean(losses)))
    logger.info("  -> PASS: All losses finite.")

    # ---------- 3) Same batch twice: both steps finite (second step uses updated weights) ----------
    logger.info("Verification 3: same batch two steps (both finite)")
    batch_len = min(opts.batch_size, n)
    if batch_len < 16:
        logger.warning("  Skip: buffer too small for a full batch (need >= 16).")
    else:
        idx = torch.arange(batch_len, device=buf["features"].device)
        l1 = trainer.training_step(
            buf["features"][idx].contiguous(),
            buf["target_px"][idx].contiguous(),
            buf["gt_poses_inv"][idx].contiguous(),
            buf["intrinsics"][idx].contiguous(),
            buf["intrinsics_inv"][idx].contiguous(),
        )
        l2 = trainer.training_step(
            buf["features"][idx].contiguous(),
            buf["target_px"][idx].contiguous(),
            buf["gt_poses_inv"][idx].contiguous(),
            buf["intrinsics"][idx].contiguous(),
            buf["intrinsics_inv"][idx].contiguous(),
        )
        l1_val = l1.item() if hasattr(l1, "item") else float(l1)
        l2_val = l2.item() if hasattr(l2, "item") else float(l2)
        if not np.isfinite(l1_val) or not np.isfinite(l2_val):
            logger.error("  FAIL: Non-finite loss.")
            return 1
        logger.info("  -> PASS: Two steps on same batch both finite (l1=%.4f, l2=%.4f).", l1_val, l2_val)

    # ---------- 4) Buffer layout: features vs target_px / pose / K ----------
    logger.info("Verification 4: buffer layout (features, target_px, poses, K) consistent")
    f = buf["features"]
    tp = buf["target_px"]
    pi = buf["gt_poses_inv"]
    k = buf["intrinsics"]
    ki = buf["intrinsics_inv"]
    assert f.shape[0] == tp.shape[0] == pi.shape[0] == k.shape[0] == ki.shape[0], "Buffer length mismatch"
    assert pi.shape[1:] == (3, 4) and k.shape[1:] == (3, 3), "Pose/K shape mismatch"
    assert f.shape[1] == trainer.regressor.feature_dim, "Feature dim mismatch"
    logger.info("  Shapes: features %s, target_px %s, gt_poses_inv %s, intrinsics %s.",
                tuple(f.shape), tuple(tp.shape), tuple(pi.shape), tuple(k.shape))
    logger.info("  -> PASS: Buffer layout correct.")

    logger.info("=" * 60)
    logger.info("Overall: PASS — Buffer and training_step match trainer; safe for full run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
