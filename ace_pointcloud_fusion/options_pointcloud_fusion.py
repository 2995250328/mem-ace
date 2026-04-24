#!/usr/bin/env python3
"""Argument parser for ACE FCN + point-cloud feature fusion."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace_fcn_lmc"))

from options_ace_lmc import get_lmc_train_parser  # noqa: E402
from utils_lmc import _strtobool  # noqa: E402


def get_pointcloud_fusion_train_parser():
    parser = get_lmc_train_parser()
    parser.description = "Train ACE/DINO image features with precomputed point-cloud features and a cross-modal adaptor."

    group = parser.add_argument_group("Point-cloud fusion")
    group.add_argument(
        "--image_backbone",
        type=str,
        default="ace_fcn",
        choices=("ace_fcn", "dinov2"),
        help="Image feature extractor used before point-cloud fusion.",
    )
    group.add_argument(
        "--dinov2_path",
        type=Path,
        default=Path("/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth"),
        help="DINOv2 ViT-L/14 pretrained weights, used when --image_backbone dinov2.",
    )
    group.add_argument(
        "--point_feature_path",
        type=Path,
        required=True,
        help="Precomputed point feature bank produced by extract_pointcloud_features.py.",
    )
    group.add_argument(
        "--adaptor_hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of the ray-point cross-attention adaptor.",
    )
    group.add_argument(
        "--adaptor_heads",
        type=int,
        default=8,
        help="Number of attention heads in the cross-modal adaptor.",
    )
    group.add_argument(
        "--adaptor_dropout",
        type=float,
        default=0.1,
        help="Dropout used inside the cross-modal adaptor.",
    )
    group.add_argument(
        "--max_context_points",
        type=int,
        default=4096,
        help="Maximum scene points attended by each training batch.",
    )
    group.add_argument(
        "--train_point_adaptor",
        type=_strtobool,
        default=True,
        help="Reserved switch for future ablations; current trainer keeps adaptor trainable.",
    )
    return parser
