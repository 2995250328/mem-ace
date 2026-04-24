#!/usr/bin/env python3
"""Extract Utonia point features for a scene point cloud."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ace_pointcloud_fusion.pointcloud_io import load_scene_point_cloud  # noqa: E402

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build a reusable point feature bank from a COLMAP/numpy/text scene cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Scene cloud path or COLMAP scene directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt feature bank.")
    parser.add_argument("--transform_path", type=Path, default=None, help="Optional 4x4 transform, e.g. colmap_to_wai.npy.")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size before Utonia encoding. <=0 disables.")
    parser.add_argument("--max_points", type=int, default=200000, help="Maximum points before Utonia encoding. <=0 disables.")
    parser.add_argument("--seed", type=int, default=2089, help="Deterministic downsampling seed.")
    parser.add_argument("--utonia_root", type=Path, default=Path("/home/xwh/project/Utonia"), help="Utonia repo root.")
    parser.add_argument("--model", type=str, default="utonia", help="Utonia model name or local checkpoint path.")
    parser.add_argument("--repo_id", type=str, default="Pointcept/Utonia", help="HuggingFace repo id for Utonia weights.")
    parser.add_argument("--device", type=str, default="cuda", help="Encoding device.")
    parser.add_argument("--scale", type=float, default=1.0, help="Utonia transform scale.")
    parser.add_argument("--apply_z_positive", type=_strtobool, default=True, help="Use Utonia CenterShift(apply_z=True).")
    parser.add_argument("--normalize_coord", type=_strtobool, default=False, help="Use Utonia NormalizeCoord.")
    parser.add_argument("--levels_to_concat", type=int, default=2, help="How many Utonia upcast levels to concatenate.")
    parser.add_argument("--disable_flash", action="store_true", help="Load Utonia with FlashAttention disabled.")
    return parser


def main():
    args = get_parser().parse_args()
    voxel_size = args.voxel_size if args.voxel_size and args.voxel_size > 0 else None
    max_points = args.max_points if args.max_points and args.max_points > 0 else None

    cloud = load_scene_point_cloud(
        args.input,
        transform_path=args.transform_path,
        voxel_size=voxel_size,
        max_points=max_points,
        seed=args.seed,
    )
    _logger.info("Loaded cloud: points=%d source=%s", cloud["coord"].shape[0], args.input)

    from ace_pointcloud_fusion.utonia_encoder import encode_with_utonia, save_feature_bank

    bank = encode_with_utonia(
        cloud["coord"],
        cloud.get("color"),
        utonia_root=args.utonia_root,
        model_name_or_path=args.model,
        repo_id=args.repo_id,
        device=args.device,
        scale=args.scale,
        apply_z_positive=args.apply_z_positive,
        normalize_coord=args.normalize_coord,
        levels_to_concat=args.levels_to_concat,
        enable_flash=False if args.disable_flash else None,
    )
    metadata = {
        "source_path": str(args.input),
        "transform_path": str(args.transform_path) if args.transform_path else None,
        "voxel_size": voxel_size,
        "max_points": max_points,
        "utonia_root": str(args.utonia_root),
        "model": args.model,
        "repo_id": args.repo_id,
        "levels_to_concat": args.levels_to_concat,
        "cloud_points": int(cloud["coord"].shape[0]),
        "feature_dim": int(bank["features"].shape[-1]),
    }
    save_feature_bank(args.output, bank, metadata=metadata)
    _logger.info("Saved point feature bank: %s", args.output)
    _logger.info("Metadata: %s", json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
