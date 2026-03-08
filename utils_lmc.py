"""Shared utility helpers for DINOv2 LMC training stack."""

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import torch

_logger = logging.getLogger(__name__)


def setup_cuda_environment():
    """在导入 torch 相关训练模块前设置 CUDA_VISIBLE_DEVICES。"""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:0')
    pre_args, _ = pre_parser.parse_known_args()
    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: CUDA_VISIBLE_DEVICES = {gpu_id}", flush=True)


def _strtobool(x):
    """把 CLI 布尔参数统一解析为 bool。"""
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {x!r}')


def _sanitize_tag(text: str) -> str:
    """Make a filesystem-safe tag for run folder names."""
    text = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(text))
    return text.strip('._-') or "run"


def _sanitize_stem(p: Path) -> str:
    """文件名安全：只保留字母数字下划线横线。"""
    s = p.stem if hasattr(p, 'stem') else str(p)
    return "".join(c for c in s if c.isalnum() or c in "._-") or "model"


def _format_buf_million(x: int) -> str:
    """Format buffer size into compact M unit for folder naming."""
    return f"{x / 1_000_000:.1f}".rstrip('0').rstrip('.')


def estimate_memory_front_visibility(
    pooled_points: torch.Tensor,
    all_poses: torch.Tensor,
    max_points: int = 4096,
) -> Dict[str, float] | None:
    """Estimate how many memory points lie in front of cameras (z>0 in camera frame)."""
    if pooled_points is None or all_poses is None:
        return None
    if not isinstance(pooled_points, torch.Tensor) or not isinstance(all_poses, torch.Tensor):
        return None
    if pooled_points.numel() == 0 or all_poses.numel() == 0:
        return None

    pts = pooled_points.detach().float().cpu()
    poses = all_poses.detach().float().cpu()
    if poses.ndim == 4 and poses.shape[1:] == (1, 4, 4):
        poses = poses[:, 0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return None

    n_pts = pts.shape[0]
    if n_pts > max_points:
        perm = torch.randperm(n_pts)[:max_points]
        pts = pts[perm]
        n_pts = pts.shape[0]

    pts_h = torch.cat([pts, torch.ones(n_pts, 1)], dim=1).t()  # (4, N)
    ratios = []
    for p in poses:
        try:
            w2c = torch.linalg.inv(p)
        except RuntimeError:
            continue
        cam = (w2c @ pts_h)[:3]
        z = cam[2]
        ratios.append(float((z > 0).float().mean().item()))

    if len(ratios) == 0:
        return None
    return {
        "mean_front_ratio": float(sum(ratios) / len(ratios)),
        "min_front_ratio": float(min(ratios)),
        "median_front_ratio": float(torch.tensor(ratios).median().item()),
        "max_front_ratio": float(max(ratios)),
        "num_points_sampled": int(n_pts),
        "num_poses": int(len(ratios)),
    }


def load_memory_features(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load memory tensors from disk (same contract as map-anything load_memory_features).
    Supports POOLED memory bank format; unknown format raises with available keys.

    Validation layers:
    - Shape: pooled_points must be (N, >=3), pooled_features (N, D), N must match and be > 0.
    - NaN/Inf: raises ValueError for required fields; warns for optional fields.
    - Recommended field warnings: scene_center, all_scale_tokens, all_poses.
    - Point cloud stats: logs num_points, feature_dim, xyz range/center.
    - _meta traceability dict added to returned dict.
    """
    t_start = time.time()
    payload = torch.load(path, map_location=device, weights_only=False)

    def safe_to_device(key: str):
        val = payload.get(key)
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return val

    if "pooled_points" in payload and "pooled_features" in payload:
        _logger.info("[LMC] Detected POOLED memory bank at %s", path)
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.tensor(x, device=device)

        pooled_points = _to_tensor(payload["pooled_points"])
        pooled_features = _to_tensor(payload["pooled_features"])

        # --- Shape validation ---
        if pooled_points.ndim != 2 or pooled_points.shape[1] < 3:
            raise ValueError(
                f"[LMC] pooled_points has invalid shape {tuple(pooled_points.shape)}; "
                f"expected (N, >=3)."
            )
        if pooled_features.ndim != 2:
            raise ValueError(
                f"[LMC] pooled_features has invalid shape {tuple(pooled_features.shape)}; "
                f"expected (N, D)."
            )
        N_pts = pooled_points.shape[0]
        N_feats = pooled_features.shape[0]
        if N_pts != N_feats:
            raise ValueError(
                f"[LMC] pooled_points has {N_pts} rows but pooled_features has {N_feats}; "
                f"they must match."
            )
        if N_pts == 0:
            raise ValueError("[LMC] pooled_points/pooled_features have 0 rows (empty memory).")

        feature_dim = pooled_features.shape[1]

        # --- NaN/Inf checks for required fields ---
        if torch.isnan(pooled_points).any():
            raise ValueError(
                f"[LMC] pooled_points contains NaN values "
                f"({int(torch.isnan(pooled_points).sum().item())} NaN elements)."
            )
        if torch.isinf(pooled_points).any():
            raise ValueError(
                f"[LMC] pooled_points contains Inf values "
                f"({int(torch.isinf(pooled_points).sum().item())} Inf elements)."
            )
        if torch.isnan(pooled_features).any():
            raise ValueError(
                f"[LMC] pooled_features contains NaN values "
                f"({int(torch.isnan(pooled_features).sum().item())} NaN elements)."
            )
        if torch.isinf(pooled_features).any():
            raise ValueError(
                f"[LMC] pooled_features contains Inf values "
                f"({int(torch.isinf(pooled_features).sum().item())} Inf elements)."
            )

        # --- NaN/Inf warnings for optional fields ---
        for opt_key in ("scene_center", "all_scale_tokens"):
            opt_val = safe_to_device(opt_key)
            if isinstance(opt_val, torch.Tensor) and opt_val.numel() > 0:
                if torch.isnan(opt_val).any():
                    _logger.warning(
                        "[LMC] %s contains NaN values (%d NaN elements).",
                        opt_key, int(torch.isnan(opt_val).sum().item()),
                    )
                if torch.isinf(opt_val).any():
                    _logger.warning(
                        "[LMC] %s contains Inf values (%d Inf elements).",
                        opt_key, int(torch.isinf(opt_val).sum().item()),
                    )

        # --- Recommended field warnings ---
        has_scene_center = (safe_to_device("scene_center") is not None)
        has_scale_tokens = (safe_to_device("all_scale_tokens") is not None)
        has_all_poses = (safe_to_device("all_poses") is not None)
        if not has_scene_center:
            _logger.warning("[LMC] Recommended field 'scene_center' is missing from memory file.")
        if not has_scale_tokens:
            _logger.warning("[LMC] Recommended field 'all_scale_tokens' is missing from memory file.")
        if not has_all_poses:
            _logger.warning("[LMC] Recommended field 'all_poses' is missing from memory file.")

        # --- Point cloud statistics ---
        xyz = pooled_points[:, :3].float()
        xyz_min = xyz.min(dim=0)[0]
        xyz_max = xyz.max(dim=0)[0]
        xyz_center = xyz.mean(dim=0)
        xyz_range = xyz_max - xyz_min
        _logger.info(
            "[LMC] Memory stats: num_points=%d, feature_dim=%d, "
            "xyz_range=(%.3f, %.3f, %.3f), xyz_center=(%.3f, %.3f, %.3f), "
            "xyz_min=(%.3f, %.3f, %.3f), xyz_max=(%.3f, %.3f, %.3f)",
            N_pts, feature_dim,
            float(xyz_range[0]), float(xyz_range[1]), float(xyz_range[2]),
            float(xyz_center[0]), float(xyz_center[1]), float(xyz_center[2]),
            float(xyz_min[0]), float(xyz_min[1]), float(xyz_min[2]),
            float(xyz_max[0]), float(xyz_max[1]), float(xyz_max[2]),
        )

        # --- Build _meta traceability dict ---
        _meta = {
            "source_path": str(path),
            "load_timestamp": time.time(),
            "num_points": int(N_pts),
            "feature_dim": int(feature_dim),
            "has_scale_tokens": has_scale_tokens,
            "has_scene_center": has_scene_center,
            "has_all_poses": has_all_poses,
            "xyz_range": [float(xyz_range[0]), float(xyz_range[1]), float(xyz_range[2])],
            "xyz_center": [float(xyz_center[0]), float(xyz_center[1]), float(xyz_center[2])],
        }

        t_elapsed = time.time() - t_start
        _logger.info("[LMC] Memory loaded in %.2fs", t_elapsed)

        return {
            "type": "pooled",
            "pooled_points": pooled_points,
            "pooled_features": pooled_features,
            "pooled_colors": safe_to_device("pooled_colors"),
            "all_poses": safe_to_device("all_poses"),
            "all_intrinsics": safe_to_device("all_intrinsics"),
            "all_scale_tokens": safe_to_device("all_scale_tokens"),
            "scene_center": safe_to_device("scene_center"),
            "ref_pose": safe_to_device("ref_pose"),
            "patch_stride": payload.get("patch_stride", 14.0),
            "voxel_size": payload.get("voxel_size", 0.05),
            "original_views": payload.get("original_views", 0),
            "scene": payload.get("scene", "unknown"),
            "layers_idx": payload.get("layers_idx", []),
            "_meta": _meta,
        }
    if "intermediate" in payload or "final" in payload:
        raise ValueError(
            f"Memory file at {path} is FULL intermediate format. "
            "ace_depth LMC only supports POOLED memory bank (pooled_points, pooled_features)."
        )
    available = list(payload.keys()) if isinstance(payload, dict) else "Not a dict"
    raise ValueError(f"Unknown memory file format at {path}. Keys: {available}")


def preflight_memory_features(
    bank_data: Dict[str, Any],
    *,
    strict: bool = False,
    expect_scale_tokens: bool = True,
    scene_center_tol: float = 1.0,
) -> Dict[str, Any]:
    """Run readonly consistency checks on pooled memory before training starts."""
    issues = []
    stats: Dict[str, Any] = {}

    def _record(ok: bool, message: str):
        if ok:
            _logger.info("[LMC preflight] %s", message)
        else:
            issues.append(message)
            if strict:
                raise ValueError(f"[LMC preflight] {message}")
            _logger.warning("[LMC preflight] %s", message)

    pooled_points = bank_data.get("pooled_points")
    pooled_features = bank_data.get("pooled_features")
    if not isinstance(pooled_points, torch.Tensor) or not isinstance(pooled_features, torch.Tensor):
        _record(False, "pooled_points/pooled_features must both be torch.Tensor.")
        return {"ok": len(issues) == 0, "issues": issues, "stats": stats}

    n_points = int(pooled_points.shape[0])
    feature_dim = int(pooled_features.shape[1]) if pooled_features.ndim == 2 else -1
    stats["num_points"] = n_points
    stats["feature_dim"] = feature_dim

    scene_center = bank_data.get("scene_center")
    if scene_center is None:
        _record(False, "scene_center is missing; trainer will fallback to mean(pooled_points).")
    elif not isinstance(scene_center, torch.Tensor):
        _record(False, f"scene_center must be a torch.Tensor, got {type(scene_center).__name__}.")
    else:
        sc = scene_center.detach().float().view(-1).cpu()
        if sc.numel() < 3:
            _record(False, f"scene_center shape is invalid: got {tuple(scene_center.shape)}, expected at least 3 values.")
        else:
            pooled_mean = pooled_points[:, :3].detach().float().mean(dim=0).cpu()
            center_dist = float(torch.linalg.norm(sc[:3] - pooled_mean).item())
            stats["scene_center_dist_to_pooled_mean"] = center_dist
            _record(
                center_dist <= scene_center_tol,
                f"scene_center distance to mean(pooled_points) = {center_dist:.4f} m (tol={scene_center_tol:.4f} m).",
            )

    layers_idx = bank_data.get("layers_idx")
    if isinstance(layers_idx, (list, tuple)) and len(layers_idx) > 0:
        stats["num_layers"] = int(len(layers_idx))
        if feature_dim > 0:
            divisible = (feature_dim % len(layers_idx) == 0)
            _record(
                divisible,
                f"pooled_features dim={feature_dim} {'is' if divisible else 'is not'} divisible by len(layers_idx)={len(layers_idx)}.",
            )
        if str(layers_idx[-1]) != "final":
            _record(False, f"layers_idx last entry is {layers_idx[-1]!r}, expected 'final' from extractor contract.")
    else:
        _record(False, "layers_idx is missing/empty; trainer will fallback to default num_layers=4.")

    original_views = bank_data.get("original_views", 0)
    try:
        original_views = int(original_views)
    except Exception:
        original_views = 0
    stats["original_views"] = original_views

    def _normalize_view_count(name: str, value: Any):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            _record(False, f"{name} must be a torch.Tensor when present, got {type(value).__name__}.")
            return None
        if value.numel() == 0:
            _record(False, f"{name} is present but empty.")
            return None
        if name == "all_poses":
            ok = value.ndim in (3, 4) and tuple(value.shape[-2:]) == (4, 4)
        elif name == "all_intrinsics":
            ok = value.ndim in (3, 4) and tuple(value.shape[-2:]) == (3, 3)
        else:
            ok = value.ndim == 2
        _record(ok, f"{name} shape={tuple(value.shape)}.")
        return int(value.shape[0]) if ok else None

    pose_views = _normalize_view_count("all_poses", bank_data.get("all_poses"))
    intr_views = _normalize_view_count("all_intrinsics", bank_data.get("all_intrinsics"))
    scale_views = _normalize_view_count("all_scale_tokens", bank_data.get("all_scale_tokens"))

    if expect_scale_tokens and bank_data.get("all_scale_tokens") is None:
        _record(False, "use_scale_token=True but all_scale_tokens is missing.")

    known_view_counts = [v for v in (pose_views, intr_views, scale_views) if v is not None]
    if original_views > 0:
        for name, count in (("all_poses", pose_views), ("all_intrinsics", intr_views), ("all_scale_tokens", scale_views)):
            if count is not None:
                _record(
                    count == original_views,
                    f"{name} view count={count} {'matches' if count == original_views else 'does not match'} original_views={original_views}.",
                )
    elif known_view_counts:
        baseline = known_view_counts[0]
        for name, count in (("all_poses", pose_views), ("all_intrinsics", intr_views), ("all_scale_tokens", scale_views)):
            if count is not None:
                _record(
                    count == baseline,
                    f"{name} view count={count} {'matches' if count == baseline else 'does not match'} peer view count={baseline}.",
                )

    for scalar_key in ("patch_stride", "voxel_size"):
        val = bank_data.get(scalar_key)
        if val is None:
            continue
        try:
            val_f = float(val)
        except Exception:
            _record(False, f"{scalar_key}={val!r} is not numeric.")
            continue
        stats[scalar_key] = val_f
        _record(val_f > 0, f"{scalar_key}={val_f:.4f} should be > 0.")

    ref_pose = bank_data.get("ref_pose")
    if ref_pose is not None:
        if not isinstance(ref_pose, torch.Tensor) or tuple(ref_pose.shape[-2:]) != (4, 4):
            _record(False, f"ref_pose shape is invalid: got {getattr(ref_pose, 'shape', None)}.")
        else:
            _record(True, f"ref_pose shape={tuple(ref_pose.shape)}.")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "stats": stats,
    }


def _normalize_scene_tag(scene: str | Path | None) -> str:
    """Normalize scene identifiers for robust memory-vs-train matching."""
    if scene is None:
        return ""
    s = str(scene).strip()
    if not s:
        return ""
    p = Path(s)
    name = p.name
    if name in ("train", "val", "test"):
        name = p.parent.name
    if name.endswith(("_train", "_val", "_test")):
        name = name.rsplit("_", 1)[0]
    return name
