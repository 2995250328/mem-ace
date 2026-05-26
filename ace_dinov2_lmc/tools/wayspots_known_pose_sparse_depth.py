#!/usr/bin/env python3
"""Build sparse depth for an ACE-format scene using known poses.

The intended first target is Niantic Wayspots. The dataset already provides
metric poses and per-frame focal lengths, but no sparse SfM points. This script
uses COLMAP/pycolmap only for feature matching and triangulation:

1. extract or import local features into a COLMAP database,
2. replace database intrinsics with the ACE calibration files,
3. create a known-pose COLMAP text model from ACE poses,
4. triangulate points from known poses,
5. project the resulting point cloud into ACE sparse_depth npz files.

Only the selected split is used for triangulation. Use the train split for
training-time sparse depth to avoid test image leakage.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIMPLE_PINHOLE_MODEL_ID = 0


@dataclass(frozen=True)
class Frame:
    name: str
    rgb_path: Path
    pose_path: Path
    calib_path: Path
    width: int
    height: int
    focal: float
    c2w: np.ndarray


def _frame_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[-1]
    return stem


def _read_focal(path: Path) -> float:
    calib = np.loadtxt(path).astype(np.float64)
    flat = np.asarray(calib).reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    if calib.shape == (3, 3):
        return float(calib[0, 0])
    raise ValueError(f"Unsupported calibration shape {calib.shape} at {path}")


def _load_frames(split_root: Path, max_images: int | None) -> list[Frame]:
    rgb_dir = split_root / "rgb"
    pose_dir = split_root / "poses"
    calib_dir = split_root / "calibration"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not pose_dir.is_dir():
        raise FileNotFoundError(pose_dir)
    if not calib_dir.is_dir():
        raise FileNotFoundError(calib_dir)

    poses = {_frame_id(p): p for p in pose_dir.iterdir() if p.is_file()}
    calibs = {_frame_id(p): p for p in calib_dir.iterdir() if p.is_file()}
    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_images is not None:
        rgb_files = rgb_files[:max_images]

    frames: list[Frame] = []
    for rgb_path in rgb_files:
        fid = _frame_id(rgb_path)
        pose_path = poses.get(fid)
        calib_path = calibs.get(fid)
        if pose_path is None or calib_path is None:
            raise FileNotFoundError(f"Missing pose/calibration for {rgb_path.name}")
        image = imageio.imread(rgb_path)
        height, width = int(image.shape[0]), int(image.shape[1])
        c2w = np.loadtxt(pose_path).astype(np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose at {pose_path}, got {c2w.shape}")
        frames.append(
            Frame(
                name=rgb_path.name,
                rgb_path=rgb_path,
                pose_path=pose_path,
                calib_path=calib_path,
                width=width,
                height=height,
                focal=_read_focal(calib_path),
                c2w=c2w,
            )
        )

    if not frames:
        raise ValueError(f"No images found under {rgb_dir}")
    return frames


def _rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to COLMAP qvec order: qw, qx, qy, qz."""
    k = np.array(
        [
            [rot[0, 0] - rot[1, 1] - rot[2, 2], 0.0, 0.0, 0.0],
            [rot[1, 0] + rot[0, 1], rot[1, 1] - rot[0, 0] - rot[2, 2], 0.0, 0.0],
            [rot[2, 0] + rot[0, 2], rot[2, 1] + rot[1, 2], rot[2, 2] - rot[0, 0] - rot[1, 1], 0.0],
            [rot[1, 2] - rot[2, 1], rot[2, 0] - rot[0, 2], rot[0, 1] - rot[1, 0], rot[0, 0] + rot[1, 1] + rot[2, 2]],
        ],
        dtype=np.float64,
    )
    k /= 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0.0:
        qvec *= -1.0
    return qvec


def _extract_features(database_path: Path, image_dir: Path, image_names: list[str], use_gpu: bool, gpu_index: str, num_threads: int) -> None:
    import pycolmap

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_PINHOLE"
    reader_options.camera_params = ""

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.num_threads = int(num_threads)
    extraction_options.use_gpu = bool(use_gpu)
    extraction_options.gpu_index = str(gpu_index)

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    pycolmap.extract_features(
        database_path,
        image_dir,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )


def _import_images(database_path: Path, image_dir: Path, image_names: list[str]) -> None:
    import pycolmap

    database_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.touch(exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "SIMPLE_PINHOLE"
    reader_options.camera_params = ""
    pycolmap.import_images(
        database_path,
        image_dir,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        image_names=image_names,
        options=reader_options,
    )


def _patch_database_cameras(database_path: Path, frames: list[Frame]) -> dict[str, int]:
    frames_by_name = {frame.name: frame for frame in frames}
    conn = sqlite3.connect(str(database_path))
    try:
        rows = conn.execute("SELECT image_id, name FROM images").fetchall()
        image_id_by_name = {str(name): int(image_id) for image_id, name in rows}
        missing = sorted(set(frames_by_name) - set(image_id_by_name))
        if missing:
            raise RuntimeError(f"{len(missing)} images missing from COLMAP database, first={missing[:3]}")

        conn.execute("DELETE FROM cameras")
        conn.execute("DELETE FROM rigs")
        conn.execute("DELETE FROM rig_sensors")
        conn.execute("DELETE FROM frames")
        conn.execute("DELETE FROM frame_data")
        for name, frame in frames_by_name.items():
            image_id = image_id_by_name[name]
            params = np.array([frame.focal, frame.width / 2.0, frame.height / 2.0], dtype=np.float64)
            conn.execute(
                "INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    image_id,
                    SIMPLE_PINHOLE_MODEL_ID,
                    frame.width,
                    frame.height,
                    sqlite3.Binary(params.tobytes()),
                    1,
                ),
            )
            conn.execute("UPDATE images SET camera_id = ? WHERE image_id = ?", (image_id, image_id))
            conn.execute(
                "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, ?)",
                (image_id, image_id, 0),
            )
            conn.execute(
                "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
                (image_id, image_id),
            )
            conn.execute(
                "INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, ?)",
                (image_id, image_id, image_id, 0),
            )
        conn.commit()
    finally:
        conn.close()
    return image_id_by_name


def _pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 == image_id2:
        raise ValueError("pair_id requires two distinct image ids")
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return 2147483647 * int(image_id1) + int(image_id2)


def _load_grayscale_image(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3:
        rgb = image[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {image.shape} at {path}")
    if gray.max() > 1.0:
        gray /= 255.0
    return gray.astype(np.float32, copy=False)


def _resize_long_edge(image: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    if max_long_edge <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image, 1.0
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for resized SuperPoint extraction.") from exc
    scale = float(max_long_edge) / float(long_edge)
    new_w = max(8, int(round(width * scale)))
    new_h = max(8, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32, copy=False), scale


def _extract_superpoint_features(
    frames: list[Frame],
    weights_path: Path,
    use_gpu: bool,
    max_keypoints: int,
    conf_thresh: float,
    nms_dist: int,
    nn_thresh: float,
    resize_max_long_edge: int,
):
    import sys
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "SuperPoint backend requires torch in the current environment. "
            "Use an env that contains both pycolmap and torch, or install torch into the COLMAP env."
        ) from exc

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from superpoint import PointTracker, SuperPointFrontend, SuperPointNet

    if not weights_path.is_file():
        raise FileNotFoundError(f"SuperPoint weights not found: {weights_path}")

    frontend = SuperPointFrontend(
        superpoint_net=SuperPointNet(),
        weights_path=str(weights_path),
        nms_dist=int(nms_dist),
        conf_thresh=float(conf_thresh),
        nn_thresh=float(nn_thresh),
        cuda=bool(use_gpu and torch.cuda.is_available()),
    )
    matcher_backend = PointTracker(max_length=2, nn_thresh=float(nn_thresh))

    feature_map: dict[str, dict[str, np.ndarray]] = {}
    for frame in tqdm(frames, desc="Extract SuperPoint"):
        image = _load_grayscale_image(frame.rgb_path)
        image_infer, scale = _resize_long_edge(image, resize_max_long_edge)
        corners, desc, _ = frontend.run(image_infer)
        if desc is None:
            desc = np.zeros((256, 0), dtype=np.float32)
        corners = corners.astype(np.float32, copy=False)
        desc = desc.astype(np.float32, copy=False)
        if max_keypoints > 0 and corners.shape[1] > max_keypoints:
            corners = corners[:, :max_keypoints]
            desc = desc[:, :max_keypoints]
        if scale != 1.0 and corners.shape[1] > 0:
            corners[0, :] /= scale
            corners[1, :] /= scale
        feature_map[frame.name] = {
            "keypoints_xy": corners[:2, :].T.astype(np.float32, copy=False),
            "scores": corners[2, :].astype(np.float32, copy=False),
            "descriptors": desc.T.astype(np.float32, copy=False),
        }
    return feature_map, matcher_backend


def _insert_keypoints_and_descriptors(
    database_path: Path,
    image_id_by_name: dict[str, int],
    feature_map: dict[str, dict[str, np.ndarray]],
    descriptor_type: int = 1,
) -> None:
    conn = sqlite3.connect(str(database_path))
    try:
        conn.execute("DELETE FROM keypoints")
        conn.execute("DELETE FROM descriptors")
        for name, image_id in image_id_by_name.items():
            features = feature_map[name]
            keypoints_xy = np.ascontiguousarray(features["keypoints_xy"], dtype=np.float32)
            descriptors = np.ascontiguousarray(features["descriptors"], dtype=np.float32)
            conn.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (
                    image_id,
                    int(keypoints_xy.shape[0]),
                    int(keypoints_xy.shape[1]) if keypoints_xy.ndim == 2 else 2,
                    sqlite3.Binary(keypoints_xy.tobytes()),
                ),
            )
            conn.execute(
                "INSERT INTO descriptors(image_id, type, rows, cols, data) VALUES (?, ?, ?, ?, ?)",
                (
                    image_id,
                    int(descriptor_type),
                    int(descriptors.shape[0]),
                    int(descriptors.shape[1]) if descriptors.ndim == 2 else 0,
                    sqlite3.Binary(descriptors.tobytes()),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _make_matching_options(use_gpu: bool, gpu_index: str, num_threads: int):
    import pycolmap

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.num_threads = int(num_threads)
    matching_options.use_gpu = bool(use_gpu)
    matching_options.gpu_index = str(gpu_index)
    return matching_options


def _write_window_pairs(frames: list[Frame], pairs_path: Path, window: int) -> int:
    pair_count = 0
    window = int(window)
    if window < 1:
        raise ValueError("--match-window must be >= 1")
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, frame_i in enumerate(frames):
            upper = min(len(frames), i + window + 1)
            for j in range(i + 1, upper):
                f.write(f"{frame_i.name} {frames[j].name}\n")
                pair_count += 1
    return pair_count


def _generate_frame_pairs(frames: list[Frame], matcher: str, match_window: int) -> list[tuple[str, str]]:
    if matcher == "exhaustive":
        return [(frames[i].name, frames[j].name) for i in range(len(frames)) for j in range(i + 1, len(frames))]
    if matcher in {"pairs", "sequential"}:
        pairs: list[tuple[str, str]] = []
        window = int(match_window)
        if window < 1:
            raise ValueError("--match-window must be >= 1")
        for i in range(len(frames)):
            upper = min(len(frames), i + window + 1)
            for j in range(i + 1, upper):
                pairs.append((frames[i].name, frames[j].name))
        return pairs
    raise ValueError(f"Unknown matcher: {matcher}")


def _match_features(
    database_path: Path,
    workspace: Path,
    frames: list[Frame],
    matcher: str,
    use_gpu: bool,
    gpu_index: str,
    exhaustive_block_size: int,
    num_threads: int,
    match_window: int,
) -> None:
    import pycolmap

    matching_options = _make_matching_options(use_gpu, gpu_index, num_threads)
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu

    if matcher == "exhaustive":
        pairing_options = pycolmap.ExhaustivePairingOptions()
        pairing_options.block_size = int(exhaustive_block_size)
        pycolmap.match_exhaustive(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif matcher == "pairs":
        pairs_path = workspace / "image_pairs.txt"
        pair_count = _write_window_pairs(frames, pairs_path, match_window)
        print(f"Matching imported frame-window pairs: {pair_count} pairs (window={match_window})")
        pairing_options = pycolmap.ImportedPairingOptions()
        pairing_options.match_list_path = pairs_path
        pairing_options.block_size = int(exhaustive_block_size)
        pycolmap.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif matcher == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = int(match_window)
        pairing_options.quadratic_overlap = True
        pairing_options.loop_detection = False
        pairing_options.num_threads = int(num_threads)
        pycolmap.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    else:
        raise ValueError(f"Unknown matcher: {matcher}")


def _match_superpoint_pairs(
    database_path: Path,
    workspace: Path,
    image_id_by_name: dict[str, int],
    feature_map: dict[str, dict[str, np.ndarray]],
    matcher_backend,
    frames: list[Frame],
    matcher: str,
    match_window: int,
    superpoint_nn_thresh: float,
) -> int:
    pairs = _generate_frame_pairs(frames, matcher, match_window)
    pairs_path = workspace / "image_pairs.txt"
    with pairs_path.open("w", encoding="utf-8") as f:
        for name1, name2 in pairs:
            f.write(f"{name1} {name2}\n")

    conn = sqlite3.connect(str(database_path))
    inserted = 0
    try:
        conn.execute("DELETE FROM matches")
        conn.execute("DELETE FROM two_view_geometries")
        for name1, name2 in tqdm(pairs, desc="Match SuperPoint pairs"):
            desc1 = feature_map[name1]["descriptors"].T.astype(np.float32, copy=False)
            desc2 = feature_map[name2]["descriptors"].T.astype(np.float32, copy=False)
            matches = matcher_backend.nn_match_two_way(desc1, desc2, float(superpoint_nn_thresh))
            if matches.shape[1] == 0:
                continue
            match_ids = np.stack(
                [matches[0, :].astype(np.uint32), matches[1, :].astype(np.uint32)],
                axis=1,
            )
            pid = _pair_id(image_id_by_name[name1], image_id_by_name[name2])
            conn.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES (?, ?, ?, ?)",
                (
                    pid,
                    int(match_ids.shape[0]),
                    int(match_ids.shape[1]),
                    sqlite3.Binary(np.ascontiguousarray(match_ids).tobytes()),
                ),
            )
            inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def _geometric_verification(database_path: Path) -> None:
    import pycolmap

    pycolmap.geometric_verification(
        database_path,
        verifier_options=pycolmap.GeometricVerifierOptions(),
        pairing_options=pycolmap.ExistingMatchedPairingOptions(),
        two_view_geometry_options=pycolmap.TwoViewGeometryOptions(),
    )


def _write_known_pose_model(model_dir: Path, frames: list[Frame], image_id_by_name: dict[str, int]) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = _build_known_pose_reconstruction(frames, image_id_by_name)
    reconstruction.write_text(str(model_dir))


def _build_known_pose_reconstruction(frames: list[Frame], image_id_by_name: dict[str, int]):
    import pycolmap

    reconstruction = pycolmap.Reconstruction()
    for frame in frames:
        image_id = image_id_by_name[frame.name]
        camera = pycolmap.Camera(
            camera_id=image_id,
            model="SIMPLE_PINHOLE",
            width=frame.width,
            height=frame.height,
            params=np.array([frame.focal, frame.width / 2.0, frame.height / 2.0], dtype=np.float64),
        )
        camera.has_prior_focal_length = True
        reconstruction.add_camera_with_trivial_rig(camera)

        image = pycolmap.Image(
            name=frame.name,
            camera_id=image_id,
            image_id=image_id,
        )
        w2c = np.linalg.inv(frame.c2w)
        cam_from_world = pycolmap.Rigid3d(w2c[:3, :4])
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)
    return reconstruction


def _triangulate(
    database_path: Path,
    image_dir: Path,
    frames: list[Frame],
    image_id_by_name: dict[str, int],
    output_model_dir: Path,
    min_angle: float,
):
    import pycolmap

    reconstruction = _build_known_pose_reconstruction(frames, image_id_by_name)
    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.min_angle = float(min_angle)
    options.mapper.fix_existing_frames = True
    options.fix_existing_frames = True
    output_model_dir.mkdir(parents=True, exist_ok=True)
    return pycolmap.triangulate_points(
        reconstruction,
        database_path,
        image_dir,
        output_model_dir,
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )


def _reconstruction_points(reconstruction) -> np.ndarray:
    points = []
    for point in reconstruction.points3D.values():
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if np.all(np.isfinite(xyz)):
            points.append(xyz)
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def _project_depth(points_world: np.ndarray, frame: Frame, max_depth_m: float) -> tuple[np.ndarray, int]:
    depth = np.zeros((frame.height, frame.width), dtype=np.float32)
    if points_world.size == 0:
        return depth, 0
    w2c = np.linalg.inv(frame.c2w)
    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 0.0) & (z < max_depth_m)
    if not np.any(valid):
        return depth, 0
    pts_cam = pts_cam[valid]
    z = z[valid]
    u = frame.focal * (pts_cam[:, 0] / z) + frame.width / 2.0
    v = frame.focal * (pts_cam[:, 1] / z) + frame.height / 2.0
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    inside = (px >= 0) & (px < frame.width) & (py >= 0) & (py < frame.height)
    if not np.any(inside):
        return depth, 0
    px = px[inside]
    py = py[inside]
    z = z[inside].astype(np.float32)
    order = np.argsort(z)
    written = 0
    for x, y, d in zip(px[order], py[order], z[order]):
        if depth[y, x] == 0.0:
            depth[y, x] = d
            written += 1
    return depth, written


def _depth_nms_order(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray, mode: str) -> np.ndarray:
    vals = depth[ys, xs]
    if mode == "near":
        return np.argsort(vals)
    if mode == "far":
        return np.argsort(-vals)
    if mode == "uniform":
        # Deterministic spatial hash avoids the near-depth bias caused by sorting only by z.
        keys = ((xs.astype(np.uint64) * 73856093) ^ (ys.astype(np.uint64) * 19349663)) & np.uint64(0xFFFFFFFF)
        return np.argsort(keys, kind="stable")
    raise ValueError(f"Unknown depth NMS rank mode: {mode}")


def _apply_depth_nms(depth: np.ndarray, radius_px: int, max_points: int, rank_mode: str) -> tuple[np.ndarray, int]:
    if radius_px <= 0 and max_points <= 0:
        return depth, int(np.count_nonzero(depth > 0))

    ys, xs = np.where(depth > 0)
    if ys.size == 0:
        return depth, 0

    order = _depth_nms_order(depth, ys, xs, rank_mode)
    ys = ys[order]
    xs = xs[order]

    radius_px = max(0, int(radius_px))
    if radius_px <= 0:
        keep = np.arange(ys.size, dtype=np.int64)
        if max_points > 0:
            keep = keep[: int(max_points)]
    else:
        r2 = float(radius_px * radius_px)
        cell_size = max(1, radius_px)
        occupied: dict[tuple[int, int], list[tuple[int, int]]] = {}
        keep_list: list[int] = []
        for i, (y, x) in enumerate(zip(ys, xs)):
            gx = int(x) // cell_size
            gy = int(y) // cell_size
            suppressed = False
            for ny in (gy - 1, gy, gy + 1):
                for nx in (gx - 1, gx, gx + 1):
                    for ky, kx in occupied.get((ny, nx), []):
                        dy = float(y - ky)
                        dx = float(x - kx)
                        if dx * dx + dy * dy <= r2:
                            suppressed = True
                            break
                    if suppressed:
                        break
                if suppressed:
                    break
            if suppressed:
                continue
            keep_list.append(i)
            occupied.setdefault((gy, gx), []).append((int(y), int(x)))
            if max_points > 0 and len(keep_list) >= int(max_points):
                break
        keep = np.asarray(keep_list, dtype=np.int64)

    out = np.zeros_like(depth)
    out[ys[keep], xs[keep]] = depth[ys[keep], xs[keep]]
    return out, int(keep.size)


def _write_sparse_depth(
    frames: list[Frame],
    points_world: np.ndarray,
    out_dir: Path,
    max_depth_m: float,
    depth_nms_radius: int = 0,
    depth_nms_max_points: int = 0,
    depth_nms_rank: str = "uniform",
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    nonempty = 0
    for frame in tqdm(frames, desc="Project sparse depth"):
        depth, written = _project_depth(points_world, frame, max_depth_m=max_depth_m)
        depth, written = _apply_depth_nms(
            depth,
            radius_px=depth_nms_radius,
            max_points=depth_nms_max_points,
            rank_mode=depth_nms_rank,
        )
        np.savez_compressed(out_dir / f"{frame.rgb_path.stem}.npz", arr_0=depth)
        total_written += written
        nonempty += int(written > 0)
    return nonempty, total_written


def build_sparse_depth(args: argparse.Namespace) -> None:
    scene_root = args.scene_root.resolve()
    split_root = scene_root / args.split
    image_dir = split_root / "rgb"
    workspace = args.workspace.resolve()
    if workspace.exists() and args.overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    frames = _load_frames(split_root, args.max_images)
    image_names = [frame.name for frame in frames]
    database_path = workspace / "database.db"
    input_model_dir = workspace / "known_pose_model"
    output_model_dir = workspace / "triangulated_model"

    print(f"Scene: {scene_root}")
    print(f"Split: {args.split}, frames: {len(frames)}")
    print(f"Workspace: {workspace}")

    if args.feature_backend == "sift":
        print("Extracting SIFT features...")
        _extract_features(database_path, image_dir, image_names, args.use_gpu, args.gpu_index, args.num_threads)
    elif args.feature_backend == "superpoint":
        print("Importing images for external SuperPoint features...")
        _import_images(database_path, image_dir, image_names)
    else:
        raise ValueError(f"Unknown feature backend: {args.feature_backend}")

    print("Patching COLMAP database cameras with ACE intrinsics...")
    image_id_by_name = _patch_database_cameras(database_path, frames)

    if args.feature_backend == "sift":
        print(f"Matching SIFT features ({args.matcher})...")
        _match_features(
            database_path,
            workspace,
            frames,
            args.matcher,
            args.use_gpu,
            args.gpu_index,
            args.exhaustive_block_size,
            args.num_threads,
            args.match_window,
        )
    else:
        print("Extracting SuperPoint features...")
        feature_map, matcher_backend = _extract_superpoint_features(
            frames=frames,
            weights_path=args.superpoint_weights,
            use_gpu=args.use_gpu,
            max_keypoints=args.superpoint_max_keypoints,
            conf_thresh=args.superpoint_conf_thresh,
            nms_dist=args.superpoint_nms_dist,
            nn_thresh=args.superpoint_nn_thresh,
            resize_max_long_edge=args.superpoint_resize_max_long_edge,
        )
        print("Importing SuperPoint keypoints/descriptors into COLMAP database...")
        _insert_keypoints_and_descriptors(database_path, image_id_by_name, feature_map)
        print(f"Matching SuperPoint features ({args.matcher})...")
        inserted_pairs = _match_superpoint_pairs(
            database_path=database_path,
            workspace=workspace,
            image_id_by_name=image_id_by_name,
            feature_map=feature_map,
            matcher_backend=matcher_backend,
            frames=frames,
            matcher=args.matcher,
            match_window=args.match_window,
            superpoint_nn_thresh=args.superpoint_nn_thresh,
        )
        print(f"Inserted raw matches for {inserted_pairs} image pairs")
        print("Running geometric verification on imported matches...")
        _geometric_verification(database_path)

    print("Writing known-pose COLMAP model...")
    _write_known_pose_model(input_model_dir, frames, image_id_by_name)

    print("Triangulating points from known poses...")
    reconstruction = _triangulate(database_path, image_dir, frames, image_id_by_name, output_model_dir, args.min_triangulation_angle)
    points_world = _reconstruction_points(reconstruction)
    print(f"Triangulated points: {points_world.shape[0]}")

    sparse_depth_dir = split_root / args.output_subdir
    nonempty, total_written = _write_sparse_depth(
        frames,
        points_world,
        sparse_depth_dir,
        args.max_depth_m,
        depth_nms_radius=args.depth_nms_radius,
        depth_nms_max_points=args.depth_nms_max_points,
        depth_nms_rank=args.depth_nms_rank,
    )
    print(f"Sparse depth written: {sparse_depth_dir}")
    print(f"Non-empty frames: {nonempty}/{len(frames)}, projected pixels: {total_written}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sparse depth for an ACE scene using known-pose pycolmap.")
    parser.add_argument("scene_root", type=Path, help="ACE scene root containing train/test folders.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-subdir", default="sparse_depth")
    parser.add_argument("--feature-backend", default="sift", choices=["sift", "superpoint"])
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--max-depth-m", type=float, default=1000.0)
    parser.add_argument("--min-triangulation-angle", type=float, default=1.0)
    parser.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "pairs", "sequential"],
                        help="Feature matching mode. Use pairs for full Wayspots scenes.")
    parser.add_argument("--match-window", type=int, default=20,
                        help="Neighbor window for --matcher pairs/sequential.")
    parser.add_argument("--exhaustive-block-size", type=int, default=50,
                        help="Block size for exhaustive/imported-pairs matching.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-index", default="-1")
    parser.add_argument("--superpoint-weights", type=Path, default=Path("superpoint_v1.pth"))
    parser.add_argument("--superpoint-max-keypoints", type=int, default=4096)
    parser.add_argument("--superpoint-conf-thresh", type=float, default=0.015)
    parser.add_argument("--superpoint-nms-dist", type=int, default=4)
    parser.add_argument("--superpoint-nn-thresh", type=float, default=0.7)
    parser.add_argument("--superpoint-resize-max-long-edge", type=int, default=1600,
                        help="Resize long edge before SuperPoint extraction; keypoints are scaled back to original image coords.")
    parser.add_argument("--depth-nms-radius", type=int, default=0,
                        help="Spatial NMS radius in pixels applied after projecting sparse depth (0 disables).")
    parser.add_argument("--depth-nms-max-points", type=int, default=0,
                        help="Maximum sparse depth points per image after NMS (0 disables cap).")
    parser.add_argument("--depth-nms-rank", default="uniform", choices=["uniform", "near", "far"],
                        help="Candidate ordering for depth-map NMS. uniform avoids depth-range bias.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_sparse_depth(args)


if __name__ == "__main__":
    main()
