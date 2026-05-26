#!/usr/bin/env python3
"""Extract one NAVER Indoor scene and convert a minimal subset to ACE/WAI.

This script is intentionally scoped to data preparation only:
- staged archives stay under /data/xwh/dataset_staging/naver_indoor
- extracted raw trees stay under /data/xwh/dataset_staging/naver_indoor_extracted
- ACE outputs go to /data/xwh/NAVER_ace
- WAI outputs go to /data/xwh/NAVER_wai

The current implementation targets the public NAVER Indoor Localization
Dataset package layout used by HyundaiDepartmentStore_4F:

  <scene>_release_mapping.tar.gz
  <scene>_release_validation.tar.gz
  <scene>_release_mapping_lidar_only.tar.gz   (optional)
  <scene>_release_test.tar.gz                 (optional)

It extracts the requested archives once, parses kapture-style metadata from the
extracted directory, auto-selects one shared camera sensor across mapping and
validation, and writes a minimal ACE scene:

  <ace_root>/<scene_alias>/
    train/{rgb,poses,calibration}
    test/{rgb,poses,calibration}
    manifests/*.json

The generated ACE scene can then be converted to WAI directly via the optional
--write-wai flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraRecord:
    timestamp: str
    device_id: str
    relative_path: str


@dataclass(frozen=True)
class CameraSensor:
    sensor_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def intrinsics_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def _read_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([part.strip() for part in csv.reader([line], skipinitialspace=True).__next__()])
    return rows


def _discover_extracted_split(scene_root: Path, split_name: str) -> Path:
    candidates = sorted(scene_root.glob(f"**/release/{split_name}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find extracted split '{split_name}' under {scene_root}")
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous extracted split '{split_name}' under {scene_root}: {candidates}")
    return candidates[0]


def _discover_file(split_root: Path, relative_candidates: list[str]) -> Path:
    for rel in relative_candidates:
        candidate = split_root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"None of the candidate files exist under {split_root}: {relative_candidates}")


def _extract_archive(archive_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xzf", str(archive_path), "-C", str(extract_root)]
    subprocess.run(cmd, check=True)


def _maybe_extract_archives(scene_archive_root: Path, extract_root: Path, archive_stems: list[str]) -> dict[str, Path]:
    extracted: dict[str, Path] = {}
    for stem in archive_stems:
        archive_path = scene_archive_root / stem
        if not archive_path.is_file():
            continue
        split_key = stem.replace(".tar.gz", "")
        extracted_dir = extract_root / split_key
        marker = extracted_dir / ".done"
        if not marker.exists():
            extracted_dir.mkdir(parents=True, exist_ok=True)
            _extract_archive(archive_path, extracted_dir)
            marker.write_text(f"{archive_path}\n", encoding="utf-8")
        extracted[split_key] = extracted_dir
    if not extracted:
        raise FileNotFoundError(f"No NAVER archives found in {scene_archive_root}")
    return extracted


def _parse_sensors(path: Path) -> dict[str, CameraSensor]:
    sensors: dict[str, CameraSensor] = {}
    for row in _read_rows(path):
        if len(row) < 10:
            continue
        sensor_id = row[0]
        sensor_type = row[2].lower()
        model = row[3].upper()
        if sensor_type != "camera":
            continue
        if model != "OPENCV":
            raise ValueError(f"Unsupported NAVER camera model {model} at {path}")
        sensors[sensor_id] = CameraSensor(
            sensor_id=sensor_id,
            width=int(float(row[4])),
            height=int(float(row[5])),
            fx=float(row[6]),
            fy=float(row[7]),
            cx=float(row[8]),
            cy=float(row[9]),
        )
    if not sensors:
        raise ValueError(f"No camera sensors found in {path}")
    return sensors


def _parse_camera_records(path: Path) -> list[CameraRecord]:
    records: list[CameraRecord] = []
    for row in _read_rows(path):
        if len(row) < 3:
            continue
        records.append(CameraRecord(timestamp=row[0], device_id=row[1], relative_path=row[2]))
    if not records:
        raise ValueError(f"No camera records found in {path}")
    return records


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    qw, qx, qy, qz = q.tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _parse_trajectories(path: Path) -> dict[tuple[str, str], np.ndarray]:
    poses: dict[tuple[str, str], np.ndarray] = {}
    for row in _read_rows(path):
        if len(row) < 8:
            continue
        if len(row) >= 9:
            timestamp = row[0]
            device_id = row[1]
            qw, qx, qy, qz, tx, ty, tz = map(float, row[2:9])
        else:
            timestamp = row[0]
            device_id = ""
            qw, qx, qy, qz, tx, ty, tz = map(float, row[1:8])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _quat_to_rot(qw, qx, qy, qz)
        pose[:3, 3] = [tx, ty, tz]
        poses[(timestamp, device_id)] = pose
    if not poses:
        raise ValueError(f"No trajectories found in {path}")
    return poses


def _build_pose_lookup(poses: dict[tuple[str, str], np.ndarray]) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    by_key = dict(poses)
    by_timestamp: dict[str, np.ndarray] = {}
    counts = Counter(timestamp for timestamp, _ in poses.keys())
    for (timestamp, _), pose in poses.items():
        if counts[timestamp] == 1:
            by_timestamp[timestamp] = pose
    return by_key, by_timestamp


def _pose_for_record(
    record: CameraRecord,
    by_key: dict[tuple[str, str], np.ndarray],
    by_timestamp: dict[str, np.ndarray],
) -> np.ndarray | None:
    pose = by_key.get((record.timestamp, record.device_id))
    if pose is not None:
        return pose
    return by_timestamp.get(record.timestamp)


def _group_records_by_device(records: list[CameraRecord]) -> dict[str, list[CameraRecord]]:
    grouped: dict[str, list[CameraRecord]] = defaultdict(list)
    for record in records:
        grouped[record.device_id].append(record)
    return dict(grouped)


def _find_shared_sensor(
    train_records: dict[str, list[CameraRecord]],
    test_records: dict[str, list[CameraRecord]],
    train_sensors: dict[str, CameraSensor],
    test_sensors: dict[str, CameraSensor],
    preferred_sensor: str | None,
) -> str:
    shared = set(train_records) & set(test_records) & set(train_sensors) & set(test_sensors)
    if preferred_sensor is not None:
        if preferred_sensor not in shared:
            raise ValueError(f"Requested sensor {preferred_sensor} not shared across mapping/validation. Shared={sorted(shared)}")
        return preferred_sensor
    if not shared:
        raise ValueError("No shared camera sensor found across mapping and validation")
    return max(shared, key=lambda sensor_id: min(len(train_records[sensor_id]), len(test_records[sensor_id])))


def _safe_link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _write_pose(path: Path, pose: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, pose, fmt="%.10f")


def _write_calibration(path: Path, sensor: CameraSensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, sensor.intrinsics_matrix(), fmt="%.10f")


def _materialize_split(
    split_name: str,
    split_root: Path,
    records: list[CameraRecord],
    sensor: CameraSensor,
    poses_by_key: dict[tuple[str, str], np.ndarray],
    poses_by_timestamp: dict[str, np.ndarray],
    out_root: Path,
    max_frames: int | None,
    copy_files: bool,
) -> dict[str, object]:
    records_data_root = split_root / "sensors" / "records_data"
    kept = 0
    missing_pose = 0
    missing_image = 0
    manifest_frames: list[dict[str, object]] = []

    for index, record in enumerate(records):
        if max_frames is not None and kept >= max_frames:
            break
        image_src = records_data_root / record.relative_path
        pose = _pose_for_record(record, poses_by_key, poses_by_timestamp)
        if pose is None:
            missing_pose += 1
            continue
        if not image_src.is_file():
            missing_image += 1
            continue
        stem = f"{index:06d}_{record.timestamp}_{record.device_id}"
        rgb_dst = out_root / split_name / "rgb" / f"{stem}{image_src.suffix.lower()}"
        pose_dst = out_root / split_name / "poses" / f"{stem}.txt"
        calib_dst = out_root / split_name / "calibration" / f"{stem}.txt"
        _safe_link_or_copy(image_src, rgb_dst, copy_files=copy_files)
        _write_pose(pose_dst, pose)
        _write_calibration(calib_dst, sensor)
        manifest_frames.append(
            {
                "stem": stem,
                "timestamp": record.timestamp,
                "device_id": record.device_id,
                "image_src": str(image_src),
                "rgb_dst": str(rgb_dst),
                "pose_dst": str(pose_dst),
                "calibration_dst": str(calib_dst),
            }
        )
        kept += 1

    if kept == 0:
        raise RuntimeError(f"No usable frames materialized for split={split_name} sensor={sensor.sensor_id}")

    return {
        "split": split_name,
        "sensor_id": sensor.sensor_id,
        "frames_kept": kept,
        "missing_pose": missing_pose,
        "missing_image": missing_image,
        "frames": manifest_frames,
    }


def _summarize_candidate_counts(records_by_device: dict[str, list[CameraRecord]]) -> dict[str, int]:
    return {sensor_id: len(records) for sensor_id, records in sorted(records_by_device.items())}


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_ace_to_wai(ace_scene_root: Path, wai_root: Path, dataset_name: str, copy_files: bool) -> None:
    cmd = [
        "python",
        str(Path(__file__).with_name("convert_ace_to_wai.py")),
        str(ace_scene_root),
        str(wai_root),
        "--dataset-name",
        dataset_name,
    ]
    if copy_files:
        cmd.append("--copy-files")
    subprocess.run(cmd, check=True)


def prepare_scene(args: argparse.Namespace) -> dict[str, object]:
    scene_archive_root = args.archive_root / args.scene_name
    extract_root = args.extract_root / args.scene_name
    archives = [
        f"{args.scene_name}_release_mapping.tar.gz",
        f"{args.scene_name}_release_validation.tar.gz",
        f"{args.scene_name}_release_mapping_lidar_only.tar.gz",
        f"{args.scene_name}_release_test.tar.gz",
    ]
    extracted = _maybe_extract_archives(scene_archive_root, extract_root, archives)

    mapping_root = _discover_extracted_split(extracted[f"{args.scene_name}_release_mapping"], "mapping")
    validation_root = _discover_extracted_split(extracted[f"{args.scene_name}_release_validation"], "validation")

    mapping_sensors = _parse_sensors(_discover_file(mapping_root, ["sensors/sensors.txt"]))
    validation_sensors = _parse_sensors(_discover_file(validation_root, ["sensors/sensors.txt"]))
    mapping_records = _group_records_by_device(_parse_camera_records(_discover_file(mapping_root, ["sensors/records_camera.txt"])))
    validation_records = _group_records_by_device(_parse_camera_records(_discover_file(validation_root, ["sensors/records_camera.txt"])))
    mapping_pose_lookup = _build_pose_lookup(_parse_trajectories(_discover_file(mapping_root, ["sensors/trajectories.txt", "trajectories.txt"])))
    validation_pose_lookup = _build_pose_lookup(_parse_trajectories(_discover_file(validation_root, ["sensors/trajectories.txt", "trajectories.txt"])))

    selected_sensor = _find_shared_sensor(
        mapping_records,
        validation_records,
        mapping_sensors,
        validation_sensors,
        preferred_sensor=args.sensor_id,
    )

    ace_scene_root = args.ace_root / args.scene_alias
    if ace_scene_root.exists():
        shutil.rmtree(ace_scene_root)
    ace_scene_root.mkdir(parents=True, exist_ok=True)

    train_manifest = _materialize_split(
        "train",
        mapping_root,
        mapping_records[selected_sensor],
        mapping_sensors[selected_sensor],
        mapping_pose_lookup[0],
        mapping_pose_lookup[1],
        ace_scene_root,
        args.max_train_frames,
        args.copy_files,
    )
    test_manifest = _materialize_split(
        "test",
        validation_root,
        validation_records[selected_sensor],
        validation_sensors[selected_sensor],
        validation_pose_lookup[0],
        validation_pose_lookup[1],
        ace_scene_root,
        args.max_test_frames,
        args.copy_files,
    )

    summary = {
        "scene_name": args.scene_name,
        "scene_alias": args.scene_alias,
        "selected_sensor": selected_sensor,
        "archive_root": str(scene_archive_root),
        "extract_root": str(extract_root),
        "ace_scene_root": str(ace_scene_root),
        "copy_files": bool(args.copy_files),
        "candidates": {
            "mapping": _summarize_candidate_counts(mapping_records),
            "validation": _summarize_candidate_counts(validation_records),
        },
        "train": train_manifest,
        "test": test_manifest,
    }
    _write_manifest(ace_scene_root / "manifests" / "summary.json", summary)

    if args.write_wai:
        _run_ace_to_wai(
            ace_scene_root,
            args.wai_root,
            dataset_name=args.wai_dataset_name,
            copy_files=args.copy_files,
        )
        summary["wai_root"] = str(args.wai_root)
        _write_manifest(ace_scene_root / "manifests" / "summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one minimal NAVER Indoor scene into ACE/WAI format.")
    parser.add_argument("--scene-name", default="HyundaiDepartmentStore_4F", help="Archive folder name under archive-root.")
    parser.add_argument("--scene-alias", default="naver_hyundai_4f_minival", help="Output ACE scene directory name.")
    parser.add_argument("--archive-root", type=Path, default=Path("/data/xwh/dataset_staging/naver_indoor"), help="Root containing per-scene NAVER archives.")
    parser.add_argument("--extract-root", type=Path, default=Path("/data/xwh/dataset_staging/naver_indoor_extracted"), help="Root for extracted raw NAVER trees.")
    parser.add_argument("--ace-root", type=Path, default=Path("/data/xwh/NAVER_ace"), help="Root for converted ACE scenes.")
    parser.add_argument("--wai-root", type=Path, default=Path("/data/xwh/NAVER_wai"), help="Root for converted WAI scenes.")
    parser.add_argument("--wai-dataset-name", default="naver_indoor", help="dataset_name stored in WAI scene_meta.json.")
    parser.add_argument("--sensor-id", default=None, help="Optional fixed camera sensor id. Auto-selects the best shared id when omitted.")
    parser.add_argument("--max-train-frames", type=int, default=300, help="Cap train frames for the first minimal scene.")
    parser.add_argument("--max-test-frames", type=int, default=120, help="Cap test frames for the first minimal scene.")
    parser.add_argument("--copy-files", action="store_true", help="Copy files instead of symlinking into ACE/WAI outputs.")
    parser.add_argument("--write-wai", action="store_true", help="Also convert the generated ACE scene into WAI format.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = prepare_scene(args)
    print(json.dumps(
        {
            "scene_alias": summary["scene_alias"],
            "selected_sensor": summary["selected_sensor"],
            "train_frames": summary["train"]["frames_kept"],
            "test_frames": summary["test"]["frames_kept"],
            "ace_scene_root": summary["ace_scene_root"],
            "wai_root": summary.get("wai_root"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
