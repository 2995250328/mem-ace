#!/usr/bin/env python3
"""Audit Cambridge ACE-format scenes for LMC/GLACE preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    if not path.is_dir():
        return -1
    suffixes_l = tuple(s.lower() for s in suffixes)
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes_l)


def _file_stems(path: Path, suffixes: tuple[str, ...]) -> list[str]:
    if not path.is_dir():
        return []
    suffixes_l = tuple(s.lower() for s in suffixes)
    return [p.stem for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in suffixes_l]


def _feature_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    return int(np.load(path, mmap_mode="r").shape[0])


def _scene_roots(root: Path, scenes: list[str] | None) -> list[Path]:
    if scenes:
        return [root / scene for scene in scenes]
    if (root / "train" / "rgb").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "train" / "rgb").is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Cambridge ACE scene data.")
    parser.add_argument("cambridge_root", type=Path)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--require-sparse-depth", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    print("scene\tsplit\trgb\tposes\tcalibration\tfeatures\tfeature_rows\tsparse_depth\tnvm\tstem_ok")
    for scene in _scene_roots(args.cambridge_root.resolve(), args.scenes):
        if not scene.is_dir():
            failures.append(f"missing scene: {scene}")
            continue
        nvm_ok = (scene / "reconstruction.nvm").is_file()
        if not nvm_ok:
            failures.append(f"{scene.name}: missing reconstruction.nvm")
        for split in ("train", "test"):
            split_root = scene / split
            rgb_stems = _file_stems(split_root / "rgb", (".png", ".jpg", ".jpeg"))
            pose_stems = _file_stems(split_root / "poses", (".txt",))
            calib_stems = _file_stems(split_root / "calibration", (".txt",))
            sparse_stems = _file_stems(split_root / "sparse_depth", (".npz", ".npy"))
            rgb = len(rgb_stems) if (split_root / "rgb").is_dir() else -1
            poses = len(pose_stems) if (split_root / "poses").is_dir() else -1
            calib = len(calib_stems) if (split_root / "calibration").is_dir() else -1
            feat_path = split_root / "features.npy"
            features = feat_path.is_file()
            feature_rows = _feature_rows(feat_path)
            sparse_depth = len(sparse_stems) if (split_root / "sparse_depth").is_dir() else -1
            stem_ok = rgb_stems == pose_stems == calib_stems
            print(
                f"{scene.name}\t{split}\t{rgb}\t{poses}\t{calib}\t"
                f"{int(features)}\t{feature_rows}\t{sparse_depth}\t{int(nvm_ok)}\t{int(stem_ok)}"
            )
            if rgb <= 0:
                failures.append(f"{scene.name}/{split}: no RGB files")
            if rgb != poses or rgb != calib:
                failures.append(
                    f"{scene.name}/{split}: count mismatch rgb={rgb} poses={poses} calibration={calib}"
                )
            if not stem_ok:
                failures.append(f"{scene.name}/{split}: RGB/pose/calibration stems are not aligned")
            if args.require_features and not features:
                failures.append(f"{scene.name}/{split}: missing features.npy")
            if features and feature_rows != rgb:
                failures.append(
                    f"{scene.name}/{split}: features.npy row mismatch rows={feature_rows} rgb={rgb}"
                )
            if args.require_sparse_depth and sparse_depth != rgb:
                failures.append(
                    f"{scene.name}/{split}: sparse_depth count mismatch sparse={sparse_depth} rgb={rgb}"
                )
            if sparse_depth > 0 and set(sparse_stems) != set(rgb_stems):
                failures.append(f"{scene.name}/{split}: sparse_depth stems do not match RGB stems")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
