#!/usr/bin/env python3
"""Losslessly compress WAI numpy modalities (.npy -> .npz) and rewrite scene_meta paths.

The WAI loader already supports both .npy and .npz for `format: "numpy"`.
This script converts one modality directory in-place, verifies exact equality
after reload, updates frame paths in `scene_meta.json`, and can optionally
delete the original .npy files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _iter_scene_roots(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "scene_meta.json").exists()])


def _compress_file(src: Path, dst: Path) -> None:
    arr = np.load(src, allow_pickle=False)
    np.savez_compressed(dst, arr_0=arr)
    arr_reloaded = np.load(dst, allow_pickle=False)["arr_0"]
    if arr.shape != arr_reloaded.shape or arr.dtype != arr_reloaded.dtype or not np.array_equal(arr, arr_reloaded):
        raise RuntimeError(f"Verification failed for {src} -> {dst}")


def _rewrite_scene(scene_root: Path, modality: str, delete_original: bool, dry_run: bool) -> dict[str, int]:
    meta_path = scene_root / "scene_meta.json"
    original_meta_text = meta_path.read_text(encoding="utf-8")
    scene_meta = json.loads(original_meta_text)
    frames = scene_meta.get("frames", [])
    changed = 0
    converted = 0
    skipped = 0

    for frame in frames:
        rel = frame.get(modality)
        if not rel:
            continue
        rel_path = Path(rel)
        if rel_path.suffix == ".npz":
            skipped += 1
            continue
        if rel_path.suffix != ".npy":
            raise RuntimeError(f"Unexpected {modality} path suffix in {meta_path}: {rel}")
        src = scene_root / rel_path
        dst_rel = rel_path.with_suffix(".npz")
        dst = scene_root / dst_rel
        if not src.exists():
            raise FileNotFoundError(src)
        if not dry_run:
            _compress_file(src, dst)
            if delete_original:
                src.unlink()
        frame[modality] = dst_rel.as_posix()
        changed += 1
        converted += 1

    if changed > 0 and not dry_run:
        backup = meta_path.with_suffix(meta_path.suffix + ".bak_before_npz")
        if not backup.exists():
            backup.write_text(original_meta_text, encoding="utf-8")
        meta_path.write_text(json.dumps(scene_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"converted": converted, "changed": changed, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wai-root", type=Path, required=True, help="WAI dataset root, e.g. /data/.../wai_data/indoor6")
    parser.add_argument("--modality", type=str, default="gt_depth", choices=["gt_depth", "colmap_depth"], help="Modality to compress")
    parser.add_argument("--scene", type=str, default=None, help="Only process one scene, e.g. scene1_train")
    parser.add_argument("--delete-original", action="store_true", help="Delete source .npy after successful .npz verification")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing files")
    args = parser.parse_args()

    if args.scene:
        scene_roots = [args.wai_root / args.scene]
    else:
        scene_roots = _iter_scene_roots(args.wai_root)

    total_converted = 0
    total_changed = 0
    total_skipped = 0
    for scene_root in scene_roots:
        if not scene_root.exists():
            raise FileNotFoundError(scene_root)
        stats = _rewrite_scene(scene_root, args.modality, args.delete_original, args.dry_run)
        total_converted += stats["converted"]
        total_changed += stats["changed"]
        total_skipped += stats["skipped"]
        print(
            f"{scene_root.name}: modality={args.modality} converted={stats['converted']} "
            f"changed={stats['changed']} skipped={stats['skipped']}"
        )

    print(
        f"total: modality={args.modality} converted={total_converted} "
        f"changed={total_changed} skipped={total_skipped} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
