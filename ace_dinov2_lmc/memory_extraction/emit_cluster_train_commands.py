#!/usr/bin/env python3
"""Emit per-cluster training commands for a clustered memory package."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


def _sanitize_tag(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(keep).strip("._-") or "cluster"


def _resolve_clustered_package(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    preferred = path / "memory_bse.clustered.pt"
    if preferred.exists():
        return preferred

    matches = sorted(path.glob("*.clustered.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *.clustered.pt found under: {path}")
    raise ValueError(
        f"Found multiple clustered packages under {path}; pass one explicitly: "
        + ", ".join(str(p.name) for p in matches)
    )


def _parse_cluster_ids(raw: Optional[str]) -> Optional[set[int]]:
    if raw is None or not raw.strip():
        return None
    values = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.add(int(token))
    return values


def _infer_cluster_memory_path(package_path: Path, cluster: Dict[str, Any], cluster_id: int) -> Path:
    memory_path = cluster.get("memory_path")
    if isinstance(memory_path, str) and memory_path.strip():
        p = Path(memory_path)
        if not p.is_absolute():
            p = package_path.parent / p
        return p

    package_name = package_path.name
    if package_name.endswith(".clustered.pt"):
        stem = package_name[: -len(".clustered.pt")]
    else:
        stem = package_path.stem
    return package_path.parent / f"{stem}.cluster_{cluster_id:02d}.pt"


def _iter_clusters(package: Dict[str, Any], selected_ids: Optional[set[int]]) -> Iterable[Dict[str, Any]]:
    for idx, cluster in enumerate(package.get("clusters", []), start=1):
        if not isinstance(cluster, dict):
            continue
        cluster_id = int(cluster.get("cluster_id", idx))
        if selected_ids is not None and cluster_id not in selected_ids:
            continue
        yield cluster


def _build_command(args: argparse.Namespace, cluster_id: int, memory_path: Path, scene_tag: str, run_tag: str) -> str:
    repo_root = args.repo_root_abs
    output_suffix = f"{scene_tag}_{run_tag}_cluster{cluster_id:02d}.pt"
    cmd: List[str] = []
    if args.ace_data_root:
        cmd.append(f"ACE_DATA_ROOT={shlex.quote(str(Path(args.ace_data_root).expanduser()))}")
    cmd.extend(
        [
            shlex.quote(args.python),
            shlex.quote(str(repo_root / "ace_dinov2_lmc" / "train_ace_dinov2_lmc.py")),
            shlex.quote(str(args.scene_cli)),
            shlex.quote(output_suffix),
            "--train_preset",
            shlex.quote(args.train_preset),
            "--data_backend",
            shlex.quote(args.data_backend),
            "--device",
            shlex.quote(args.device),
            "--post_train_eval_device",
            shlex.quote(args.post_train_eval_device),
            "--use_lmc",
            "True",
            "--memory_path",
            shlex.quote(str(memory_path)),
        ]
    )
    if args.lmc_mode:
        cmd.extend(["--lmc_mode", shlex.quote(args.lmc_mode)])
    if args.experiment_root:
        cmd.extend(["--experiment_root", shlex.quote(str(Path(args.experiment_root).expanduser().resolve()))])
    if args.experiment_subdir:
        cmd.extend(["--experiment_subdir", shlex.quote(args.experiment_subdir)])
    if args.run_name_prefix:
        cmd.extend(["--run_name", shlex.quote(f"{args.run_name_prefix}_cluster{cluster_id:02d}")])
    if args.wai_repo_root:
        cmd.extend(["--wai_repo_root", shlex.quote(str(Path(args.wai_repo_root).expanduser().resolve()))])
    return " ".join(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-cluster train commands from a clustered memory package."
    )
    parser.add_argument("cluster_source", help="Path to memory_bse.clustered.pt or its run directory.")
    parser.add_argument("--scene", type=Path, required=True, help="Training scene path passed to train_ace_dinov2_lmc.py.")
    parser.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parents[2], help="ace_depth repo root.")
    parser.add_argument("--python", default="python", help="Python executable used in the emitted commands.")
    parser.add_argument("--train_preset", default="memory_compare_ace_g_v1", help="Training preset for emitted commands.")
    parser.add_argument("--data_backend", default="ace", choices=["ace", "wai"], help="Data backend passed to training.")
    parser.add_argument("--device", default="cuda:0", help="Training device.")
    parser.add_argument("--post_train_eval_device", default="cuda:0", help="Post-train eval device.")
    parser.add_argument("--lmc_mode", default="global", help="LMC mode passed to training.")
    parser.add_argument("--cluster_ids", default=None, help="Optional comma-separated subset, e.g. 1,2.")
    parser.add_argument("--experiment_root", default=None, help="Optional experiment root override.")
    parser.add_argument("--experiment_subdir", default="memory_pooled_vs_asb", help="Optional experiment subdir override.")
    parser.add_argument("--run_name_prefix", default=None, help="Optional fixed run_name prefix.")
    parser.add_argument("--ace_data_root", default=None, help="Optional ACE_DATA_ROOT env prefix for emitted commands.")
    parser.add_argument("--wai_repo_root", default=None, help="Optional explicit wai repo root for training.")
    args = parser.parse_args()

    args.repo_root = Path(args.repo_root).expanduser()
    if not args.repo_root.is_absolute():
        args.repo_root = (Path.cwd() / args.repo_root).resolve()
    args.repo_root_abs = args.repo_root.resolve()
    args.scene_cli = Path(args.scene).expanduser()
    if not args.scene_cli.is_absolute():
        args.scene_cli = Path.cwd() / args.scene_cli

    package_path = _resolve_clustered_package(args.cluster_source)
    payload = torch.load(package_path, map_location="cpu", weights_only=False)
    if str(payload.get("mode", "")) != "clustered":
        raise ValueError(f"Expected a clustered package, got mode={payload.get('mode')!r} from {package_path}")

    selected_ids = _parse_cluster_ids(args.cluster_ids)
    clusters = list(_iter_clusters(payload, selected_ids))
    if not clusters:
        raise ValueError("No clusters selected.")

    scene_tag = _sanitize_tag(args.scene_cli.name)
    run_tag = _sanitize_tag(package_path.parent.name)

    print(f"# package: {package_path}")
    print(f"# repo_root: {args.repo_root_abs}")
    print(f"# scene: {args.scene_cli}")
    print(f"# clusters: {len(clusters)}")
    print()
    print(f"cd {shlex.quote(str(args.repo_root_abs))}")
    print()

    emitted: List[Dict[str, Any]] = []
    for fallback_idx, cluster in enumerate(clusters, start=1):
        cluster_id = int(cluster.get("cluster_id", fallback_idx))
        meta = cluster.get("cluster_metadata") or {}
        memory_path = _infer_cluster_memory_path(package_path, cluster, cluster_id)
        if not memory_path.exists():
            raise FileNotFoundError(f"Per-cluster memory file not found: {memory_path}")

        print(
            f"# cluster {cluster_id}: views={len(meta.get('view_ids', []))}, "
            f"low_confidence={bool(meta.get('low_confidence', False))}, "
            f"failure={meta.get('failure_reason')}"
        )
        print(
            _build_command(
                args=args,
                cluster_id=cluster_id,
                memory_path=memory_path,
                scene_tag=scene_tag,
                run_tag=run_tag,
            )
        )
        print()

        emitted.append(
            {
                "cluster_id": cluster_id,
                "memory_path": str(memory_path),
                "view_count": len(meta.get("view_ids", [])),
                "low_confidence": bool(meta.get("low_confidence", False)),
                "failure_reason": meta.get("failure_reason"),
            }
        )

    print("# summary-json")
    print(json.dumps(emitted, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
