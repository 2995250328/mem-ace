#!/usr/bin/env python3
# result_manager.py
# 结构化结果管理系统
# 根据数据集、场景、算法、配置、时间等信息组织训练结果

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

_logger = logging.getLogger(__name__)


class ResultManager:
    """结构化结果管理器"""

    def __init__(self, experiment_root: Path = None):
        """
        初始化结果管理器

        Args:
            experiment_root: 实验根目录（默认 'output'）
        """
        self.experiment_root = Path(experiment_root or "output").resolve()
        self.experiment_root.mkdir(parents=True, exist_ok=True)

    def parse_scene_info(self, scene_path: Path) -> Dict[str, str]:
        """
        从场景路径解析数据集和场景信息

        Args:
            scene_path: 场景路径（如 /data/xwh/7scenes_chess 或 /data/xwh/indoor6_ace/scene3）

        Returns:
            {'dataset': str, 'scene': str}
        """
        scene_path = Path(scene_path).resolve()
        parts = scene_path.parts

        if len(parts) >= 2:
            if scene_path.name in ("train", "test", "val") and len(parts) >= 3:
                dataset = parts[-3]
                scene = parts[-2]
            else:
                dataset = parts[-2]
                scene = parts[-1]
        else:
            dataset = "default"
            scene = scene_path.name or "scene"

        return {"dataset": dataset, "scene": scene}

    def build_hierarchical_path(
        self,
        scene_path: Path,
        algorithm: str,
        config_tag: str,
        timestamp: Optional[str] = None,
    ) -> Path:
        """
        构建层级化的结果目录

        结构：
        output/
        ├── {dataset}/
        │   └── {scene}/
        │       ├── {algorithm}_baseline/
        │       │   └── {timestamp}_{config_tag}/
        │       └── {algorithm}_lmc/
        │           └── {timestamp}_{config_tag}/

        Args:
            scene_path: 场景路径
            algorithm: 算法名称（如 'ace_fcn', 'dino_ace'）
            config_tag: 配置标签（如 'vanilla_buf2.6M_K64_it25_bs5120'）
            timestamp: 时间戳（默认当前时间）

        Returns:
            结果目录路径
        """
        scene_info = self.parse_scene_info(scene_path)
        dataset = scene_info["dataset"]
        scene = scene_info["scene"]

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = (
            self.experiment_root
            / dataset
            / scene
            / algorithm
            / f"{timestamp}_{config_tag}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def save_run_metadata(
        self,
        run_dir: Path,
        args: Any,
        algorithm: str,
        scene_info: Dict[str, str],
    ) -> None:
        """
        保存运行元数据

        Args:
            run_dir: 运行目录
            args: 命令行参数
            algorithm: 算法名称
            scene_info: 场景信息
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "algorithm": algorithm,
            "dataset": scene_info["dataset"],
            "scene": scene_info["scene"],
            "scene_path": str(args.scene),
            "encoder_path": str(args.encoder_path),
            "device": args.device,
            "use_lmc": args.use_lmc,
            "image_resolution": args.image_resolution,
            "training_buffer_size": args.training_buffer_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate_max": args.learning_rate_max,
        }

        if args.use_lmc:
            metadata.update({
                "memory_path": str(args.memory_path),
                "num_latent_tokens": args.num_latent_tokens,
                "lmc_iterations": args.lmc_iterations,
                "lmc_mode": args.lmc_mode,
                "s1_learning_rate_max": args.s1_learning_rate_max,
                "s2_learning_rate_max": args.s2_learning_rate_max,
            })
        else:
            metadata.update({
                "vanilla_iterations": args.vanilla_iterations,
            })

        with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def save_training_summary(
        self,
        run_dir: Path,
        training_stats: Dict[str, Any],
    ) -> None:
        """
        保存训练摘要

        Args:
            run_dir: 运行目录
            training_stats: 训练统计信息
                {
                    'total_time': float,
                    'best_iter': int,
                    'best_score': float,
                    'best_metric': str,
                    'num_iterations': int,
                    'final_checkpoint': str,
                }
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "training_stats": training_stats,
        }

        with open(run_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 同时保存为易读的文本格式
        with open(run_dir / "training_summary.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write("\n")
            for key, value in training_stats.items():
                if isinstance(value, float):
                    f.write(f"{key:.<40} {value:.4f}\n")
                else:
                    f.write(f"{key:.<40} {value}\n")
            f.write("=" * 80 + "\n")

    def save_evaluation_results(
        self,
        run_dir: Path,
        eval_results: Dict[str, Any],
        session: str = "post_train",
    ) -> None:
        """
        保存评估结果

        Args:
            run_dir: 运行目录
            eval_results: 评估结果
                {
                    'median_rErr': float,
                    'median_tErr': float,
                    'pct5': float,
                    'pct25_5': float,
                    'pct10_5': float,
                    'pct2': float,
                    'pct1': float,
                    'avg_time': float,
                    'total_frames': int,
                }
            session: 评估 session 名称
        """
        eval_dir = run_dir / "eval_results"
        eval_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_file = eval_dir / f"{timestamp}_{session}_eval.json"

        eval_data = {
            "timestamp": datetime.now().isoformat(),
            "session": session,
            "results": eval_results,
        }

        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

        # 保存为易读的文本格式
        txt_file = eval_dir / f"{timestamp}_{session}_eval.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"EVALUATION RESULTS ({session})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {eval_data['timestamp']}\n")
            f.write("\n")
            f.write("Accuracy Metrics:\n")
            f.write(f"  25cm/5deg:  {eval_results['pct25_5']:>6.2f}%\n")
            f.write(f"  10cm/5deg:  {eval_results['pct10_5']:>6.2f}%\n")
            f.write(f"   5cm/5deg:  {eval_results['pct5']:>6.2f}%\n")
            f.write(f"   2cm/2deg:  {eval_results['pct2']:>6.2f}%\n")
            f.write(f"   1cm/1deg:  {eval_results['pct1']:>6.2f}%\n")
            f.write("\n")
            f.write("Error Metrics:\n")
            f.write(f"  Median Rotation Error: {eval_results['median_rErr']:>8.4f}°\n")
            f.write(f"  Median Translation Error: {eval_results['median_tErr']:>8.4f} cm\n")
            f.write("\n")
            f.write("Performance:\n")
            f.write(f"  Avg Time per Frame: {eval_results['avg_time'] * 1000:>8.2f} ms\n")
            f.write(f"  Total Frames: {eval_results['total_frames']:>8d}\n")
            f.write("=" * 80 + "\n")

        _logger.info("Evaluation results saved to: %s", eval_dir)

    def save_iteration_results(
        self,
        run_dir: Path,
        iteration: int,
        eval_results: Dict[str, Any],
    ) -> None:
        """
        保存单次迭代的评估结果

        Args:
            run_dir: 运行目录
            iteration: 迭代编号
            eval_results: 评估结果
        """
        iter_dir = run_dir / "iteration_results"
        iter_dir.mkdir(parents=True, exist_ok=True)

        iter_file = iter_dir / f"iter_{iteration:02d}_eval.json"

        iter_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "results": eval_results,
        }

        with open(iter_file, "w", encoding="utf-8") as f:
            json.dump(iter_data, f, indent=2, ensure_ascii=False)

    def create_results_index(self) -> None:
        """
        创建结果索引文件，便于快速查找所有训练结果
        """
        index = {}

        for dataset_dir in self.experiment_root.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name
            index[dataset] = {}

            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir():
                    continue

                scene = scene_dir.name
                index[dataset][scene] = {}

                for algo_dir in scene_dir.iterdir():
                    if not algo_dir.is_dir():
                        continue

                    algorithm = algo_dir.name
                    index[dataset][scene][algorithm] = []

                    for run_dir in algo_dir.iterdir():
                        if not run_dir.is_dir():
                            continue

                        metadata_file = run_dir / "run_metadata.json"
                        summary_file = run_dir / "training_summary.json"
                        eval_dir = run_dir / "eval_results"

                        run_info = {
                            "run_name": run_dir.name,
                            "run_path": str(run_dir),
                            "has_metadata": metadata_file.exists(),
                            "has_summary": summary_file.exists(),
                            "has_eval": eval_dir.exists(),
                        }

                        if metadata_file.exists():
                            with open(metadata_file, "r", encoding="utf-8") as f:
                                metadata = json.load(f)
                                run_info["timestamp"] = metadata.get("timestamp")
                                run_info["use_lmc"] = metadata.get("use_lmc")

                        if summary_file.exists():
                            with open(summary_file, "r", encoding="utf-8") as f:
                                summary = json.load(f)
                                run_info["best_score"] = summary.get("training_stats", {}).get("best_score")

                        index[dataset][scene][algorithm].append(run_info)

        index_file = self.experiment_root / "results_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        _logger.info("Results index created: %s", index_file)

    def print_results_summary(self) -> None:
        """打印结果摘要"""
        index_file = self.experiment_root / "results_index.json"
        if not index_file.exists():
            _logger.warning("Results index not found. Run create_results_index() first.")
            return

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        _logger.info("=" * 80)
        _logger.info("RESULTS SUMMARY")
        _logger.info("=" * 80)

        for dataset, scenes in index.items():
            _logger.info("\nDataset: %s", dataset)
            for scene, algorithms in scenes.items():
                _logger.info("  Scene: %s", scene)
                for algorithm, runs in algorithms.items():
                    _logger.info("    Algorithm: %s (%d runs)", algorithm, len(runs))
                    for run in sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)[:3]:
                        score = run.get("best_score", "N/A")
                        timestamp = run.get("timestamp", "N/A")
                        _logger.info("      - %s (score: %s, time: %s)", run["run_name"], score, timestamp)

        _logger.info("=" * 80)
