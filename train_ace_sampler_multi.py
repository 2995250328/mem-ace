#!/usr/bin/env python3
"""Train a universal SamplerNet across multiple scenes.

Usage:
    python train_ace_sampler_multi.py configs/sampler_7scenes_pgt.txt \
        output/universal_sampler.pt --device cuda:0
"""
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from ace_sampler.options import add_multi_sampler_train_args
from ace_sampler.multi_trainer import MultiSceneSamplerTrainer
from ace_sampler.multi_dataset import SceneEntry

_EVAL_ROOT = Path(__file__).parent / 'ace_sampler' / '04_evaluation' / 'universal'


def parse_scene_list(path: Path):
    """Parse a text file of scene entries.

    Supported formats (one per line, # = comment):
      scene_path  head_path
      scene_path  head_path  num_clusters  cluster_idx   ← Cambridge ensemble
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            assert len(parts) in (2, 4), \
                f"Expected 'scene head' or 'scene head num_clusters cluster_idx', got: {line!r}"
            if len(parts) == 4:
                entries.append(SceneEntry(
                    scene_path=Path(parts[0]), head_path=Path(parts[1]),
                    num_clusters=int(parts[2]), cluster_idx=int(parts[3])))
            else:
                entries.append(SceneEntry(
                    scene_path=Path(parts[0]), head_path=Path(parts[1])))
    return entries


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    parser = argparse.ArgumentParser(
        description='Train universal ACE SamplerNet on multiple scenes')
    add_multi_sampler_train_args(parser)
    options = parser.parse_args()

    if options.device.startswith('cuda:'):
        os.environ['CUDA_VISIBLE_DEVICES'] = options.device.split(':')[1]

    scene_entries = parse_scene_list(options.scene_list)

    # Rewrite output to eval dir with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = Path(options.sampler_output).stem
    _EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    options.sampler_output = _EVAL_ROOT / f'{timestamp}_{stem}.pt'

    logging.getLogger(__name__).info(
        f'Training on {len(scene_entries)} entries → {options.sampler_output}')

    MultiSceneSamplerTrainer(scene_entries, options).train()
