#!/usr/bin/env python3
"""Phase 1 entry: train SamplerNet e2e from reprojection error."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

from ace_sampler.options import add_sampler_train_args
from ace_sampler.trainer import SamplerTrainer

_EVAL_ROOT = Path(__file__).parent / 'ace_sampler' / '04_evaluation'


def resolve_sampler_output_path(scene: Path, requested_output: Path):
    """
    Rewrite sampler_output to ace_sampler/04_evaluation/{dataset}/{scene}/{timestamp}_{stem}.pt

    Returns (run_dir, output_path).
    """
    scene = Path(scene).resolve()
    parts = scene.parts
    if len(parts) >= 2:
        dataset = parts[-2]
        scene_name = parts[-1]
    else:
        dataset = 'default'
        scene_name = scene.name or 'scene'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = Path(requested_output).stem
    run_dir = _EVAL_ROOT / dataset / scene_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f'{timestamp}_{stem}.pt'
    return run_dir, output_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description='Train ACE SamplerNet (Phase 1)')
    add_sampler_train_args(parser)
    options = parser.parse_args()

    run_dir, options.sampler_output = resolve_sampler_output_path(options.scene, options.sampler_output)
    logging.getLogger(__name__).info('Run dir: %s', run_dir)
    logging.getLogger(__name__).info('Sampler output: %s', options.sampler_output)

    SamplerTrainer(options).train()
