"""
BSE-enhanced memory extraction for Map-Anything pipeline.
Incremental enhancement: replaces voxel pooling with BSE + adds Welford normalization.
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, List

# Map-Anything dependencies (to be preserved)
from mapanything.models import init_model
from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices

# Local modules
from .bse_pooling import BSEPooler
from .welford_meter import WelfordNormalizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='BSE-enhanced memory extraction')

    # Dataset
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    parser.add_argument('output_path', type=str, help='Output .pt file path')
    parser.add_argument('--n_memory', type=int, default=100, help='Number of memory views')

    # BSE pooling
    parser.add_argument('--use_bse', action='store_true', help='Use BSE pooling (else voxel)')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size')
    parser.add_argument('--use_otsu', action='store_true', default=True, help='Use Otsu threshold')
    parser.add_argument('--bse_tau', type=float, default=0.90, help='Fixed tau if not using Otsu')

    # Model
    parser.add_argument('--model_config', type=str, required=True, help='Model config path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Loading dataset from {args.dataset_path}")
    print(f"Output will be saved to {args.output_path}")
    print(f"BSE pooling: {args.use_bse}, Otsu: {args.use_otsu}")

    # TODO: Implement two-pass processing
    # Pass 1: Extract + pool + accumulate statistics
    # Pass 2: Normalize + save

    raise NotImplementedError("Stage 4 skeleton - implementation in Stage 5")


if __name__ == '__main__':
    main()
