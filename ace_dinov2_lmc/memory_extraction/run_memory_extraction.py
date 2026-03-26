"""
BSE-enhanced memory extraction for Map-Anything pipeline.
Incremental enhancement: replaces voxel pooling with BSE + adds Welford normalization.
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import Dict, List

# Add map-anything to path
MAP_ANYTHING_PATH = Path(__file__).parent.parent.parent.parent / "map-anything"
if MAP_ANYTHING_PATH.exists():
    sys.path.insert(0, str(MAP_ANYTHING_PATH))

# Map-Anything dependencies (to be preserved)
# NOTE: These imports will work at runtime when map-anything is in PYTHONPATH
# from mapanything.models import init_model
# from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices

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

    print(f"[Stage 5] BSE-enhanced memory extraction")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_path}")
    print(f"N_MEMORY: {args.n_memory}")
    print(f"BSE: {args.use_bse}, Otsu: {args.use_otsu}, voxel_size: {args.voxel_size}")

    # Initialize modules
    device = torch.device(args.device)
    welford = WelfordNormalizer()

    if args.use_bse:
        pooler = BSEPooler(
            voxel_size=args.voxel_size,
            use_otsu=args.use_otsu,
            otsu_bins=256,
            unimodal_threshold=0.02
        )
    else:
        # Fallback to vanilla voxel pooling (TODO: implement)
        raise NotImplementedError("Vanilla voxel pooling not yet implemented in Stage 5")

    # TODO: Load dataset and model (requires map-anything integration)
    # from mapanything.models import init_model
    # from mapanything.datasets import SevenScenesWAI, Indoor6WAI
    # from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices

    print("[Stage 5] TODO: Implement dataset loading, model inference, and two-pass processing")
    print("[Stage 5] Skeleton complete. Full implementation requires:")
    print("  1. Dataset loader integration")
    print("  2. Model inference (model.infer())")
    print("  3. Multi-scale feature processing")
    print("  4. Unprojection with real depth")
    print("  5. Two-pass BSE + Welford normalization")

    raise NotImplementedError("Stage 5 minimal skeleton - full implementation pending")


if __name__ == '__main__':
    main()
