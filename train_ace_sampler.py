#!/usr/bin/env python3
"""Phase 1 entry: train SamplerNet e2e from reprojection error."""
import argparse
import logging

from ace_sampler.options import add_sampler_train_args
from ace_sampler.trainer import SamplerTrainer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description='Train ACE SamplerNet (Phase 1)')
    add_sampler_train_args(parser)
    options = parser.parse_args()
    SamplerTrainer(options).train()
