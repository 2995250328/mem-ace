#!/usr/bin/env python3
"""ACE-format RGB dataset adapter for EUPE feature grids."""

from __future__ import annotations

import logging
import math

import numpy as np

from dataset_dinov2 import CamLocDatasetDINOv2

from ace_eupe.ace_network_eupe import Regressor

_logger = logging.getLogger(__name__)


class CamLocDatasetEUPE(CamLocDatasetDINOv2):
    """DINOv2 RGB dataset variant with EUPE's model-specific patch stride.

    Training and evaluation use ``mode=0``. Sparse/depth initialization modes are
    kept compatible with the parent implementation where possible.
    """

    patch_size = Regressor.OUTPUT_SUBSAMPLE

    def __init__(self, *args, output_subsample=Regressor.OUTPUT_SUBSAMPLE, **kwargs):
        mode = kwargs.get("mode", args[1] if len(args) > 1 else 0)
        if mode != 0:
            raise ValueError("CamLocDatasetEUPE currently supports mode=0 only.")
        self.output_subsample = int(output_subsample)
        super().__init__(*args, **kwargs)
        self.patch_size = self.output_subsample

    def _round_to_patch_size(self, size):
        return int(round(size / self.output_subsample) * self.output_subsample)

    def _create_prediction_grid(self):
        prediction_grid = np.zeros(
            (
                2,
                math.ceil(5000 / self.output_subsample),
                math.ceil(5000 / self.output_subsample),
            )
        )

        for x in range(prediction_grid.shape[2]):
            for y in range(prediction_grid.shape[1]):
                prediction_grid[0, y, x] = x * self.output_subsample
                prediction_grid[1, y, x] = y * self.output_subsample

        return prediction_grid
