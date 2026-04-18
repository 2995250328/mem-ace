#!/usr/bin/env python3
"""Trainer for ACE with an EUPE backbone."""

from __future__ import annotations

import logging

from ace_eupe.ace_network_eupe import Regressor, backbone_spec
from ace_eupe.dataset_eupe import CamLocDatasetEUPE
import trainer_dinov2 as _base_trainer
from trainer_dinov2 import TrainerACEDINOv2

_logger = logging.getLogger(__name__)
_base_trainer._logger = _logger


def _feature_dim_for_model(model_name: str) -> int:
    return int(backbone_spec(model_name)["feature_dim"])


class TrainerACEEUPE(TrainerACEDINOv2):
    """ACE trainer with EUPE features and the standard ACE head-only phase."""

    def _create_regressor(self):
        feature_dim = _feature_dim_for_model(self.options.eupe_model_name)
        regressor = Regressor.create_from_encoder(
            eupe_root=self.options.eupe_root,
            eupe_checkpoint=self.options.eupe_checkpoint,
            eupe_model_name=self.options.eupe_model_name,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            num_encoder_features=feature_dim,
            freeze_backbone=self.options.freeze_backbone,
        )
        _logger.info("Loaded EUPE encoder %s from: %s", self.options.eupe_model_name, self.options.eupe_checkpoint)
        return regressor

    def _build_train_dataset(
        self,
        image_width=None,
        augment=False,
        aug_rotation=0,
        aug_scale_max=1.0,
        aug_scale_min=1.0,
    ):
        backend = getattr(self.options, "data_backend", "ace")
        if backend != "ace":
            raise ValueError("TrainerACEEUPE currently supports ACE-format datasets only.")

        return CamLocDatasetEUPE(
            root_dir=self._get_train_root(),
            mode=0,
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            image_width=image_width,
            output_subsample=int(backbone_spec(self.options.eupe_model_name)["output_subsample"]),
            augment=augment,
            aug_rotation=aug_rotation,
            aug_scale_max=aug_scale_max,
            aug_scale_min=aug_scale_min,
        )
