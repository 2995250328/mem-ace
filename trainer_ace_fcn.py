# trainer_ace_fcn.py
# Thin subclasses of TrainerACEDINOv2 / TrainerACEDINOv2LMC that swap the backbone
# from DINOv2 to the ACE original FCN encoder (ace_encoder_pretrained.pt).
#
# Only _create_regressor() is overridden; all training logic is inherited unchanged.

import logging

from trainer_dinov2 import TrainerACEDINOv2
from trainer_dinov2_lmc import TrainerACEDINOv2LMC

_logger = logging.getLogger(__name__)


class TrainerACEFCN(TrainerACEDINOv2):
    """ACE trainer using the original FCN encoder instead of DINOv2."""

    def _create_regressor(self):
        from ace_network_ace import RegressorACE
        regressor = RegressorACE.create_from_encoder(
            encoder_path=self.options.encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            num_encoder_features=512,
            freeze_backbone=self.options.freeze_backbone,
        )
        _logger.info("Loaded ACE FCN encoder from: %s", self.options.encoder_path)
        return regressor


class TrainerACEFCNLMC(TrainerACEDINOv2LMC):
    """ACE + GeoLMC trainer using the original FCN encoder instead of DINOv2."""

    def _create_regressor(self):
        from ace_network_ace import RegressorACE
        regressor = RegressorACE.create_from_encoder(
            encoder_path=self.options.encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            num_encoder_features=512,
            freeze_backbone=self.options.freeze_backbone,
        )
        _logger.info("Loaded ACE FCN encoder from: %s", self.options.encoder_path)
        return regressor
