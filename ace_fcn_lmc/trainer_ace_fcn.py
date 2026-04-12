# trainer_ace_fcn.py
# Thin subclasses of TrainerACEDINOv2 / TrainerACEDINOv2LMC that swap the backbone
# from DINOv2 to the ACE original FCN encoder (ace_encoder_pretrained.pt).
#
# Only _create_regressor() is overridden; all training logic is inherited unchanged.

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

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

    def _evaluate_checkpoint(self, ckpt_path, iter_idx):
        """FCN 训练无 dinov2_path；用 encoder_path + test_ace_lmc.run_evaluation_lmc。"""
        from test_ace_lmc import run_evaluation_lmc

        _eval_d = self.device
        eval_device = "cuda:0" if _eval_d.type == "cuda" else str(_eval_d)
        _logger.info(
            "[Eval] iter %d: ckpt=%s  eval_device=%s",
            iter_idx + 1,
            ckpt_path,
            eval_device,
        )
        eval_opt = SimpleNamespace(
            scene=self.options.scene,
            network=ckpt_path,
            encoder_path=self.options.encoder_path,
            device=eval_device,
            image_resolution=self.options.image_resolution,
            session=f"iter_{iter_idx + 1:02d}",
            hypotheses=64,
            threshold=10,
            inlieralpha=100,
            maxpixelerror=100,
            log_per_frame=True,
        )
        return run_evaluation_lmc(eval_opt)
