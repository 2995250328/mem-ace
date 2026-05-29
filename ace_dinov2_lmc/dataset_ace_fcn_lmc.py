"""ACE-FCN dataset wrappers for the LMC trainer.

This keeps the original ACE preprocessing/stride-8 target grid while optionally
returning GLACE image-level features aligned by dataset index.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from dataset_origin import CamLocDataset


_logger = logging.getLogger(__name__)


class CamLocDatasetACEFCNLMC(CamLocDataset):
    """Original ACE dataset with optional per-image global features."""

    def __init__(self, *args, feat_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_feats = None
        self.global_feat_dim = 0
        if feat_name:
            root_dir = Path(args[0] if args else kwargs["root_dir"])
            feat_path = root_dir / feat_name
            self.global_feats = np.load(feat_path)
            if self.global_feats.shape[0] != len(self.rgb_files):
                raise ValueError(
                    f"Global feature count mismatch: {feat_path} has {self.global_feats.shape[0]} "
                    f"rows but dataset has {len(self.rgb_files)} RGB files."
                )
            self.global_feat_dim = int(self.global_feats.shape[1])
            _logger.info("Loaded ACE-FCN global features from %s dim=%d", feat_path, self.global_feat_dim)

    def _get_single_item(self, idx, image_height):
        item = super()._get_single_item(idx, image_height)
        if self.global_feats is None:
            return item
        real_idx = int(self.valid_file_indices[int(idx)])
        global_feat = torch.from_numpy(self.global_feats[real_idx]).float()
        return (*item, global_feat, real_idx)
