"""Multi-scene dataset: concatenates multiple CamLocDataset instances.

Each item returns the standard 8-tuple from CamLocDataset plus a scene_idx integer,
so the trainer can select the correct scene-specific regressor head.

Supports optional spatial clustering (for Cambridge ensemble heads):
  scene_entries = [
      SceneEntry(scene_path, head_path),                        # full scene
      SceneEntry(scene_path, head_path, num_clusters=4, cluster_idx=0),  # cluster
  ]
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from dataset_origin import CamLocDataset


@dataclass
class SceneEntry:
    scene_path:   Path
    head_path:    Path
    num_clusters: Optional[int] = None
    cluster_idx:  Optional[int] = None


class MultiSceneDataset(Dataset):
    """Wraps N CamLocDataset instances and exposes them as a single flat dataset.

    Args:
        entries : list[SceneEntry]
        options : namespace with image_resolution, use_aug, aug_rotation, aug_scale,
                  use_half (same fields as single-scene trainer)
    """
    def __init__(self, entries: list, options):
        self.datasets = [
            CamLocDataset(
                root_dir=Path(e.scene_path) / 'train',
                mode=0,
                use_half=options.use_half,
                image_height=options.image_resolution,
                augment=options.use_aug,
                aug_rotation=options.aug_rotation,
                aug_scale_max=options.aug_scale,
                aug_scale_min=1.0 / options.aug_scale,
                num_clusters=e.num_clusters,
                cluster_idx=e.cluster_idx,
            )
            for e in entries
        ]

        # Build flat index: (scene_idx, local_idx) for every sample
        self._scene_idx = []
        self._local_idx = []
        for s_idx, ds in enumerate(self.datasets):
            for l_idx in range(len(ds)):
                self._scene_idx.append(s_idx)
                self._local_idx.append(l_idx)

    def __len__(self):
        return len(self._scene_idx)

    def __getitem__(self, idx):
        s_idx = self._scene_idx[idx]
        l_idx = self._local_idx[idx]
        item  = self.datasets[s_idx][l_idx]   # 8-tuple from CamLocDataset
        return (*item, s_idx)
