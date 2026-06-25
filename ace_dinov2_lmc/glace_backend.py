#!/usr/bin/env python3
"""Local adapter for loading GLACE components from the sibling project.

This module keeps all dynamic loading logic in one place so the current
ace_dinov2_lmc codebase can treat GLACE as an optional backend without
rewriting the external project as an importable package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import torch


_MODULE_CACHE: dict[tuple[str, str], Any] = {}


class GLACEDecoderFeatureResidualAdapter(torch.nn.Module):
    """Conservative residual adapter for GLACE decoder features.

    The adapter keeps the original GLACE decoder-feature path as the identity
    initialization and only learns a bounded residual toward the LMC-fused
    feature space. This preserves the original GLACE head/global-feature logic
    and makes the worst-case behavior equivalent to vanilla GLACE features.
    """

    def __init__(
        self,
        residual_gate_init: float = 0.0,
        *,
        mode: str = "decoder_delta_tanh_scalar",
        global_dim: int = 0,
    ):
        super().__init__()
        if mode not in ("decoder_delta_tanh_scalar", "local_delta_tanh_scalar"):
            raise ValueError(f"Unsupported GLACE residual adapter mode: {mode!r}")
        self.mode = mode
        self.global_dim = max(0, int(global_dim))
        self.residual_logit = torch.nn.Parameter(torch.tensor(float(residual_gate_init), dtype=torch.float32))

    def residual_gain(self) -> torch.Tensor:
        return torch.tanh(self.residual_logit)

    def forward(self, base_decoder_features: torch.Tensor, fused_decoder_features: torch.Tensor) -> torch.Tensor:
        if tuple(base_decoder_features.shape) != tuple(fused_decoder_features.shape):
            raise ValueError(
                f"GLACE residual adapter shape mismatch: base={tuple(base_decoder_features.shape)} "
                f"fused={tuple(fused_decoder_features.shape)}"
            )
        gain = self.residual_gain().to(device=base_decoder_features.device, dtype=base_decoder_features.dtype)
        if self.mode == "local_delta_tanh_scalar" and self.global_dim > 0:
            if base_decoder_features.shape[-1] <= self.global_dim:
                raise ValueError(
                    f"GLACE local residual adapter needs feature dim > global_dim, "
                    f"got feature_dim={base_decoder_features.shape[-1]} global_dim={self.global_dim}"
                )
            global_part = base_decoder_features[..., : self.global_dim]
            base_local = base_decoder_features[..., self.global_dim :]
            fused_local = fused_decoder_features[..., self.global_dim :]
            mixed_local = base_local + gain * (fused_local - base_local)
            return torch.cat((global_part, mixed_local), dim=-1)
        return base_decoder_features + gain * (fused_decoder_features - base_decoder_features)


# Backward-compatible alias for earlier local-feature-only naming.
GLACELocalFeatureResidualAdapter = GLACEDecoderFeatureResidualAdapter

def _torch_load_trusted_checkpoint(path: Path | str, *, map_location: str | torch.device = "cpu"):
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_python_module(module_path: Path, unique_name: str):
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_glace_module(glace_root: Path | str, module_stem: str):
    root = Path(glace_root).resolve()
    cache_key = (str(root), module_stem)
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module_path = root / f"{module_stem}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing GLACE module: {module_path}")

    unique_name = f"_glace_ext_{module_stem}_{abs(hash((str(root), module_stem)))}"
    if module_stem == "dataset":
        ace_network_module = _load_glace_module(root, "ace_network")
        previous = sys.modules.get("ace_network")
        try:
            sys.modules["ace_network"] = ace_network_module
            module = _load_python_module(module_path, unique_name)
        finally:
            if previous is None:
                sys.modules.pop("ace_network", None)
            else:
                sys.modules["ace_network"] = previous
    else:
        module = _load_python_module(module_path, unique_name)

    _MODULE_CACHE[cache_key] = module
    return module


def get_glace_regressor_class(glace_root: Path | str):
    module = _load_glace_module(glace_root, "ace_network")
    return module.Regressor


def get_glace_head_class(glace_root: Path | str):
    module = _load_glace_module(glace_root, "ace_network")
    return module.Head


def get_glace_dataset_class(glace_root: Path | str):
    module = _load_glace_module(glace_root, "dataset")
    return module.CamLocDataset


def load_glace_encoder_state_dict(
    encoder_path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    state = _torch_load_trusted_checkpoint(Path(encoder_path), map_location=map_location)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected GLACE encoder checkpoint type: {type(state).__name__}")
    return state


def build_glace_camloc_dataset(
    *,
    glace_root: Path | str,
    root_dir: Path | str,
    mode: int = 0,
    sparse: bool = False,
    augment: bool = False,
    aug_rotation: float = 0.0,
    aug_scale_min: float = 1.0,
    aug_scale_max: float = 1.0,
    image_height: int = 480,
    use_half: bool = False,
    num_clusters: int | None = None,
    cluster_idx: int | None = None,
    feat_name: str = "features.npy",
):
    dataset_cls = get_glace_dataset_class(glace_root)
    return dataset_cls(
        root_dir=root_dir,
        mode=mode,
        sparse=sparse,
        augment=augment,
        aug_rotation=aug_rotation,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        image_height=image_height,
        use_half=use_half,
        num_clusters=num_clusters,
        cluster_idx=cluster_idx,
        feat_name=feat_name,
    )


def create_glace_regressor_from_encoder(
    *,
    glace_root: Path | str,
    encoder_path: Path | str,
    mean,
    num_head_blocks: int,
    use_homogeneous: bool,
    global_feat_dim: int,
    head_channels: int = 512,
    mlp_ratio: float = 1.0,
    map_location: str | torch.device = "cpu",
):
    regressor_cls = get_glace_regressor_class(glace_root)
    encoder_state_dict = load_glace_encoder_state_dict(encoder_path, map_location=map_location)
    return regressor_cls.create_from_encoder(
        encoder_state_dict,
        mean=mean,
        num_head_blocks=num_head_blocks,
        use_homogeneous=use_homogeneous,
        global_feat_dim=global_feat_dim,
        head_channels=head_channels,
        mlp_ratio=mlp_ratio,
    )


def create_glace_regressor_from_split_state_dict(
    *,
    glace_root: Path | str,
    encoder_path: Path | str,
    head_state_dict,
    map_location: str | torch.device = "cpu",
):
    regressor_cls = get_glace_regressor_class(glace_root)
    encoder_state_dict = load_glace_encoder_state_dict(encoder_path, map_location=map_location)
    return regressor_cls.create_from_split_state_dict(encoder_state_dict, head_state_dict)
