"""CPU smoke test for the cleaned ACE-G paper implementation."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from aceg_method import (
    ACEGTwoStageController,
    GlobalGeoLMC,
    MemoryEnhancedSCR,
    MinimalCoordinateHead,
    ReprojectionConfig,
    SingleMemoryFusion,
    TrackBatch,
    inter_frame_track_reprojection_loss,
    load_scene_memory,
    load_stgs_sidecar,
    make_scene_memory,
    save_scene_memory,
    single_frame_reprojection_loss,
)


def camera_batch(count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = torch.eye(4).repeat(count, 1, 1)[:, :3]
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    ).repeat(count, 1, 1)
    return world_to_camera, intrinsics, torch.linalg.inv(intrinsics)


def test_model_and_schedule() -> None:
    torch.manual_seed(7)
    memory = make_scene_memory(torch.randn(40, 3), torch.randn(40, 24))
    model = MemoryEnhancedSCR(
        compressor=GlobalGeoLMC(
            memory_feature_dim=24,
            token_dim=32,
            num_tokens=8,
            num_attention_layers=2,
            num_heads=4,
            dropout=0.0,
        ),
        fusion=SingleMemoryFusion(
            query_feature_dim=16,
            token_dim=32,
            output_dim=32,
            num_heads=4,
            dropout=0.0,
        ),
        coordinate_head=MinimalCoordinateHead(32),
    )
    controller = ACEGTwoStageController(model)
    query = torch.randn(2, 16, 4, 5)

    controller.enter_stage1()
    stage1_coordinates = controller.stage1_forward(query, memory)
    assert stage1_coordinates.shape == (2, 3, 4, 5)
    stage1_coordinates.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.compressor.parameters())

    controller.enter_stage2_global(memory)
    assert all(not parameter.requires_grad for parameter in model.compressor.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    stage2_coordinates = controller.stage2_global_forward(query)
    assert stage2_coordinates.shape == (2, 3, 4, 5)


def test_reprojection_losses() -> None:
    points = torch.tensor([[0.0, 0.0, 5.0], [0.5, -0.25, 4.0]])
    world_to_camera, intrinsics, inverse_intrinsics = camera_batch(2)
    source_pixels = torch.tensor([[50.0, 40.0], [62.5, 33.75]])
    config = ReprojectionConfig()
    single_loss, single_stats = single_frame_reprojection_loss(
        points,
        source_pixels,
        world_to_camera,
        intrinsics,
        inverse_intrinsics,
        config,
        step=0,
        total_steps=100,
    )
    assert single_loss.item() < 1e-6
    assert single_stats["valid_mask"].all()

    tracks = TrackBatch(
        target_pixels=source_pixels,
        target_world_to_camera=world_to_camera,
        target_intrinsics=intrinsics,
        valid=torch.ones(2, dtype=torch.bool),
        weights=torch.ones(2),
    )
    inter_loss, inter_stats = inter_frame_track_reprojection_loss(
        points,
        tracks,
        config,
        step=0,
        total_steps=100,
    )
    assert inter_loss.item() < 1e-6
    assert inter_stats["valid_mask"].all()


def test_io_contracts() -> None:
    memory = make_scene_memory(torch.randn(10, 3), torch.randn(10, 8))
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        memory_path = root / "memory.pt"
        save_scene_memory(memory, memory_path)
        loaded = load_scene_memory(memory_path)
        assert torch.equal(memory.points, loaded.points)
        assert torch.equal(memory.features, loaded.features)

        count = 3
        sidecar_path = root / "keyframe_channel.npz"
        np.savez_compressed(
            sidecar_path,
            anchor_image_idx=np.arange(count, dtype=np.int64),
            anchor_feature_yx=np.zeros((count, 2), dtype=np.int32),
            anchor_patch_center_xy=np.zeros((count, 2), dtype=np.float32),
            target_image_idx=np.arange(count, dtype=np.int64),
            target_xy_model=np.ones((count, 2), dtype=np.float32),
            target_patch_center_xy=np.ones((count, 2), dtype=np.float32),
            point3D_id=np.arange(count, dtype=np.int64),
            track_xyz_world=np.ones((count, 3), dtype=np.float32),
            alignment_weight=np.ones(count, dtype=np.float32),
            track_flag=np.ones(count, dtype=bool),
            parallax_deg=np.ones(count, dtype=np.float32),
            colmap_reproj_error=np.zeros(count, dtype=np.float32),
            track_length=np.full(count, 4, dtype=np.int32),
        )
        sidecar = load_stgs_sidecar(sidecar_path)
        assert sidecar.count == count
        assert sidecar.filtered(min_track_length=4).count == count


if __name__ == "__main__":
    test_model_and_schedule()
    test_reprojection_losses()
    test_io_contracts()
    print("ACE-G paper attachment smoke test: PASS")
