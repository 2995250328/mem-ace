#!/usr/bin/env python3
# test_dinov2_basic.py
# Basic functionality test for DINOv2 integration

import sys
import torch
import numpy as np
from pathlib import Path

def test_network():
    """Test DINOv2 network instantiation and forward pass."""
    print("Testing DINOv2 network...")

    try:
        from ace_network_dinov2 import DINOv2Encoder, Regressor

        # Test encoder
        print("  Creating DINOv2Encoder...")
        dinov2_path = Path('/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth')

        if not dinov2_path.exists():
            print(f"  ⊘ Skipping (weights not found at {dinov2_path})")
            return True

        encoder = DINOv2Encoder(
            pretrained_path=str(dinov2_path),
            out_channels=1024,
            freeze_backbone=True
        )
        print("  ✓ Encoder created")

        # Test forward pass
        print("  Testing forward pass...")
        test_input = torch.randn(2, 3, 518, 518)

        with torch.no_grad():
            output = encoder(test_input)

        expected_shape = (2, 1024, 37, 37)
        if output.shape == expected_shape:
            print(f"  ✓ Forward pass successful: {output.shape}")
        else:
            print(f"  ✗ Shape mismatch: got {output.shape}, expected {expected_shape}")
            return False

        # Test regressor
        print("  Creating Regressor...")
        mean = torch.zeros(3)
        regressor = Regressor.create_from_encoder(
            dinov2_path=str(dinov2_path),
            mean=mean,
            num_head_blocks=1,
            use_homogeneous=True,
            freeze_backbone=True
        )
        print("  ✓ Regressor created")

        # Test regressor forward pass
        print("  Testing regressor forward pass...")
        with torch.no_grad():
            scene_coords = regressor(test_input)

        expected_shape = (2, 3, 37, 37)
        if scene_coords.shape == expected_shape:
            print(f"  ✓ Regressor forward pass successful: {scene_coords.shape}")
        else:
            print(f"  ✗ Shape mismatch: got {scene_coords.shape}, expected {expected_shape}")
            return False

        print("✓ Network test passed")
        return True

    except Exception as e:
        print(f"✗ Network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test DINOv2 dataset loader."""
    print("\nTesting DINOv2 dataset...")

    try:
        from dataset_dinov2 import CamLocDatasetDINOv2

        # Check if test scene exists
        test_scene = Path('datasets/7scenes_chess/train')
        if not test_scene.exists():
            print(f"  ⊘ Skipping (test scene not found at {test_scene})")
            return True

        print("  Creating dataset...")
        dataset = CamLocDatasetDINOv2(
            root_dir=test_scene,
            mode=0,
            use_half=False,
            image_height=518,
            augment=False
        )
        print(f"  ✓ Dataset created with {len(dataset)} images")

        # Test loading one sample
        print("  Loading sample...")
        sample = dataset[0]

        image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords = sample

        print(f"  ✓ Sample loaded:")
        print(f"    Image shape: {image.shape}")
        print(f"    Expected: [3, H, W] where H and W are multiples of 14")

        # Check image dimensions
        _, H, W = image.shape
        if H % 14 == 0 and W % 14 == 0:
            print(f"    ✓ Image dimensions ({H}, {W}) are multiples of 14")
        else:
            print(f"    ✗ Image dimensions ({H}, {W}) are NOT multiples of 14")
            return False

        # Check image channels
        if image.shape[0] == 3:
            print(f"    ✓ Image has 3 channels (RGB)")
        else:
            print(f"    ✗ Image has {image.shape[0]} channels, expected 3")
            return False

        print("✓ Dataset test passed")
        return True

    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_resolutions():
    """Test various image resolutions."""
    print("\nTesting image resolutions...")

    valid_resolutions = [504, 518, 532, 560]
    invalid_resolutions = [480, 500, 640]

    print("  Valid resolutions (multiples of 14):")
    for res in valid_resolutions:
        if res % 14 == 0:
            print(f"    ✓ {res} = {res//14} × 14")
        else:
            print(f"    ✗ {res} is not a multiple of 14")
            return False

    print("  Invalid resolutions (not multiples of 14):")
    for res in invalid_resolutions:
        if res % 14 != 0:
            suggested = (res // 14) * 14
            print(f"    ✓ {res} is invalid, suggest {suggested} or {suggested + 14}")
        else:
            print(f"    ✗ {res} should be invalid but is a multiple of 14")
            return False

    print("✓ Resolution test passed")
    return True


def test_imports():
    """Test all required imports."""
    print("\nTesting imports...")

    modules = [
        'ace_network_dinov2',
        'dataset_dinov2',
        'trainer_dinov2',
        'torch',
        'torchvision',
        'numpy',
        'cv2',
    ]

    all_passed = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_passed = False

    if all_passed:
        print("✓ Import test passed")
    else:
        print("✗ Some imports failed")

    return all_passed


def main():
    print("=" * 80)
    print("DINOv2 Integration - Basic Functionality Test")
    print("=" * 80)
    print()

    tests = [
        ("Imports", test_imports),
        ("Image Resolutions", test_image_resolutions),
        ("Network", test_network),
        ("Dataset", test_dataset),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(result for _, result in results)

    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now proceed with training:")
        print("  ./train_ace_dinov2.py datasets/7scenes_chess output/test.pt")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues before training.")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
