#!/usr/bin/env python3
# validate_dinov2_setup.py
# Validation script to check DINOv2 integration setup

import sys
from pathlib import Path
import torch

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ Module '{module_name}' can be imported")
        return True
    except ImportError as e:
        print(f"✗ Module '{module_name}' import failed: {e}")
        return False

def check_dinov2_weights(weights_path):
    """Check if DINOv2 weights can be loaded."""
    try:
        state_dict = torch.load(weights_path, map_location='cpu')
        print(f"✓ DINOv2 weights loaded successfully")
        print(f"  Keys in state_dict: {len(state_dict.keys()) if isinstance(state_dict, dict) else 'N/A'}")
        return True
    except Exception as e:
        print(f"✗ Failed to load DINOv2 weights: {e}")
        return False

def check_image_resolution(resolution):
    """Check if image resolution is valid for DINOv2."""
    if resolution % 14 == 0:
        print(f"✓ Image resolution {resolution} is valid (multiple of 14)")
        return True
    else:
        suggested = (resolution // 14) * 14
        print(f"✗ Image resolution {resolution} is NOT valid (not multiple of 14)")
        print(f"  Suggested: {suggested} or {suggested + 14}")
        return False

def main():
    print("=" * 80)
    print("DINOv2 Integration Validation")
    print("=" * 80)
    print()

    all_checks_passed = True

    # Check Python files
    print("Checking Python files...")
    files_to_check = [
        ('ace_network_dinov2.py', 'DINOv2 network'),
        ('dataset_dinov2.py', 'DINOv2 dataset'),
        ('trainer_dinov2.py', 'DINOv2 trainer'),
        ('train_ace_dinov2.py', 'Training script'),
        ('test_ace_dinov2.py', 'Testing script'),
    ]

    for filename, description in files_to_check:
        if not check_file_exists(filename, description):
            all_checks_passed = False
    print()

    # Check documentation
    print("Checking documentation...")
    doc_files = [
        ('DINOV2_USAGE.md', 'DINOv2 usage guide'),
        ('CLAUDE.md', 'Claude guide'),
    ]

    for filename, description in doc_files:
        if not check_file_exists(filename, description):
            all_checks_passed = False
    print()

    # Check imports
    print("Checking module imports...")
    modules_to_check = [
        'torch',
        'torchvision',
        'numpy',
        'cv2',
    ]

    for module in modules_to_check:
        if not check_import(module):
            all_checks_passed = False
    print()

    # Check DINOv2 weights
    print("Checking DINOv2 weights...")
    dinov2_path = Path('/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth')
    if dinov2_path.exists():
        if not check_dinov2_weights(dinov2_path):
            all_checks_passed = False
    else:
        print(f"✗ DINOv2 weights not found at {dinov2_path}")
        print(f"  Please download or specify correct path with --dinov2_path")
        all_checks_passed = False
    print()

    # Check image resolutions
    print("Checking common image resolutions...")
    resolutions = [480, 518, 532, 504, 560, 640]
    for res in resolutions:
        check_image_resolution(res)
    print()

    # Check CUDA availability
    print("Checking CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"✗ CUDA is NOT available")
        print(f"  DINOv2 training will be very slow on CPU")
        all_checks_passed = False
    print()

    # Try to instantiate DINOv2 network (if weights available)
    print("Testing DINOv2 network instantiation...")
    if dinov2_path.exists():
        try:
            from ace_network_dinov2 import DINOv2Encoder
            encoder = DINOv2Encoder(
                pretrained_path=str(dinov2_path),
                out_channels=1024,
                freeze_backbone=True
            )
            print(f"✓ DINOv2Encoder instantiated successfully")

            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 518, 518)
            with torch.no_grad():
                output = encoder(dummy_input)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected output shape: [1, 1024, 37, 37]")

            if output.shape == (1, 1024, 37, 37):
                print(f"✓ Output shape is correct!")
            else:
                print(f"✗ Output shape mismatch!")
                all_checks_passed = False

        except Exception as e:
            print(f"✗ Failed to instantiate or test DINOv2 network: {e}")
            all_checks_passed = False
    else:
        print(f"⊘ Skipping network test (weights not available)")
    print()

    # Summary
    print("=" * 80)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You can now train ACE with DINOv2:")
        print("  ./train_ace_dinov2.py datasets/7scenes_chess output/chess_dinov2.pt")
        print()
        print("For detailed usage, see DINOV2_USAGE.md")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        print("Please fix the issues above before training.")
        print("See DINOV2_USAGE.md for setup instructions.")
    print("=" * 80)

    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())
