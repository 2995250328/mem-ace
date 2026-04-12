#!/usr/bin/env python3
import argparse
import logging
import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import CamLocDataset
from pathlib import Path

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path, help='path to a scene in the dataset folder.')
    parser.add_argument('--image_resolution', type=int, default=480)
    opt = parser.parse_args()

    # 1. Setup Dataset
    dataset = CamLocDataset(
        opt.scene / "test",
        mode=0,
        image_height=opt.image_resolution,
        )
    loader = DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

    print("="*50)
    print(f"Dataset Loaded. Images: {len(dataset)}")

    # 2. Get One Batch
    try:
        data = next(iter(loader))
    except StopIteration:
        print("Error: Dataset is empty.")
        return

    # 这里严格按照您提供的解包顺序（假设第一个是 RGB）
    image_RGB = data[0]

    # --- 修正开始 ---
    # 3. 打印核心统计信息
    # 关键修改：先 .float() 再计算 mean，避免 ByteTensor 报错
    image_RGB_float = image_RGB.float()

    v_min = image_RGB_float.min().item()
    v_max = image_RGB_float.max().item()
    v_mean = image_RGB_float.mean().item()

    print(f"\n[Stats] Image RGB Tensor Statistics:")
    print(f"  Shape: {image_RGB.shape}")
    print(f"  Type : {image_RGB.dtype}")  # 打印一下数据类型
    print(f"  Min  : {v_min:.4f}")
    print(f"  Max  : {v_max:.4f}")
    print(f"  Mean : {v_mean:.4f}")
    # --- 修正结束 ---

    # 4. 判断状态
    if v_min < 0:
        print("\n[Conclusion] ⚠️ 检测到负值！图片 **已经** 被归一化了 (Normalized)。")
        print("  -> 可视化时必须使用 (x * std + mean) 进行还原。")
    elif v_max > 1.0:
        print("\n[Conclusion] ⚠️ 数值大于 1.0！图片是 0-255 的 uint8 原始数据。")
        print("  -> 可视化时：")
        print("     1. 不需要反归一化 (x * std + mean)。")
        print("     2. 但可能需要除以 255.0 转为 0-1 float 才能被 matplotlib 正确显示。")
    else:
        print("\n[Conclusion] ✅ 数值在 [0, 1] 之间。图片是 Float 格式 (仅 ToTensor)。")
        print("  -> 可视化时直接显示即可，不要反归一化。")

    # 5. 直接保存 Raw Tensor
    # save_image 会自动处理 uint8 或 float，只要 float 在 0-1 或 uint8 在 0-255 都能正常保存
    # 如果是 uint8，我们需要除以 255 转为 float 0-1 给 save_image (虽然它有时候也能处理 uint8，但转一下更稳妥)
    if image_RGB.dtype == torch.uint8:
        save_tensor = image_RGB.float() / 255.0
    else:
        save_tensor = image_RGB

    torchvision.utils.save_image(save_tensor, "debug_raw_output.png")
    print(f"\n[Output] Saved raw tensor to 'debug_raw_output.png'")

    # 6. 如果是归一化的（负值），尝试还原
    if v_min < 0:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        # 注意要用 float 版计算
        img_restored = image_RGB_float.cpu() * std + mean
        torchvision.utils.save_image(img_restored, "debug_restored_imagenet.png")
        print(f"[Output] Saved ImageNet-restored tensor to 'debug_restored_imagenet.png'")

    print("="*50)

if __name__ == '__main__':
    main()