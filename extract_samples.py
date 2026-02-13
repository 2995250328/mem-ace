import os
import shutil
import numpy as np
from pathlib import Path

# --- 配置区域 ---
# 源文件夹路径
SOURCE_DIR = Path("/data/xwh/Cambridge/GreatCourt/seq1")
# 目标文件夹路径 (注意：这里使用的是你要求的 ace_depth，如果是笔误请自行修改为 ace_depth)
DEST_DIR = Path("/home/xwh/project/ace_depth/test_images")
# 需要提取的图片数量
NUM_SAMPLES = 10
# 支持的图片扩展名
VALID_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def main():
    # 1. 检查源目录
    if not SOURCE_DIR.exists():
        print(f"❌ 错误: 源目录不存在 -> {SOURCE_DIR}")
        return

    # 2. 创建目标目录 (如果不存在会自动创建)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ 目标目录准备就绪 -> {DEST_DIR}")

    # 3. 获取所有图片并排序
    # 排序很重要，确保采样的均匀性是基于时间序列或文件名的
    all_files = sorted([
        f for f in SOURCE_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTS
    ])

    total_files = len(all_files)
    print(f"📂 在源目录中发现 {total_files} 张图片。")

    if total_files == 0:
        print("⚠️ 未找到图片，程序退出。")
        return

    # 4. 计算均匀采样的索引
    if total_files <= NUM_SAMPLES:
        # 如果图片总数少于要提取的数量，则提取所有
        selected_indices = range(total_files)
        print(f"⚠️ 图片总数少于 {NUM_SAMPLES}，将复制所有图片。")
    else:
        # 使用 linspace 生成均匀分布的索引 (例如: 0, 100, 200, ... 900)
        # astype(int) 确保索引是整数
        selected_indices = np.linspace(0, total_files - 1, NUM_SAMPLES, dtype=int)
        print(f"📊 正在均匀提取 {NUM_SAMPLES} 张图片...")

    # 5. 执行复制
    count = 0
    for idx in selected_indices:
        src_file = all_files[idx]
        dst_file = DEST_DIR / src_file.name

        try:
            shutil.copy2(src_file, dst_file)
            # print(f"  Copied: {src_file.name}") # 取消注释可查看详细列表
            count += 1
        except Exception as e:
            print(f"❌ 复制 {src_file.name} 失败: {e}")

    print(f"🎉 完成！已成功将 {count} 张图片保存到: {DEST_DIR}")


if __name__ == "__main__":
    main()