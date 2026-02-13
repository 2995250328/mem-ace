# DINOv2 训练 CUDA OOM 原因分析

## 1. Buffer 会占用这么多显存吗？

**不会。** Training buffer **不占用 GPU 显存**。

- 在 `create_training_buffer()` 里，buffer 用 `torch.zeros(buffer_size, feature_dim)` 在 **CPU** 上创建（未指定 device）。
- 写入时用 `.cpu()` 放回 CPU：`sampled_features[:num_to_add].cpu()`。
- 因此 `training_buffer = (buffer_features, buffer_coords)` 全部在 **CPU 内存**，约 4M × 1024 × 4 bytes ≈ 16 GB 内存，与本次 50 GiB 的 GPU OOM 无关。

---

## 2. 是否因为 DINO 没冻结导致梯度？

**不是主要原因。** 本次 OOM 发生在 **前向传播**，而不是反向传播。

- 报错栈显示崩溃在：`attention.py` 的 `attn = q @ k.transpose(-2, -1)`，即计算 attention 矩阵时。
- 这是单次前向中的一次大显存分配（约 50.21 GiB），与是否冻结 backbone、是否算梯度无关。
- 若 `freeze_backbone=False`，反向时还会多占显存（存中间激活），但当前 OOM 的**直接触发点**是前向里的 attention 矩阵太大。

---

## 3. 真正原因：单步里“图像 batch × 分辨率”过大

训练时每个 step 是从 DataLoader 取 **一整批图像**，做 `regressor(image)` 前向（含整颗 DINOv2 ViT-L），**没有**用 buffer 里的特征。

- 默认 `batch_size=512`（**512 张图**/step）。
- 默认 `image_resolution=518`（高 518，宽按比例且为 14 的倍数，例如 ≈ 518×686 → 37×49 个 patch）。
- ViT-L 的 attention 张量形状为 `(B, num_heads, N, N)`，其中 `N = 图像 patch 数`。
  - 例如：B=512, num_heads=16, N≈37×49≈1813 → 512×16×1813² ≈ 26.9e9 个元素，float32 约 **107 GB**。
- 你看到的 “Tried to allocate 50.21 GiB” 正是其中某次 attention 矩阵的分配；GPU 只有 23.52 GiB，因此 OOM。

结论：**显存爆掉是因为“每步 512 张 518 分辨率图一起过 ViT-L”**，attention 的 (B, heads, N, N) 太大。

---

## 4. 建议修改

1. **显著减小 batch_size（最有效）**  
   例如改为 **4 或 8**（或 2），再根据显存逐步调大：
   ```bash
   --batch_size 4
   ```

2. **可选：降低输入分辨率**  
   例如 224 或 280（保持 14 的倍数）：
   ```bash
   --image_resolution 224
   ```

3. **保持 backbone 冻结**  
   继续使用 `--freeze_backbone True`（默认），可减少反向时的显存和计算。

4. **可选：使用 xFormers 的 memory_efficient_attention**  
   若安装并启用 xFormers，DINOv2 里可用 `MemEffAttention` 替代标准 attention，降低 attention 的显存占用（需在 `depth_anything_v2` 中启用）。

推荐先只把 `--batch_size` 降到 4 或 8 再跑，一般即可避免此次 OOM。
