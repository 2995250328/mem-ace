#!/usr/bin/env python3
import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# 尝试导入 transformers，用于语义分割
try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ===============================================================================
# Part 1: SuperPoint 网络定义 (保持不变)
# ===============================================================================

class SuperPointNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self):
        super(SuperPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))
        return semi, desc


class SuperPointFrontend(object):
    def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh, cuda=False):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.cell = 8
        self.border_remove = 4

        self.net = SuperPointNet()
        if cuda:
            self.net.load_state_dict(torch.load(weights_path))
            self.net = self.net.cuda()
        else:
            self.net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        self.net.eval()

    def nms_fast(self, in_corners, H, W, dist_thresh):
        grid = np.zeros((H, W)).astype(int)
        inds = np.zeros((H, W)).astype(int)
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        count = 0
        for i, rc in enumerate(rcorners.T):
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, img):
        assert img.ndim == 2, 'Image must be grayscale.'
        assert img.dtype == np.float32, 'Image must be float32.'
        H, W = img.shape[0], img.shape[1]
        inp = img.copy()
        inp = (inp.reshape(1, H, W))
        inp = torch.from_numpy(inp)
        inp = torch.autograd.Variable(inp).view(1, 1, H, W)
        if self.cuda:
            inp = inp.cuda()
        outs = self.net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        semi = semi.data.cpu().numpy().squeeze()
        dense = np.exp(semi)
        dense = dense / (np.sum(dense, axis=0) + .00001)
        nodust = dense[:-1, :, :]
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])

        xs, ys = np.where(heatmap >= self.conf_thresh)
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))
        pts[0, :] = ys  # Row 0 -> X
        pts[1, :] = xs  # Row 1 -> Y
        pts[2, :] = heatmap[xs, ys]

        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            if self.cuda:
                samp_pts = samp_pts.cuda()
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


# ===============================================================================
# Part 2: 天空分割模块 (SegFormer)
# ===============================================================================

class SkySegmenter:
    """
    使用 HuggingFace SegFormer 模型对天空进行分割。
    模型: nvidia/segformer-b0-finetuned-ade-512-512
    ADE20K 数据集中，天空的 Class ID 通常是 2。
    """

    def __init__(self, device='cuda'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请先安装 transformers: `pip install transformers`")

        self.device = device
        print("==> 加载 SegFormer (ADE20K) 模型用于天空抑制...")
        # 使用最轻量的 b0 版本，速度极快
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # ADE20K 类别索引: 2 是 Sky
        self.SKY_CLASS_ID = 2

    def get_sky_mask(self, img_bgr):
        """
        输入: BGR 图片 (OpenCV 读取)
        输出: 布尔型 Mask (H, W)，True 表示是天空
        """
        # 转 RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 预处理
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (1, 150, H/4, W/4)

        # 上采样回原图尺寸
        # 注意: interpolate 需要 (N, C, H, W)
        upsampled_logits = F.interpolate(
            logits,
            size=img_rgb.shape[:2],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        # 获取类别索引 (N, H, W) -> (H, W)
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # 生成掩码: 等于天空ID的地方为 True
        sky_mask = (pred_seg == self.SKY_CLASS_ID)

        return sky_mask.cpu().numpy()


# ===============================================================================
# Part 3: 主处理逻辑
# ===============================================================================

def main(args):
    # 1. 初始化
    weights_path = args.weights_path
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"❌ 错误: 找不到 SuperPoint 权重 {weights_path}")
        return

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"==> 使用设备: {'CUDA' if use_cuda else 'CPU'}")

    # 2. 初始化模型
    sp_fe = SuperPointFrontend(weights_path=weights_path,
                               nms_dist=args.nms_dist,
                               conf_thresh=args.conf_thresh,
                               nn_thresh=0.7,
                               cuda=use_cuda)

    sky_seg = None
    if args.mask_sky:
        sky_seg = SkySegmenter(device='cuda' if use_cuda else 'cpu')

    # 3. 获取图片
    exts = ['*.jpg', '*.png', '*.jpeg']
    image_files = []
    for e in exts:
        image_files.extend(list(input_dir.glob(e)) + list(input_dir.glob(e.upper())))
    image_files.sort()

    print(f"==> 开始处理 {len(image_files)} 张图片...")

    for i, img_path in enumerate(image_files):
        filename = img_path.name

        # 读取图片
        img_orig = cv2.imread(str(img_path))
        if img_orig is None: continue

        H, W = img_orig.shape[:2]

        # A. 提取 SuperPoint
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        img_input = img_gray.astype('float32') / 255.0
        # pts: [3, N] -> [X, Y, Conf]
        pts, desc, heatmap = sp_fe.run(img_input)

        # B. 获取天空 Mask (如果启用)
        sky_mask = None
        if sky_seg:
            sky_mask = sky_seg.get_sky_mask(img_orig)

            # 可选：保存 Mask 看看效果
            # mask_vis = (sky_mask * 255).astype(np.uint8)
            # cv2.imwrite(str(output_dir / f"mask_{filename}"), mask_vis)

        # C. 过滤与绘制
        vis_img = img_orig.copy()
        kept_count = 0
        removed_count = 0

        for j in range(pts.shape[1]):
            x = int(round(pts[0, j]))  # Col
            y = int(round(pts[1, j]))  # Row

            # 边界检查
            x = min(max(x, 0), W - 1)
            y = min(max(y, 0), H - 1)

            # 天空检查
            is_sky = False
            if sky_mask is not None:
                if sky_mask[y, x]:  # 注意 mask 是 [H, W]，索引是 [y, x]
                    is_sky = True

            if is_sky:
                removed_count += 1
                # 可选：把被剔除的点画成红色，方便调试
                # cv2.circle(vis_img, (x, y), 2, (0, 0, 255), -1)
                continue

            # 保留的点画绿色
            cv2.circle(vis_img, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            kept_count += 1

        out_path = output_dir / f"vis_{filename}"
        cv2.imwrite(str(out_path), vis_img)
        print(f"[{i + 1}/{len(image_files)}] {filename}: 保留 {kept_count}, 剔除 {removed_count} (天空)")

    print(f"==> 全部完成！结果保存在 {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='output_vis_masked')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth')
    parser.add_argument('--nms_dist', type=int, default=4)
    # 既然有 Mask 了，阈值可以设低一点，比如 0.015
    parser.add_argument('--conf_thresh', type=float, default=0.015)
    parser.add_argument('--no_cuda', action='store_true')
    # 新增开关
    parser.add_argument('--mask_sky', action='store_true', help='启用天空分割掩码进行过滤')

    args = parser.parse_args()
    main(args)