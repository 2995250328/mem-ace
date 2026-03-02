#!/usr/bin/env python3
# train_ace_dinov2_lmc.py
# -----------------------------------------------------------------------------
# ACE DINOv2 + GeoLMC 训练入口。
#
# 运行模式：
#   1) Vanilla（不启用 LMC）
#      --use_lmc False
#
#   2) LMC（启用两阶段迭代训练）
#      --use_lmc True --memory_path <pooled_memory.pt>
#
# 关键约束：
#   - memory 文件仅支持 POOLED 格式（需包含 pooled_points / pooled_features）。
#   - 输入分辨率需要是 14 的倍数（DINOv2 patch size = 14）。
# -----------------------------------------------------------------------------

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def setup_cuda_environment():
    """在导入 torch 前设置 CUDA_VISIBLE_DEVICES，避免进程占错 GPU。"""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:0')
    pre_args, _ = pre_parser.parse_known_args()
    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: CUDA_VISIBLE_DEVICES = {gpu_id}")


setup_cuda_environment()

import torch
from trainer_dinov2_lmc import TrainerACEDINOv2LMC

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    """把 CLI 布尔参数统一解析为 bool。"""
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {x!r}')


def _sanitize_tag(text: str) -> str:
    """Make a filesystem-safe tag for run folder names."""
    text = re.sub(r'[^a-zA-Z0-9._-]+', '_', str(text))
    return text.strip('._-') or "run"


def _sanitize_stem(p: Path) -> str:
    """文件名安全：只保留字母数字下划线横线。"""
    s = p.stem if hasattr(p, 'stem') else str(p)
    return "".join(c for c in s if c.isalnum() or c in "._-") or "model"


def _format_buf_million(x: int) -> str:
    """Format buffer size into compact M unit for folder naming."""
    return f"{x / 1_000_000:.1f}".rstrip('0').rstrip('.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Train ACE DINOv2 with optional GeoLMC iterative training.\n'
            'Vanilla mode: --use_lmc False\n'
            'LMC mode: --use_lmc True --memory_path <pooled_memory.pt>'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------------------------
    # 必选参数
    # ------------------------------------------------------------------
    parser.add_argument(
        'scene',
        type=Path,
        help=(
            '训练场景目录。\n'
            'data_backend=ace 时读取 <scene>/train。\n'
            'data_backend=wai 时读取 <scene>/scene_meta.json（scene 需指向单个 WAI 场景目录，如 scene1_train）。'
        ),
    )
    parser.add_argument(
        '--data_backend',
        type=str,
        default='ace',
        choices=['ace', 'wai'],
        help='数据读取后端：ace=原始 rgb/poses/calibration 目录；wai=WAI scene_meta.json。',
    )
    parser.add_argument(
        '--wai_repo_root',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'map-anything',
        help='map-anything 仓库根目录（用于导入 WAI 读取工具）。',
    )
    parser.add_argument(
        '--wai_image_modality',
        type=str,
        default='image',
        help='WAI 图像模态键（默认 image）。',
    )
    parser.add_argument(
        'output_map',
        type=Path,
        help=(
            '输出权重文件名后缀（如 lmc.pt）。实际路径为 <experiment_root>/<dataset>/<scene>/<function>/<run_id>/best_K*_it*_<suffix>.pt'
        ),
    )
    parser.add_argument(
        '--experiment_root',
        type=Path,
        default=None,
        help='实验输出根目录。None 时默认为 output/；下按 数据集/场景/function/run_id 组织。',
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='auto',
        help='实验目录名。auto 时按时间+场景+关键参数自动生成。',
    )
    parser.add_argument(
        '--output_layout',
        type=str,
        default='hierarchical',
        choices=['legacy', 'hierarchical'],
        help=(
            '输出目录结构。hierarchical（默认）: run_dir = <experiment_root>/<dataset>/<scene>/<function>/<run_id>，'
            'indoor6/7Scenes 等有专属子目录。legacy: run_dir = <experiment_root>/<run_name>。'
        ),
    )

    # ------------------------------------------------------------------
    # DINOv2 编码器相关
    # ------------------------------------------------------------------
    parser.add_argument(
        '--dinov2_path',
        type=Path,
        default=Path('/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth'),
        help='DINOv2 ViT-L/14 预训练权重路径。',
    )
    parser.add_argument(
        '--freeze_backbone',
        type=_strtobool,
        default=True,
        help=(
            '是否冻结 DINOv2 backbone。\n'
            'True: 显存/速度更稳，通常用于 LMC 训练；False: 会训练 backbone，代价更高。'
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help=(
            '训练设备，通常是 cuda:x。\n'
            '会用于设置 CUDA_VISIBLE_DEVICES（例如 cuda:0 -> GPU 0）。'
        ),
    )

    # ------------------------------------------------------------------
    # 回归头结构
    # ------------------------------------------------------------------
    parser.add_argument(
        '--num_head_blocks',
        type=int,
        default=4,
        help='回归头残差块数量。越大表达能力更强，但训练成本更高。',
    )
    parser.add_argument(
        '--use_homogeneous',
        type=_strtobool,
        default=True,
        help='是否使用 homogeneous 坐标头（通常保持 True）。',
    )

    # ------------------------------------------------------------------
    # 通用训练规模与学习率
    # ------------------------------------------------------------------
    parser.add_argument(
        '--training_buffer_size',
        type=int,
        default=2560000,
        help=(
            'S2 阶段训练 buffer 的样本总量。\n'
            '越大样本覆盖越好，但填充更慢、显存占用也更高。'
        ),
    )
    parser.add_argument(
        '--buffer_batch_size',
        type=int,
        default=10,
        help=(
            '填充 buffer 时每次 backbone 前向的图像数。\n'
            '增大可提升填充速度，但显存压力增大。'
        ),
    )
    parser.add_argument(
        '--buffer_image_width',
        type=int,
        default=None,
        help=(
            '填充 buffer 时的固定图像宽度（可选覆盖自动值）。\n'
            '用于控制显存和特征图大小，None 表示自动按高宽比估算。'
        ),
    )
    parser.add_argument(
        '--samples_per_image',
        type=int,
        default=512,
        help='每张图像在填充 buffer 时随机采样的特征点数量。',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=24,
        help='每轮 S2 head 训练的 epoch 数（每个 LMC iteration 都会执行）。',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5120,
        help='S2 训练 batch size。OOM 时优先下调这个参数。',
    )
    parser.add_argument(
        '--learning_rate_min',
        type=float,
        default=0.0001,
        help='基础 OneCycle 的最小学习率（非 S1/S2 特化调度时使用）。',
    )
    parser.add_argument(
        '--learning_rate_max',
        type=float,
        default=0.0001,
        help='兼容旧参数：当未显式提供 s1/s2 学习率上限时，作为回退值。',
    )
    parser.add_argument(
        '--s1_learning_rate_max',
        type=float,
        default=0.0001,
        help='S1 最大学习率（压缩器/融合模块阶段）。',
    )
    parser.add_argument(
        '--s2_learning_rate_max',
        type=float,
        default=0.001,
        help='S2 最大学习率（head-only 阶段，默认对齐 dino ace）。',
    )
    parser.add_argument(
        '--image_resolution',
        type=int,
        default=518,
        help='输入图像高度（会自动修正到 14 的倍数）。',
    )

    # ------------------------------------------------------------------
    # 数据增强
    # ------------------------------------------------------------------
    parser.add_argument(
        '--use_aug',
        type=_strtobool,
        default=True,
        help='是否启用训练数据增强（旋转/缩放）。',
    )
    parser.add_argument(
        '--aug_rotation',
        type=int,
        default=15,
        help='数据增强的最大旋转角度（度）。',
    )
    parser.add_argument(
        '--aug_scale',
        type=float,
        default=1.5,
        help='数据增强的最大缩放因子（最小为 1/aug_scale）。',
    )

    # ------------------------------------------------------------------
    # Reprojection Loss（ace_depth 公式）
    # ------------------------------------------------------------------
    parser.add_argument(
        '--repro_loss_type',
        type=str,
        default='dyntanh',
        choices=['l1', 'l2', 'smooth_l1', 'tanh', 'dyntanh'],
        help='重投影损失类型（推荐 dyntanh）。',
    )
    parser.add_argument(
        '--repro_loss_soft_clamp',
        type=float,
        default=50,
        help='重投影 soft clamp 上限（动态/静态 tanh 的上界）。',
    )
    parser.add_argument(
        '--repro_loss_soft_clamp_min',
        type=float,
        default=1,
        help='dyntanh 调度到后期时的最小 soft clamp。',
    )
    parser.add_argument(
        '--repro_loss_schedule',
        type=str,
        default='circle',
        choices=['circle', 'linear'],
        help='dyntanh 的时间调度方式。',
    )
    parser.add_argument(
        '--repro_loss_hard_clamp',
        type=float,
        default=1000,
        help='超大重投影误差硬阈值（超过判为 invalid）。',
    )
    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='有效深度下限（小于该值判为 invalid）。')
    parser.add_argument('--depth_max', type=float, default=1000,
                        help='有效深度上限（大于该值判为 invalid）。')
    parser.add_argument('--depth_target', type=float, default=10,
                        help='invalid 分支的目标深度缩放系数。')
    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='是否启用 FP16 混合精度训练。')

    # ------------------------------------------------------------------
    # LMC 核心参数
    # ------------------------------------------------------------------
    # 仅在 --use_lmc True 且提供 --memory_path 时生效。
    parser.add_argument(
        '--use_lmc',
        type=_strtobool,
        default=False,
        help='是否启用 GeoLMC 两阶段迭代训练。False 时退化为 vanilla。',
    )
    parser.add_argument(
        '--apply_baseline_contract',
        type=_strtobool,
        default=True,
        help=(
            '仅在 --use_lmc False 时生效：'
            'True=沿用 vanilla DINO-ACE 约定默认（会强制 num_head_blocks/LR 等）；'
            'False=严格使用命令行给定超参数，便于与 LMC 做一一公平对照。'
        ),
    )
    parser.add_argument(
        '--vanilla_iterations',
        type=int,
        default=1,
        help=(
            '仅在 --use_lmc False 时生效：执行多少轮 vanilla 迭代训练。'
            '每轮都会重新构建 buffer、重置 onecycle 调度并可按轮评估，'
            '用于与 LMC 的 iteration 机制做公平对照。'
        ),
    )
    parser.add_argument(
        '--memory_path',
        type=str,
        default=None,
        help=(
            'POOLED memory 文件路径。\n'
            '必须包含 pooled_points / pooled_features；可选 scene_center / all_scale_tokens / layers_idx。'
        ),
    )
    parser.add_argument(
        '--lmc_mode',
        type=str,
        default='global',
        choices=['global', 'local', 'hierarchical', 'learned'],
        help='GeoLMC 模式：global/local/hierarchical/learned。',
    )
    parser.add_argument(
        '--lmc_auto_mode_by_visibility',
        type=_strtobool,
        default=True,
        help=(
            '当 memory 在多视角下前方可见性不足时，自动将 lmc_mode 从 global '
            '回退到 lmc_visibility_fallback_mode（默认 local）。'
        ),
    )
    parser.add_argument(
        '--lmc_visibility_front_ratio_threshold',
        type=float,
        default=0.85,
        help='触发自动回退的前方可见性阈值（mean front ratio）。',
    )
    parser.add_argument(
        '--lmc_visibility_fallback_mode',
        type=str,
        default='local',
        choices=['local', 'hierarchical'],
        help='可见性不足时的自动回退模式。',
    )
    parser.add_argument(
        '--lmc_visibility_sample_points',
        type=int,
        default=4096,
        help='估计 memory 可见性时随机采样的点数上限。',
    )
    parser.add_argument(
        '--num_latent_tokens',
        type=int,
        default=64,
        help='压缩后 latent token 数量 K（FPS 采样目标数量）。',
    )
    parser.add_argument(
        '--num_attn_layers',
        type=int,
        default=4,
        help='GeoLMC 注意力层数。层数越大表达更强，耗时更高。',
    )
    parser.add_argument(
        '--s1_batch_size',
        type=int,
        default=28,
        help=(
            'S1 阶段实际使用的 batch 大小（默认 8，OOM 安全）。'
            'full_map 与 sample_per_image 都会用该 batch 过 DINOv2 encoder，显存峰值相同；'
            '24GB 显存建议设为 1 或 2 以免多步后碎片化 OOM（如 --s1_batch_size 1）。'
        ),
    )
    parser.add_argument(
        '--use_scale_token',
        type=_strtobool,
        default=True,
        help='是否把 memory 的 all_scale_tokens 注入 compressor。',
    )
    parser.add_argument(
        '--lmc_iterations',
        type=int,
        default=28,
        help='LMC 总迭代轮数。每轮均执行 S1 + S2。',
    )
    parser.add_argument(
        '--lmc_train_steps',
        type=int,
        default=600,
        help='第 2 轮及之后每轮 S1 的训练步数。',
    )
    parser.add_argument(
        '--lmc_warmup_steps',
        type=int,
        default=2000,
        help='第 1 轮 S1 的训练步数（通常比后续更大）。',
    )
    parser.add_argument(
        '--s1_early_stop',
        type=_strtobool,
        default=True,
        help='是否启用 S1 早停：当 pxErr 长时间无明显改善时提前结束本轮 S1。',
    )
    parser.add_argument(
        '--s1_early_stop_min_updates',
        type=int,
        default=400,
        help='S1 早停最少更新步数。达到该步数后才开始判断是否进入平台期。',
    )
    parser.add_argument(
        '--s1_early_stop_patience',
        type=int,
        default=180,
        help='S1 早停耐心值（更新步）。超过该窗口无显著改善则提前停止。',
    )
    parser.add_argument(
        '--s1_early_stop_rel_improve',
        type=float,
        default=0.01,
        help='S1 早停最小相对改善阈值，例如 0.01 表示需优于历史最好值 1% 才算改善。',
    )
    parser.add_argument(
        '--s1_early_stop_ema_beta',
        type=float,
        default=0.90,
        help='S1 早停用的 pxErr EMA 平滑系数，越大越平滑。',
    )
    parser.add_argument(
        '--s1_loss_mode',
        type=str,
        default='full_map',
        choices=['full_map', 'sample_per_image', 'sample_pooled'],
        help=(
            'S1 重投影损失计算方式: '
            'full_map=整图 E2E 不采样; '
            'sample_per_image=每张图固定采样数(类 S2); '
            'sample_pooled=当前方式(从整批有效点中随机采样).'
        ),
    )
    parser.add_argument(
        '--s1_full_map_max_points',
        type=int,
        default=0,
        help=(
            'full_map 模式下参与 repro loss 的最大有效点数，0=不限制。'
            '设成正整数可随机子采样，减轻困难点主导梯度（对齐 map-anything 的 buffer 采样思想）。'
        ),
    )
    parser.add_argument(
        '--s1_empty_cache_interval',
        type=int,
        default=30,
        help=(
            'S1 每 N 步调用 torch.cuda.empty_cache() 以缓解显存碎片化（full_map/sample_per_image 多步后 OOM 时有效）。'
            '0=关闭。默认 30；仍 OOM 可改为 20 或配合 --s1_batch_size 1。'
        ),
    )
    parser.add_argument(
        '--buffer_size_final',
        type=int,
        default=None,
        help='最后一轮 S2 使用的 buffer 大小；None 表示 3 * training_buffer_size。',
    )
    parser.add_argument(
        '--buffer_on_cpu',
        type=_strtobool,
        default=True,
        help=(
            'S2 阶段将训练 buffer 放在 CPU，每步只把当前 batch 搬到 GPU，避免 23GB 卡 OOM（与 map-anything 一致）。'
            '默认 True；设为 False 则 buffer 全程在 GPU（显存充足时可用）。'
        ),
    )
    parser.add_argument(
        '--head_reset_strategy',
        type=str,
        default='first_only',
        choices=['first_only', 'every', 'output_only', 'none'],
        help='S2 前 head 重置策略。注意：当前实现首轮 S2 还会强制清零一次。',
    )
    parser.add_argument(
        '--head_lr_multiplier_s2',
        type=float,
        default=1.5,
        help='S2 head 学习率乘子（基于 s2_learning_rate_max 进一步缩放）。',
    )

    # ------------------------------------------------------------------
    # S1 学习率策略（map-anything 风格）
    # ------------------------------------------------------------------
    parser.add_argument(
        '--lmc_lr_scheduler_type',
        type=str,
        default='onecycle_improved',
        choices=['auto', 'warmup_cosine', 'warmup_plateau_cosine', 'onecycle_improved', 'onecycle_legacy'],
        help=(
            'S1 LR 调度策略。auto: 第1轮 warmup_plateau_cosine，后续 warmup_cosine。'
        ),
    )
    parser.add_argument('--lmc_min_lr_ratio', type=float, default=0.01,
                        help='S1 余弦调度最小学习率占最大学习率的比例。')
    parser.add_argument('--s1_lr_scale_later', type=float, default=0.2,
                        help='第 2 轮及以后 S1 的 base LR 相对 s1_learning_rate_max 的缩放（0~1）。默认 0.2 避免 iter 2+ 时 LR 过高导致 loss 上升；可试 0.3/0.4 若 S1 收敛过慢。')
    parser.add_argument('--lmc_warmup_ratio', type=float, default=0.1,
                        help='S1 warmup 比例（0~1）。')
    parser.add_argument('--lmc_plateau_ratio', type=float, default=0.35,
                        help='S1 warmup_plateau_cosine 的 plateau 比例。')
    parser.add_argument('--lmc_lr_pct_start', type=float, default=0.3,
                        help='S1 OneCycle 的 pct_start。')
    parser.add_argument('--lmc_lr_div_factor', type=float, default=25.0,
                        help='S1 onecycle_improved 的 div_factor。')
    parser.add_argument('--lmc_lr_final_div_factor', type=float, default=10000.0,
                        help='S1 onecycle_improved 的 final_div_factor。')
    parser.add_argument(
        '--s1_early_stop_enable',
        type=_strtobool,
        default=True,
        help='是否启用 S1 平台期早停（基于 pxErr 改善幅度）。',
    )
    # ------------------------------------------------------------------
    # GeoLMC 几何结构参数（传给 GeoLMC 模块，默认与 map-anything 一致）
    # ------------------------------------------------------------------
    parser.add_argument(
        '--geo_sigma',
        type=float,
        default=0.5,
        help=(
            'GeoLMC 地理空间高斯核带宽（σ）。'
            '控制 memory token 间的几何感知权重衰减速度。'
            '默认 0.5，与 map-anything 一致。场景尺度差异大时可调整。'
        ),
    )
    parser.add_argument(
        '--num_fine',
        type=int,
        default=128,
        help=(
            'GeoLMC 细粒度（fine）FPS 候选点数。'
            '决定 local/hierarchical 模式下的精细邻域大小。默认 128。'
        ),
    )
    parser.add_argument(
        '--num_coarse',
        type=int,
        default=16,
        help=(
            'GeoLMC 粗粒度（coarse）FPS 候选点数。'
            '决定 hierarchical 模式下粗层聚合邻域大小。默认 16。'
        ),
    )

    # ------------------------------------------------------------------
    # S2 ReproLoss step 精细化 + 学习率 boost
    # ------------------------------------------------------------------
    parser.add_argument(
        '--s2_repro_rewind_first_ratio',
        type=float,
        default=0.20,
        help='首轮 S2 的 step 回拨比例（相对该轮 S2 总步数）。',
    )
    parser.add_argument(
        '--s2_repro_rewind_later_ratio',
        type=float,
        default=0.08,
        help='后续轮次 S2 的 step 回拨比例。',
    )
    parser.add_argument(
        '--s2_repro_rewind_tau',
        type=float,
        default=300.0,
        help='step 回拨指数衰减时间常数（越大回拨持续越久）。',
    )
    parser.add_argument(
        '--s2_lr_boost_first',
        type=float,
        default=1.2,
        help=(
            '首轮 S2 head 学习率额外 boost 倍数。'
            '已从 1.5 下调至 1.2：避免 Indoor6 等场景 S2 早期 LR 过高导致 loss 爆炸。'
            '若训练收敛慢，可尝试 1.3~1.5。'
        ),
    )
    parser.add_argument(
        '--s2_lr_boost_later',
        type=float,
        default=1.0,
        help=(
            '后续轮次 S2 head 学习率额外 boost 倍数。'
            '已从 1.2 下调至 1.0（不额外 boost）：稳态迭代时以 s2_learning_rate_max 为准即可。'
        ),
    )
    parser.add_argument(
        '--s2_lr_warmup_steps',
        type=int,
        default=0,
        help='每轮 S2 头部优化可选 warmup 步数（0 表示不额外 warmup）。',
    )

    # ------------------------------------------------------------------
    # 每轮评估与最优权重策略
    # ------------------------------------------------------------------
    parser.add_argument(
        '--eval_each_iteration',
        type=_strtobool,
        default=True,
        help='是否在每轮 iteration 结束后执行一次评估。',
    )
    parser.add_argument(
        '--keep_best_only',
        type=_strtobool,
        default=True,
        help='是否仅保留最优权重到 output_map（其余迭代权重删除）。',
    )
    parser.add_argument(
        '--best_metric',
        type=str,
        default='pct5',
        choices=['pct5', 'pct10_5', 'composite', 'rt_error'],
        help='最优模型判定指标：pct5 / pct10_5 / composite / rt_error（旋转+平移误差最小）。',
    )

    # ------------------------------------------------------------------
    # 训练结束后的额外评估
    # ------------------------------------------------------------------
    parser.add_argument(
        '--eval_after_train',
        type=_strtobool,
        default=True,
        help='训练全部完成后是否再跑一次最终评估。',
    )
    parser.add_argument(
        '--eval_session',
        type=str,
        default='post_train',
        help='训练后评估的 session 名称（影响输出文件名）。',
    )
    parser.add_argument(
        '--post_train_eval_device',
        type=str,
        default='cuda:0',
        help='训练后最终评估使用的设备。默认 cuda:0（与训练同进程，trainer 释放后 GPU 可用）；OOM 时可设为 cpu。',
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 参数校验与默认修正
    # ------------------------------------------------------------------
    if args.image_resolution % 14 != 0:
        args.image_resolution = (args.image_resolution // 14) * 14
        _logger.warning(
            "Image resolution adjusted to %s (must be multiple of 14)",
            args.image_resolution,
        )

    if not args.dinov2_path.exists():
        _logger.error("DINOv2 weights not found: %s", args.dinov2_path)
        sys.exit(1)

    if args.data_backend == 'wai':
        scene_meta_path = args.scene / "scene_meta.json"
        if not scene_meta_path.exists():
            _logger.error("WAI backend requires scene_meta.json at: %s", scene_meta_path)
            sys.exit(1)
        if not args.wai_repo_root.exists():
            _logger.error("WAI repo root not found: %s", args.wai_repo_root)
            sys.exit(1)
    else:
        train_dir = args.scene / "train"
        if not train_dir.exists():
            _logger.error("ACE backend expects training dir: %s", train_dir)
            sys.exit(1)

    if args.use_lmc and args.memory_path is None:
        _logger.error("--use_lmc requires --memory_path (path to POOLED .pt file)")
        sys.exit(1)
    if args.vanilla_iterations < 1:
        _logger.error("--vanilla_iterations must be >= 1")
        sys.exit(1)

    # Baseline contract: when use_lmc=False, apply vanilla DINO ACE defaults so that
    # behavior matches train_ace_dinov2.py. Structural defaults (head blocks, LR) are
    # always applied; buffer/epochs/batch are only applied when still at parser default,
    # so explicit CLI (e.g. --training_buffer_size 8760000) is respected.
    if (not args.use_lmc) and args.apply_baseline_contract:
        _vanilla_defaults = {
            "num_head_blocks": 1,
            "learning_rate_max": 0.001,
            "s1_learning_rate_max": 0.001,
            "s2_learning_rate_max": 0.001,
            "training_buffer_size": 5760000,
            "samples_per_image": 384,
            "epochs": 16,
            "batch_size": 3840,
        }
        _parser_defaults = {
            "num_head_blocks": 4,
            "learning_rate_max": 0.0001,
            "s1_learning_rate_max": 0.0001,
            "s2_learning_rate_max": 0.001,
            "training_buffer_size": 2560000,
            "samples_per_image": 512,
            "epochs": 24,
            "batch_size": 5120,
        }
        applied = []
        for key, vanilla_val in _vanilla_defaults.items():
            cur = getattr(args, key)
            parser_default = _parser_defaults[key]
            if key in ("num_head_blocks", "learning_rate_max", "s1_learning_rate_max", "s2_learning_rate_max"):
                setattr(args, key, vanilla_val)
                applied.append(f"{key}={vanilla_val}")
            elif cur == parser_default:
                setattr(args, key, vanilla_val)
                applied.append(f"{key}={vanilla_val}")
            # else: user explicitly set a non-default value, keep it
        _logger.info(
            "[Baseline contract] use_lmc=False: applied vanilla defaults where not overridden: %s.",
            ", ".join(applied),
        )
    elif not args.use_lmc:
        _logger.info(
            "[Baseline contract] disabled (apply_baseline_contract=False). "
            "Will use CLI hyperparameters as-is for iterative no-LMC baseline."
        )

    if args.buffer_size_final is None:
        args.buffer_size_final = args.training_buffer_size * 3

    # 输出结构：<experiment_root>/<dataset>/<scene>/<function>/<run_id>/best_K*_it*_<suffix>.pt
    # dataset/scene 从 scene 路径解析，与 train_ace_dinov2 一致（如 .../indoor6_ace/scene1/train -> indoor6_ace, scene1）
    run_root = Path(args.experiment_root).resolve() if args.experiment_root is not None else Path("output").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    scene_path = Path(args.scene).resolve()
    if len(scene_path.parts) >= 2:
        if scene_path.name in ("train", "test", "val") and len(scene_path.parts) >= 3:
            dataset = _sanitize_tag(scene_path.parent.parent.name)
            scene = _sanitize_tag(scene_path.parent.name)
        else:
            dataset = _sanitize_tag(scene_path.parent.name)
            scene = _sanitize_tag(scene_path.name)
    else:
        dataset = "default"
        scene = _sanitize_tag(scene_path.name or "scene")

    scene_tag = scene
    mode_tag = _sanitize_tag(args.lmc_mode if args.use_lmc else "vanilla")
    sched_tag = _sanitize_tag(args.lmc_lr_scheduler_type if args.use_lmc else "onecycle")
    buf_tag = _format_buf_million(args.training_buffer_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_run_name = (
        f"{timestamp}_{scene_tag}_{mode_tag}"
        f"_buf{buf_tag}M_K{args.num_latent_tokens}"
        f"_it{args.lmc_iterations}_bs{args.batch_size}_{sched_tag}"
    )
    run_name = auto_run_name if args.run_name == 'auto' else _sanitize_tag(args.run_name)

    # 默认使用 hierarchical：run_root / dataset / scene / function / run_id
    output_layout = getattr(args, 'output_layout', 'hierarchical')
    if output_layout == 'hierarchical':
        function = "dino_ace_baseline" if not args.use_lmc else "dino_ace_lmc_s1s2"
        run_id = run_name
        run_dir = run_root / dataset / scene / function / run_id
    else:
        run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 权重文件按关键参数命名，output_map 仅作后缀
    out_suffix = _sanitize_stem(args.output_map)
    best_pt_name = f"best_K{args.num_latent_tokens}_it{args.lmc_iterations}_{out_suffix}.pt"
    args.output_map = run_dir / best_pt_name
    args.run_dir = run_dir

    # Save full console-equivalent training log to file (map-anything style).
    full_log_path = run_dir / "training_full_log.txt"
    root_logger = logging.getLogger()
    existing_paths = {
        getattr(h, 'baseFilename', None) for h in root_logger.handlers
        if hasattr(h, 'baseFilename')
    }
    if str(full_log_path) not in existing_paths:
        file_handler = logging.FileHandler(full_log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        root_logger.addHandler(file_handler)

    # Persist run metadata for reproducibility.
    with open(run_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)
    with open(run_dir / "run_command.txt", 'w', encoding='utf-8') as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    # ------------------------------------------------------------------
    # 打印配置摘要
    # ------------------------------------------------------------------
    _logger.info("=" * 80)
    _logger.info("ACE DINOv2 Training%s", " + LMC" if args.use_lmc else "")
    _logger.info("=" * 80)
    _logger.info("Scene        : %s", args.scene)
    _logger.info("Data backend : %s", args.data_backend)
    if args.data_backend == 'wai':
        _logger.info("WAI repo     : %s", args.wai_repo_root)
        _logger.info("WAI image key: %s", args.wai_image_modality)
    _logger.info("Run Dir      : %s", args.run_dir)
    if output_layout == 'hierarchical':
        _logger.info("Output layout: hierarchical (dataset/scene/function/run_id)")
    _logger.info("Output       : %s", args.output_map)
    _logger.info("Full Log     : %s", full_log_path)
    _logger.info("Device       : %s", args.device)
    _logger.info("LMC          : %s", f"ON (mode={args.lmc_mode})" if args.use_lmc else "OFF")
    if not args.use_lmc and args.vanilla_iterations > 1:
        _logger.info("Vanilla iters: %d (iterative baseline mode)", args.vanilla_iterations)

    if args.use_lmc:
        _logger.info("Memory       : %s", args.memory_path)
        _logger.info("Tokens/Attn  : K=%d, attn_layers=%d", args.num_latent_tokens, args.num_attn_layers)
        _logger.info("Iterations   : %d (S1 warmup=%d, S1 later=%d)",
                     args.lmc_iterations, args.lmc_warmup_steps, args.lmc_train_steps)
        _logger.info(
            "Mode auto-sw  : enabled=%s, vis_thr=%.2f, fallback=%s, sample_pts=%d",
            args.lmc_auto_mode_by_visibility,
            args.lmc_visibility_front_ratio_threshold,
            args.lmc_visibility_fallback_mode,
            args.lmc_visibility_sample_points,
        )
        _logger.info("Head reset   : %s", args.head_reset_strategy)
        _logger.info(
            "LR max       : S1=%.2e, S2=%.2e (legacy learning_rate_max=%.2e)",
            args.s1_learning_rate_max, args.s2_learning_rate_max, args.learning_rate_max
        )
        _logger.info("S1 LR        : %s", args.lmc_lr_scheduler_type)
        _logger.info(
            "S2 LR boost  : first=%.2f, later=%.2f | warmup_steps=%d",
            args.s2_lr_boost_first, args.s2_lr_boost_later, args.s2_lr_warmup_steps,
        )
        _logger.info(
            "S2 step rewind: first=%.2f, later=%.2f, tau=%.1f",
            args.s2_repro_rewind_first_ratio,
            args.s2_repro_rewind_later_ratio,
            args.s2_repro_rewind_tau,
        )
        _logger.info("Buffer on CPU: %s (S2 显存不足时保持 True)", args.buffer_on_cpu)

    _logger.info(
        "Eval policy  : each_iter=%s, keep_best_only=%s, best_metric=%s",
        args.eval_each_iteration,
        args.keep_best_only,
        args.best_metric,
    )
    _logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 开始训练
    # ------------------------------------------------------------------
    trainer = TrainerACEDINOv2LMC(args)
    trainer.train()
    _logger.info("Training completed.")

    # ------------------------------------------------------------------
    # 可选：训练结束后的最终评估
    # ------------------------------------------------------------------
    if args.eval_after_train:
        _logger.info("Running post-training evaluation...")
        try:
            # 训练已在 trainer.train() 末尾释放 buffer；这里再释放 trainer 本身占用的显存，便于同进程 GPU eval
            del trainer
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            _logger.info("Freed trainer; running eval on %s.", args.post_train_eval_device)

            from test_ace_dinov2_lmc import run_evaluation_lmc
            # 使用 --post_train_eval_device，默认 cuda:0 以加速评测
            eval_device = getattr(args, "post_train_eval_device", "cuda:0")
            eval_opt = argparse.Namespace(
                scene=args.scene,
                network=args.output_map,
                dinov2_path=args.dinov2_path,
                device=eval_device,
                image_resolution=args.image_resolution,
                session=args.eval_session,
                render_visualization=False,
            )
            result = run_evaluation_lmc(eval_opt)
            _logger.info("========== Post-train Eval (current errors) ==========")
            _logger.info(
                "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | "
                "2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                result['median_rErr'], result['median_tErr'],
                result['pct25_5'], result['pct10_5'], result['pct5'],
                result['pct2'], result['pct1'],
            )
            _logger.info("  Avg time: %.2f ms | Frames: %d",
                         result['avg_time'] * 1000, result['total_frames'])
            _logger.info("=====================================================")
            # 写入 run_dir 便于与 training_full_log 一起查看
            eval_log_path = args.run_dir / "post_train_eval.txt"
            with open(eval_log_path, "w", encoding="utf-8") as f:
                f.write(f"median_rotation_deg\t{result['median_rErr']:.4f}\n")
                f.write(f"median_translation_cm\t{result['median_tErr']:.4f}\n")
                f.write(f"accuracy_25cm5deg_pct\t{result['pct25_5']:.2f}\n")
                f.write(f"accuracy_10cm5deg_pct\t{result['pct10_5']:.2f}\n")
                f.write(f"accuracy_5cm5deg_pct\t{result['pct5']:.2f}\n")
                f.write(f"accuracy_2cm2deg_pct\t{result['pct2']:.2f}\n")
                f.write(f"accuracy_1cm1deg_pct\t{result['pct1']:.2f}\n")
                f.write(f"avg_time_per_frame_ms\t{result['avg_time'] * 1000:.2f}\n")
                f.write(f"total_frames\t{result['total_frames']}\n")
            _logger.info("Eval summary also written to: %s", eval_log_path)
        except Exception as e:
            _logger.warning("Post-training eval failed: %s", e, exc_info=True)
