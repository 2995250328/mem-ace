#!/usr/bin/env python3
"""Argument parser builder for ACE DINOv2 + LMC training."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils_lmc import _strtobool

DATA_ROOT = Path(os.environ.get("ACE_DATA_ROOT", "/home/xwh/data"))


def get_lmc_train_parser() -> argparse.ArgumentParser:
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
        default=DATA_ROOT / 'map-anything',
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
            '输出权重文件名后缀（如 lmc.pt）。实际路径为 '
            '<experiment_root>[/<experiment_subdir>]/<dataset>/<scene>/<function>/<run_id>/best_K*_it*_<suffix>.pt'
        ),
    )
    parser.add_argument(
        '--experiment_root',
        type=Path,
        default=None,
        help='实验输出根目录。None 时默认为 output/；下按 数据集/场景/function/run_id 组织。',
    )
    parser.add_argument(
        '--experiment_subdir',
        type=str,
        default=None,
        help=(
            '在 experiment_root 下再套一层子目录，用于区分代码版本/分支（如 v1）。'
            '例如设为 v1 时：run_dir = <experiment_root>/v1/<dataset>/<scene>/<function>/<run_id>/。'
            'None 或空字符串表示不使用额外子目录。'
        ),
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
    parser.add_argument(
        '--overwrite_run_dir',
        type=_strtobool,
        default=True,
        help='当 run_name 非 auto 且 run_dir 已存在时，是否先清空目录再写（默认 True，保证目录里只有本次结果）。',
    )

    # ------------------------------------------------------------------
    # DINOv2 编码器相关
    # ------------------------------------------------------------------
    parser.add_argument(
        '--dinov2_path',
        type=Path,
        default=DATA_ROOT / 'checkpoints' / 'dinov2_vitl14_pretrain.pth',
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
        '--num_data_loader_workers',
        type=int,
        default=12,
        help=(
            'DataLoader 子进程数量。过大且系统/会话 open files 上限低时可能触发 '
            'OSError: [Errno 24] Too many open files；可改为 0（主进程加载）或 2–4，或执行 ulimit -n 提高上限。'
        ),
    )
    parser.add_argument(
        '--samples_per_image',
        type=int,
        default=512,
        help='每张图像在填充 buffer 时随机采样的特征点数量。',
    )
    parser.add_argument(
        '--buffer_sampling_replacement',
        type=_strtobool,
        default=None,
        help=(
            '填充 buffer 时采样是否有放回。'
            'True=允许重复采样（legacy）；False=无放回采样（map-anything 对齐）；'
            '不显式设置时由 lmc_profile 决定（legacy=True, mapany_flow_v1=False）。'
        ),
    )
    parser.add_argument(
        '--buffer_sample_valid_coords',
        type=_strtobool,
        default=True,
        help=(
            '填充训练 buffer / sampled S1 时，若 batch 提供真实深度生成的有效 scene-coordinate patch，'
            '优先从这些 patch 采样。默认 True；没有真实 coords 时自动回退到 image mask 随机采样。'
        ),
    )
    parser.add_argument(
        '--buffer_valid_coord_sample_ratio',
        type=float,
        default=1.0,
        help=(
            '每次采样预算中优先分配给有效深度/scene-coordinate patch 的比例。'
            '1.0 表示尽量全部来自有效 patch，不足时再从 image mask 随机补齐。'
        ),
    )
    parser.add_argument(
        '--buffer_valid_coord_neighbor_radius',
        type=int,
        default=1,
        help=(
            '有效深度 patch 的采样邻域半径，单位是 DINO 输出 patch。'
            '0 表示只采样有效 patch 本身；1 表示中心加一圈邻域。'
        ),
    )
    parser.add_argument(
        '--buffer_valid_coord_neighbor_mode',
        choices=['none', 'cross', 'square'],
        default='cross',
        help=(
            '有效深度 patch 的邻域形状。cross=上下左右邻域，'
            'square=包含对角邻域，none=不扩展。'
        ),
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
        help='S2 训练 batch size。当前 head reshape 约定会按 16xW 组织样本，建议设为 16 的倍数以避免尾部样本被截断；OOM 时优先下调这个参数。',
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
    parser.add_argument(
        '--use_lmc',
        type=_strtobool,
        default=False,
        help='是否启用 GeoLMC 两阶段迭代训练。False 时退化为 vanilla。',
    )
    parser.add_argument(
        '--train_preset',
        type=str,
        default='none',
        choices=[
            'none',
            'memory_compare_ace_g_v1',
            'memory_compare_ace_g_v2',
            'ace_g_indoor6_4090_global_fixedzero_v1',
        ],
        help=(
            '训练预设。none=不改动 parser 默认值；'
            'memory_compare_ace_g_v1=当前 memory compare/ACE-G 常用配置，'
            '会自动补齐 use_lmc、ace_g、strict preflight、scene/head 容差、'
            'BSE world-point 路径、S1 buffer 训练，以及 train_compare 输出目录；'
            'memory_compare_ace_g_v2=deterministic-eval/composite 指标版本；'
            'ace_g_indoor6_4090_global_fixedzero_v1=indoor6/4090 true-global fixed-zero ACE-G 配方。'
        ),
    )
    parser.add_argument(
        '--lmc_profile',
        type=str,
        default='legacy',
        choices=['legacy', 'mapany_flow_v1'],
        help=(
            'LMC 训练口径配置。legacy=保持当前行为；'
            'mapany_flow_v1=按 map-anything 流程回滚的对照分支。'
        ),
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
        '--lmc_flow',
        type=str,
        default='iterative',
        choices=['iterative', 'ace_g'],
        help=(
            'LMC 训练流模式。'
            'iterative=当前双阶段 S1+S2 循环（默认，行为完全不变）；'
            'ace_g=ACE-G 混合路径：buffer 存 raw backbone 特征，S2 按 batch 做 fusion forward。'
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
        '--lmc_memory_preflight',
        type=_strtobool,
        default=True,
        help='加载 memory 后是否先做只读一致性预检（默认 True，只告警不改训练逻辑）。',
    )
    parser.add_argument(
        '--lmc_memory_preflight_strict',
        type=_strtobool,
        default=False,
        help='memory preflight 发现问题时是否直接 raise；False=仅告警，True=失败即停止。',
    )
    parser.add_argument(
        '--lmc_memory_preflight_center_tol',
        type=float,
        default=5.0,
        help='preflight 中允许 ||scene_center - mean(pooled_points)|| 的最大偏差（米）。'
             ' scene_center 若为相机位置均值（与 ACE 对齐）会与点云质心差数米，默认 5.0 兼容两种定义。',
    )
    parser.add_argument(
        '--lmc_strict_scene_check',
        type=_strtobool,
        default=True,
        help='memory.scene 与训练 scene 名称不一致时是否直接报错。',
    )
    parser.add_argument(
        '--lmc_strict_center_check',
        type=_strtobool,
        default=True,
        help='memory.scene_center 与 dataset.mean_cam_center 偏差过大时是否直接报错。',
    )
    parser.add_argument(
        '--lmc_scene_center_max_distance',
        type=float,
        default=3.0,
        help='允许 ||memory.scene_center - dataset.mean_cam_center|| 的最大偏差（米）。',
    )
    parser.add_argument(
        '--bse_denorm_to_world',
        type=_strtobool,
        default=False,
        help='BSE memory: 将归一化坐标 (raw-mu)/sigma 还原为世界坐标 raw，'
             '后续 pipeline 与 pooled 版本完全一致（scene_center=mu，head.mean=mu，无 depth scaling）。'
             '用于验证归一化是否为性能下降根因。',
    )
    parser.add_argument(
        '--c1_ref_norm_alpha',
        type=float,
        default=1.0,
        help=(
            '仅对 contract_mode=C1 且使用 points_ref_norm 的路径生效：'
            '将 normalized reference target/runtime memory 整体乘以 alpha。'
            'alpha>1 会放大 target 数值范围，便于观察高精度 metric 是否因归一化压缩过强而受损。'
            '默认 1.0 表示保持现状。'
        ),
    )
    parser.add_argument(
        '--c1_aux_ref_loss_weight',
        type=float,
        default=0.0,
        help=(
            '仅对 contract_mode=C1 生效：在主 normalized-C1 reprojection loss 之外，'
            '额外加入一个 reference-frame SmoothL1 辅助监督。'
            '监督目标使用 dataset 返回的真实 world scene coordinates 经过 world->ref 变换后的 points_ref。'
            '默认 0.0 表示关闭；建议首轮实验从 0.1 开始。'
        ),
    )
    parser.add_argument(
        '--c1_aux_depth_root',
        type=Path,
        default=None,
        help=(
            'C1 aux_ref 使用的外部深度目录或 WAI scene 根目录。'
            '若指向 scene 根目录，会自动追加 --c1_aux_depth_kind；'
            '若为空，ACE backend 会按 ACE_DATA_ROOT 自动查找 '
            'mapanything-dataset/wai_data/indoor6/<scene>_train/<depth_kind>。'
        ),
    )
    parser.add_argument(
        '--c1_aux_depth_kind',
        type=str,
        default='gt_depth',
        choices=['gt_depth', 'colmap_depth'],
        help='C1 aux_ref 自动查找或 scene 根目录下使用的 WAI 深度子目录。',
    )
    parser.add_argument(
        '--c1_aux_ref_sample_ratio',
        type=float,
        default=0.5,
        help=(
            '兼容旧参数；新的采样控制请使用 --buffer_sample_valid_coords 和 '
            '--buffer_valid_coord_sample_ratio。'
        ),
    )
    parser.add_argument(
        '--lmc_head_mean_max_shift',
        type=float,
        default=3.0,
        help='允许 head.mean 对齐到 memory.scene_center 时相对 dataset 均值的最大偏差（米）。',
    )
    parser.add_argument(
        '--lmc_strict_head_mean_alignment',
        type=_strtobool,
        default=True,
        help='head.mean 对齐到 memory.scene_center 时若偏差过大，是否直接报错。',
    )

    # ------------------------------------------------------------------
    # ACE-G flow 特有参数（仅 --lmc_flow ace_g 时生效）
    # ------------------------------------------------------------------
    parser.add_argument(
        '--ace_g_fusion_in_s2',
        type=_strtobool,
        default=False,
        help=(
            'R2 路径：S2 中是否训练 fusion（默认 False = R1 冻结 fusion）。'
            '仅在 --lmc_flow ace_g 时生效。'
        ),
    )
    parser.add_argument(
        '--ace_g_fusion_lr_ratio',
        type=float,
        default=0.01,
        help=(
            'R2 路径：fusion LR 相对 head LR 的比例（默认 0.01 = 1%%）。'
            '仅在 --ace_g_fusion_in_s2 True 时生效。'
        ),
    )
    parser.add_argument(
        '--ace_g_cross_iter_eval',
        type=_strtobool,
        default=False,
        help=(
            '是否在每轮迭代结束后，用下一轮 S1 更新后的 compressor 做一次跨迭代 eval，'
            '检测 S2 head 是否过拟合当前迭代的 compressor 分布。仅 --lmc_flow ace_g 时生效。'
        ),
    )
    parser.add_argument(
        '--pe_normalize_input',
        type=_strtobool,
        default=False,
        help=(
            '是否对 FourierPositionEncoding 输入坐标做尺度归一化（除以 std），'
            '减轻不同场景坐标尺度差异带来的 PE 不稳定。默认 False。'
        ),
    )
    parser.add_argument(
        '--lmc_compressor_pe_scale_mode',
        type=str,
        default=None,
        choices=['raw', 'std', 'scene_scale'],
        help=(
            'Compressor Fourier PE 输入坐标尺度策略。None 表示兼容旧行为：'
            '--pe_normalize_input False -> raw，True -> std；scene_scale 使用与 fusion 相同的 scene_scale。'
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
        '--lmc_fps_start_policy',
        type=str,
        default='farthest_from_center',
        choices=['farthest_from_center', 'lowest_index', 'highest_index', 'legacy_random'],
        help=(
            'GeoLMC FPS 初始点策略。默认 farthest_from_center 是确定性的，'
            '保证训练和推理压缩出同一组 latent coordinates；legacy_random 用于回退旧随机行为。'
        ),
    )
    parser.add_argument(
        '--lmc_auto_mode_by_visibility',
        type=_strtobool,
        default=False,
        help=(
            '当 memory 在多视角下前方可见性不足时，自动将 lmc_mode 从 global '
            '回退到 lmc_visibility_fallback_mode（默认 local）。默认关闭，'
            '保证命令行指定的 lmc_mode 就是实际训练模式。'
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
        '--lmc_key_slice_idx',
        type=int,
        default=None,
        help=(
            'GeoLMC key projection 使用的 pooled_features slice（zero-based）。'
            'None 表示兼容旧行为：多层 memory 默认 slice 2，单层 memory 使用完整 feature。'
        ),
    )
    parser.add_argument(
        '--lmc_key_feature_mode',
        choices=['slice', 'scalar_mix'],
        default='slice',
        help=(
            'GeoLMC key feature 来源。slice 使用单层 feature；'
            'scalar_mix 对多层 pooled_features 做可学习 softmax 加权，value path 保持 concat(all_layers)。'
        ),
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
        '--lmc_log_runtime_stats',
        type=_strtobool,
        default=False,
        help='是否记录 LMC runtime 诊断标量（attention/token usage/norm/geometry stats）。默认 False，不改变训练输出。',
    )
    parser.add_argument(
        '--lmc_runtime_stats_interval',
        type=int,
        default=100,
        help='LMC runtime stats 的 fusion 调用日志间隔。仅在 --lmc_log_runtime_stats True 时生效。',
    )
    parser.add_argument(
        '--lmc_runtime_stats_max_pixels',
        type=int,
        default=4096,
        help='计算 attention/token stats 时最多抽样的 query pixel/token 数。仅保存标量，不保存 attention tensor。',
    )
    parser.add_argument(
        '--lmc_fusion_geometry_mode',
        type=str,
        default='value_only_raw',
        choices=['value_only_raw', 'value_only_norm', 'geokey_norm'],
        help=(
            'Fusion geometry 注入模式。value_only_raw=旧行为，raw centered memory_p 只进 value；'
            'value_only_norm=scene-scale normalized memory_p 只进 value；'
            'geokey_norm=normalized memory_p 同时以零初始化 scalar gate 进入 key。'
        ),
    )
    parser.add_argument(
        '--lmc_fusion_key_geo_init',
        type=float,
        default=0.0,
        help='geokey_norm 中 key geometry scalar gate 的初始值。默认 0.0，使 A2 初始严格等价 A1。',
    )
    parser.add_argument(
        '--lmc_fusion_scene_scale_source',
        type=str,
        default='memory_points_p95',
        choices=['memory_points_p95', 'fixed'],
        help='Fusion geometry normalization 的 scene_scale 来源。memory_points_p95=P95(||pooled_points-scene_center||)；fixed=使用显式 value。',
    )
    parser.add_argument(
        '--lmc_fusion_scene_scale_value',
        type=float,
        default=None,
        help='当 --lmc_fusion_scene_scale_source fixed 时使用的正数 scene_scale。',
    )
    parser.add_argument(
        '--s1_batch_size',
        type=int,
        default=28,
        help=(
            'S1 阶段实际使用的 batch 大小（默认 28）。'
            'full_map 与 sample_per_image 都会用该 batch 过 DINOv2 encoder，显存峰值相同；'
            '24GB 显存建议设为 1 或 2 以免多步后碎片化 OOM（如 --s1_batch_size 1）。'
        ),
    )
    parser.add_argument(
        '--s1_use_buffer',
        type=_strtobool,
        default=False,
        help=(
            'S1 是否复用 raw feature buffer 训练（默认 False）。'
            'True 时跳过每步 encoder 前向，直接从 buffer 采样。'
        ),
    )
    parser.add_argument(
        '--s1_buffer_refill_mode',
        type=str,
        default='full',
        choices=['full', 'partial'],
        help=(
            'S1 buffer 重填策略。full=每迭代清空后全量重建（默认）；'
            'partial=保留上一轮部分 raw buffer，并只重建剩余部分。'
        ),
    )
    parser.add_argument(
        '--s1_buffer_keep_ratio',
        type=float,
        default=0.5,
        help='partial refill 时保留上一轮 S1 raw buffer 的比例；默认 0.5。',
    )
    parser.add_argument(
        '--s1_buffer_refill_ratio',
        type=float,
        default=None,
        help='partial refill 时新重建部分的比例；None 时自动取 1 - s1_buffer_keep_ratio。若显式设置，需与 keep_ratio 之和为 1。',
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
        '--s1_loss_step_mode',
        type=str,
        default=None,
        choices=['fixed_zero', 'per_iter', 'global_monotonic'],
        help=(
            'S1 sampled loss 的 ReproLoss step 选择。'
            'None 表示兼容旧行为：普通 profile 使用 fixed_zero，mapany_flow_v1 使用 global_monotonic。'
            'fixed_zero=始终使用 step 0；per_iter=按当前 LMC iteration 内 S1 update 递增；'
            'global_monotonic=使用全局 monotonic step。'
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
        '--s1_last_iter_use_final_buffer',
        type=_strtobool,
        default=False,
        help=(
            '最后一轮的 S1 是否使用 buffer_size_final 大小的 buffer。'
            'False（默认）：与 V1 一致，S1 始终用 training_buffer_size；'
            'True：最后一轮 S1 扩大为 buffer_size_final，可让 S1 在最终轮见到更多数据。'
        ),
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
        '--buffer_on_cpu_final',
        type=_strtobool,
        default=True,
        help=(
            '最后一轮 S2 的 buffer_size_final buffer 强制放在 CPU，避免 3x final buffer 在 24GB 卡上 OOM。'
            '默认 True；若确实要 final buffer 也留在 GPU，可显式设为 False。'
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
        dest='s1_early_stop',
        type=_strtobool,
        default=argparse.SUPPRESS,
        help='[Deprecated] 等价于 --s1_early_stop（仅为兼容旧脚本保留）。',
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
        default=argparse.SUPPRESS,
        help=(
            '首轮 S2 的 step 回拨比例（相对该轮 S2 总步数）。'
            '未显式设置时由 lmc_profile 决定（legacy=0.20, mapany_flow_v1=0.00）。'
        ),
    )
    parser.add_argument(
        '--s2_repro_rewind_later_ratio',
        type=float,
        default=argparse.SUPPRESS,
        help=(
            '后续轮次 S2 的 step 回拨比例。'
            '未显式设置时由 lmc_profile 决定（legacy=0.08, mapany_flow_v1=0.00）。'
        ),
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
        default=argparse.SUPPRESS,
        help=(
            '首轮 S2 head 学习率额外 boost 倍数。'
            '未显式设置时由 lmc_profile 决定（legacy=1.2, mapany_flow_v1=1.0）。'
        ),
    )
    parser.add_argument(
        '--s2_lr_boost_later',
        type=float,
        default=argparse.SUPPRESS,
        help=(
            '后续轮次 S2 head 学习率额外 boost 倍数。'
            '未显式设置时由 lmc_profile 决定（legacy=1.0, mapany_flow_v1=1.0）。'
        ),
    )
    parser.add_argument(
        '--s2_lr_warmup_steps',
        type=int,
        default=0,
        help='每轮 S2 头部优化可选 warmup 步数（0 表示不额外 warmup）。',
    )
    parser.add_argument(
        '--s2_step_eff_mode',
        type=str,
        default='auto',
        choices=['legacy_rewind', 'per_iter', 'auto'],
        help=(
            'S2 阶段 step_eff（控制 soft_clamp 调度）的计算方式。\n'
            '  legacy_rewind: 全局单调步 + 指数衰减回拨（旧行为，late iter 时 soft_clamp 趋近最小值）。\n'
            '  per_iter: 每轮 S2 从 step_eff=0 开始，按比例映射到完整调度区间，\n'
            '             确保每轮都经历完整 soft_clamp 范围（max→min）。\n'
            '             有效防止 ACE-G R2 模式中融合模块更新后 soft_clamp 过小导致的发散。\n'
            '  auto (默认): ace_g_fusion_in_s2=True 时自动使用 per_iter，否则使用 legacy_rewind。'
        ),
    )
    parser.add_argument(
        '--s2_grad_clip_max_norm',
        type=float,
        default=1.0,
        help=(
            'S2 阶段反向传播后的梯度裁剪范数上限（<=0 表示禁用）。'
            ' 与 S1 阶段的 clip_grad_norm(1.0) 保持一致，防止 loss_invalid 异常时梯度爆炸导致 NaN。'
        ),
    )
    parser.add_argument(
        '--loss_invalid_max_delta',
        type=float,
        default=1000.0,
        help=(
            'loss_invalid（相机空间 L1 距离）每个分量的最大值（单位：与 depth_target 同量纲，通常为米）。'
            ' 当预测完全发散（相机坐标偏差达数百万米）时防止 loss 爆炸。'
            ' 设为 0 表示禁用裁剪（不推荐，等同于旧行为）。'
            ' 默认 1000.0 对室内外场景均适用。'
        ),
    )
    parser.add_argument(
        '--s2_polish_epochs',
        type=int,
        default=0,
        help=(
            '每轮常规 S2 训练后追加的低学习率精修 epoch 数。'
            ' 默认 0 表示关闭；用于诊断 late-iter 低 LR refinement 是否可前移。'
        ),
    )
    parser.add_argument(
        '--s2_polish_head_lr',
        type=float,
        default=1e-4,
        help='S2 polish 阶段 head 的固定学习率。',
    )
    parser.add_argument(
        '--s2_polish_fusion_lr_ratio',
        type=float,
        default=0.005,
        help=(
            'ACE-G R2 polish 阶段 fusion 学习率相对 head 学习率的比例。'
            ' 仅在 ace_g_fusion_in_s2=True 时生效。'
        ),
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
    parser.add_argument(
        '--eval_deterministic',
        type=_strtobool,
        default=False,
        help='是否启用评估去随机性：固定 PyTorch/DataLoader/DSAC*（若扩展支持 set_seed）。False 时回到旧随机评估行为。',
    )
    parser.add_argument(
        '--eval_dsacstar_seed',
        type=int,
        default=1305,
        help='评估时 DSAC* 的基础随机种子。若 dsacstar 扩展暴露了 set_seed，将用于固定 RANSAC 采样。',
    )
    parser.add_argument(
        '--eval_dsacstar_seed_per_frame',
        type=_strtobool,
        default=True,
        help='是否对每一帧使用 base_seed + frame_idx 的 DSAC* 种子，降低线程调度带来的抖动。',
    )
    parser.add_argument(
        '--eval_num_workers',
        type=int,
        default=6,
        help='评估 DataLoader 的 worker 数；会配合固定 worker seed 使用。',
    )
    parser.add_argument(
        '--post_train_eval_seeds',
        type=int,
        nargs='+',
        default=[1305],
        help='训练结束后最终评估使用的 seed 列表；多个 seed 时写入聚合后的中位数结果。',
    )
    parser.add_argument(
        '--post_train_hypotheses',
        type=int,
        default=64,
        help='训练结束后最终评估使用的 RANSAC hypotheses 数量。',
    )

    return parser
