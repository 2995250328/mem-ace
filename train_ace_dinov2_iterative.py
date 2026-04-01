#!/usr/bin/env python3
# train_ace_dinov2_iterative.py
# Iterative training script for ACE with DINOv2 encoder.

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:3', help='Target device')
    pre_args, _ = pre_parser.parse_known_args()

    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f'Info: Set CUDA_VISIBLE_DEVICES = {gpu_id}')
        print("Info: Inside PyTorch, this will be mapped to 'cuda:0'")


setup_cuda_environment()

import torch

from trainer_dinov2 import TrainerACEDINOv2


def _strtobool(x):
    """Convert string to boolean (replacement for distutils.util.strtobool)."""
    if isinstance(x, bool):
        return x
    x_lower = str(x).lower()
    if x_lower in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    if x_lower in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {x}')


def _sanitize_stem(p: Path) -> str:
    """文件名安全：只保留字母数字下划线横线。"""
    s = p.stem if hasattr(p, 'stem') else str(p)
    return ''.join(c for c in s if c.isalnum() or c in '._-') or 'model'


def _sanitize_tag(text: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(text)).strip('._-') or 'run'


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train ACE with DINOv2 encoder (iterative baseline)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('scene', type=Path,
                        help='Path to scene folder (e.g., /mnt/storage/xwh/indoor6_ace/scene1/train)')
    parser.add_argument('output_map', type=Path,
                        help='Output filename suffix (e.g. best.pt). 实际路径为 <experiment_root>/<dataset>/<scene>/<scene>_dinov2_ep*_bs*_<suffix>.pt')

    parser.add_argument('--experiment_root', type=Path, default=Path('output'),
                        help='实验输出根目录；输出为 <experiment_root>/<dataset>/<scene>/ 下按关键参数命名的 .pt')

    parser.add_argument('--dinov2_path', type=Path,
                        default=Path('/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth'),
                        help='Path to DINOv2 pretrained weights')
    parser.add_argument('--freeze_backbone', type=_strtobool, default=True,
                        help='Freeze DINOv2 backbone during training')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Training device')

    parser.add_argument('--num_head_blocks', type=int, default=4,
                        help='Depth of regression head')
    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='Use homogeneous coordinates')

    parser.add_argument('--training_buffer_size', type=int, default=10000000,
                        help='Training buffer size')
    parser.add_argument('--buffer_batch_size', type=int, default=1,
                        help='Number of images per forward when filling buffer (1=original ACE one-by-one)')
    parser.add_argument('--buffer_image_width', type=int, default=None,
                        help='Fixed image width for buffer when buffer_batch_size>1 (default: 4:3 from height)')
    parser.add_argument('--samples_per_image', type=int, default=512,
                        help='Features sampled per image for buffer')
    parser.add_argument('--epochs', type=int, default=16,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=5120,
                        help='Batch size (images per step). Keep small for DINOv2 ViT-L to avoid OOM (e.g. 4-16 on 24GB GPU)')
    parser.add_argument('--learning_rate_min', type=float, default=0.0001,
                        help='Minimum learning rate')
    parser.add_argument('--learning_rate_max', type=float, default=0.001,
                        help='Maximum learning rate')

    parser.add_argument('--image_resolution', type=int, default=518,
                        help='Image height (must be multiple of 14)')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use data augmentation')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='Max rotation angle for augmentation')
    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='Max scale factor for augmentation')

    parser.add_argument('--repro_loss_type', type=str, default='dyntanh',
                        choices=['l1', 'l2', 'smooth_l1', 'tanh', 'dyntanh'],
                        help='Reprojection loss type')
    parser.add_argument('--repro_loss_soft_clamp', type=float, default=50,
                        help='Soft clamping threshold')
    parser.add_argument('--repro_loss_soft_clamp_min', type=float, default=1,
                        help='Minimum soft clamping threshold')
    parser.add_argument('--repro_loss_schedule', type=str, default='circle',
                        choices=['circle', 'linear'],
                        help='Loss schedule type')
    parser.add_argument('--repro_loss_hard_clamp', type=float, default=1000,
                        help='Hard clamp for reprojection error (px)')
    parser.add_argument('--depth_min', type=float, default=0.1, help='Min depth for valid prediction')
    parser.add_argument('--depth_max', type=float, default=1000, help='Max depth for valid prediction')
    parser.add_argument('--depth_target', type=float, default=10, help='Target depth for invalid proxy loss')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='Use half precision training')

    parser.add_argument('--eval_after_train', type=_strtobool, default=True,
                        help='Run evaluation on test set after training and save results to eval_log.txt')
    parser.add_argument('--eval_session', type=str, default='post_train',
                        help='Session name for evaluation output files (test_*, poses_*, used when eval_after_train=True)')
    parser.add_argument('--post_train_eval_device', type=str, default='cuda:0',
                        help='Device for post-train eval subprocess. Default cuda:0 (fast); use cpu if eval OOM.')

    parser.add_argument('--iterations', type=int, default=1,
                        help='迭代轮数。1=等价于 train_ace_dinov2.py 单次训练')
    parser.add_argument('--iter_buffer_size', type=int, default=None,
                        help='每轮迭代的 buffer 大小，默认等于 training_buffer_size。设为较小值（如 2560000）以实现多次小 buffer 迭代')
    parser.add_argument('--reset_optimizer_each_iter', type=_strtobool, default=False,
                        help='每轮是否完全重置 optimizer 状态。False=只重置 scheduler（保留动量，连续学习）')
    parser.add_argument('--eval_each_iteration', type=_strtobool, default=True,
                        help='每轮结束后保存 checkpoint 并评估')
    parser.add_argument('--keep_best_only', type=_strtobool, default=True,
                        help='若按轮评估，则仅保留最优 checkpoint')
    parser.add_argument('--best_metric', type=str, default='pct5',
                        choices=['pct25_5', 'pct10_5', 'pct5', 'pct2', 'pct1', 'median'],
                        help='选择最优 checkpoint 的指标')
    parser.add_argument('--run_name', type=str, default='auto',
                        help='运行名；auto 时根据时间和设置自动生成')
    parser.add_argument('--output_layout', type=str, default='hierarchical',
                        choices=['hierarchical', 'flat'],
                        help='输出目录布局')
    parser.add_argument('--overwrite_run_dir', type=_strtobool, default=True,
                        help='当 run_name 非 auto 且目录已存在时，是否先清空目录')

    return parser


def _build_run_dir(args, dataset, scene_name):
    run_root = Path(args.experiment_root).resolve() if args.experiment_root is not None else Path('output').resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    buf_size = args.iter_buffer_size or args.training_buffer_size
    buf_tag = f'{buf_size / 1_000_000:.1f}'.rstrip('0').rstrip('.')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    auto_run_name = (
        f'{timestamp}_{_sanitize_tag(scene_name)}_iterative'
        f'_buf{buf_tag}M_it{args.iterations}_bs{args.batch_size}_ep{args.epochs}'
    )
    run_name = auto_run_name if args.run_name == 'auto' else _sanitize_tag(args.run_name)

    if args.output_layout == 'hierarchical':
        run_dir = run_root / _sanitize_tag(dataset) / _sanitize_tag(scene_name) / 'dino_ace_baseline' / run_name
    else:
        run_dir = run_root / run_name

    if run_dir.exists() and args.output_layout == 'hierarchical' and args.run_name != 'auto' and args.overwrite_run_dir:
        for p in run_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except OSError:
                pass
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _attach_full_log_file_handler(run_dir: Path):
    full_log_path = run_dir / 'training_full_log.txt'
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
    return full_log_path


def _persist_run_metadata(args):
    run_dir = args.run_dir
    with open(run_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)
    with open(run_dir / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write('python ' + ' '.join(sys.argv) + '\n')


def prepare_args(args):
    if args.iterations < 1:
        raise ValueError('--iterations must be >= 1')
    if args.iter_buffer_size is not None and args.iter_buffer_size < 1:
        raise ValueError('--iter_buffer_size must be >= 1 when provided')

    scene_path = Path(args.scene).resolve()
    if len(scene_path.parts) >= 2:
        if scene_path.name in ('train', 'test', 'val') and len(scene_path.parts) >= 3:
            dataset = scene_path.parent.parent.name
            scene_name = scene_path.parent.name
        else:
            dataset = scene_path.parent.name
            scene_name = scene_path.name
    else:
        dataset = 'default'
        scene_name = scene_path.name or 'scene'

    run_dir = _build_run_dir(args, dataset, scene_name)
    suffix = _sanitize_stem(args.output_map)
    pt_name = f'best_it{args.iterations}_{suffix}.pt'
    args.output_map = run_dir / pt_name
    args.run_dir = run_dir
    args._dataset_display_name = dataset
    args._scene_display_name = scene_name

    if args.image_resolution % 14 != 0:
        adjusted = (args.image_resolution // 14) * 14
        _logger.warning('Image resolution %s is not multiple of 14, adjusting to %s', args.image_resolution, adjusted)
        args.image_resolution = adjusted

    if not args.dinov2_path.exists():
        raise FileNotFoundError(f'DINOv2 weights not found at {args.dinov2_path}')

    # trainer 期望 scene 为场景根目录（其下才有 train/），若传入 .../train 则改为 parent
    if scene_path.name == 'train':
        args.scene = scene_path.parent

    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    _attach_full_log_file_handler(run_dir)
    _persist_run_metadata(args)
    return args


def get_iter_buffer_size(args):
    return args.iter_buffer_size or args.training_buffer_size


def build_trainer_options(args):
    iter_buffer_size = get_iter_buffer_size(args)
    trainer_args = copy.copy(args)
    trainer_args.training_buffer_size = iter_buffer_size * args.iterations
    return trainer_args


def _parse_eval_summary(summary_file: Path):
    if not summary_file.exists():
        return None
    result = {}
    for raw in summary_file.read_text(encoding='utf-8').splitlines():
        if not raw or raw.startswith('#') or '\t' not in raw:
            continue
        key, value = raw.split('\t', 1)
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def _parse_test_log(test_log_file: Path):
    if not test_log_file.exists():
        return None
    raw = test_log_file.read_text(encoding='utf-8').strip().split()
    if len(raw) < 3:
        return None
    median_r = float(raw[0])
    median_t = float(raw[1])
    avg_time = float(raw[2])
    return {
        'median_rotation_deg': median_r,
        'median_translation_cm': median_t,
        'avg_time_per_frame_ms': avg_time * 1000.0,
    }


def _score_eval(eval_result, best_metric: str):
    if eval_result is None:
        return -float('inf')
    if best_metric == 'pct25_5':
        return float(eval_result['accuracy_25cm5deg_pct'])
    if best_metric == 'pct10_5':
        return float(eval_result['accuracy_10cm5deg_pct'])
    if best_metric == 'pct2':
        return float(eval_result['accuracy_2cm2deg_pct'])
    if best_metric == 'pct1':
        return float(eval_result['accuracy_1cm1deg_pct'])
    if best_metric == 'median':
        return -float(eval_result['median_rotation_deg']) - float(eval_result['median_translation_cm'])
    return float(eval_result['accuracy_5cm5deg_pct'])


def run_post_eval(args, output_map, session):
    from test_ace_dinov2 import run_evaluation

    eval_device = getattr(args, 'post_train_eval_device', 'cuda:0')
    eval_output_dir = output_map.parent / 'eval_results' / session
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_opt = argparse.Namespace(
        scene=args.scene,
        network=output_map,
        dinov2_path=args.dinov2_path,
        device=eval_device,
        image_resolution=args.image_resolution,
        session=session,
        eval_output_dir=eval_output_dir,
        render_visualization=False,
    )
    _logger.info('Running in-process eval: scene=%s network=%s session=%s device=%s output_dir=%s',
                 args.scene, output_map, session, eval_device, eval_output_dir)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = run_evaluation(eval_opt)
    except Exception as exc:
        _logger.warning('Post-training evaluation failed for session %s: %s', session, exc, exc_info=True)
        return None

    summary_file = eval_output_dir / f'eval_summary_{args._scene_display_name}_{session}.txt'
    parsed = _parse_eval_summary(summary_file)
    if parsed is None:
        test_log_file = eval_output_dir / f'test_{args._scene_display_name}_{session}.txt'
        parsed = _parse_test_log(test_log_file)
        if parsed is None:
            return {
                'median_rotation_deg': result['median_rErr'],
                'median_translation_cm': result['median_tErr'],
                'accuracy_25cm5deg_pct': result['pct25_5'],
                'accuracy_10cm5deg_pct': result['pct10_5'],
                'accuracy_5cm5deg_pct': result['pct5'],
                'accuracy_2cm2deg_pct': result['pct2'],
                'accuracy_1cm1deg_pct': result['pct1'],
                'avg_time_per_frame_ms': result['avg_time'] * 1000.0,
                'total_frames': result['total_frames'],
            }
        _logger.warning('Eval summary not found, fell back to test log: %s', test_log_file)
    return parsed


def run_iterative_training(args, trainer_factory=TrainerACEDINOv2, eval_fn=run_post_eval):
    iter_buffer_size = get_iter_buffer_size(args)
    trainer_args = build_trainer_options(args)
    trainer = trainer_factory(trainer_args)
    trainer.training_start = time.time()

    out_base = Path(args.output_map).parent
    best_score = -float('inf')
    best_iter = None

    for iteration_idx in range(args.iterations):
        _logger.info('=== Iteration %d/%d ===', iteration_idx + 1, args.iterations)

        if iteration_idx > 0:
            trainer.reset_optimizer_scheduler(
                keep_optimizer_state=not args.reset_optimizer_each_iter,
                buffer_size=iter_buffer_size,
            )

        buffer_start = time.time()
        trainer.create_training_buffer(buffer_size=iter_buffer_size)
        _logger.info('Filled training buffer in %.1fs.', time.time() - buffer_start)

        for trainer.epoch in range(args.epochs):
            trainer.run_epoch()

        is_best = False
        eval_result = None
        if args.eval_each_iteration:
            iter_ckpt = out_base / f'{args.output_map.stem}.iter_{iteration_idx + 1:02d}.tmp.pt'
            trainer.save_model(iter_ckpt)
            eval_result = eval_fn(args, iter_ckpt, session=f'iter_{iteration_idx + 1:02d}')
            score = _score_eval(eval_result, args.best_metric)
            is_best = score > best_score
            if is_best:
                best_score = score
                best_iter = iteration_idx + 1
                os.replace(iter_ckpt, args.output_map)
                _logger.info('Updated best checkpoint at iter %d (metric=%s score=%.4f).',
                             best_iter, args.best_metric, score)
            elif iter_ckpt.exists():
                if args.keep_best_only:
                    iter_ckpt.unlink()
                else:
                    iter_ckpt.rename(out_base / f'{args.output_map.stem}.iter_{iteration_idx + 1:02d}.pt')
        else:
            _logger.info('Iteration %d completed without per-iter eval.', iteration_idx + 1)

        if eval_result is not None:
            pct5 = eval_result.get('accuracy_5cm5deg_pct', float('nan'))
            pct25_5 = eval_result.get('accuracy_25cm5deg_pct', float('nan'))
            _logger.info('Iter %02d summary: median=%.2fdeg/%.2fcm pct5=%.2f pct25_5=%.2f %s',
                         iteration_idx + 1,
                         eval_result['median_rotation_deg'],
                         eval_result['median_translation_cm'],
                         pct5,
                         pct25_5,
                         '[BEST]' if is_best else '')

    if not args.output_map.exists():
        trainer.save_model(args.output_map)
    _logger.info('Done. Total time: %.1fs. best_iter=%s best_score=%.4f',
                 time.time() - trainer.training_start,
                 best_iter,
                 best_score if best_iter is not None else float('nan'))
    return trainer


def log_configuration(args):
    _logger.info('=' * 80)
    _logger.info('Training ACE with DINOv2 Encoder (Iterative Baseline)')
    _logger.info('=' * 80)
    _logger.info('Scene: %s', args.scene)
    _logger.info('Run Dir: %s', args.run_dir)
    _logger.info('Output: %s', args.output_map)
    _logger.info('DINOv2 weights: %s', args.dinov2_path)
    _logger.info('Freeze backbone: %s', args.freeze_backbone)
    _logger.info('Image resolution: %s', args.image_resolution)
    _logger.info('Epochs per iteration: %s', args.epochs)
    _logger.info('Iterations: %s', args.iterations)
    _logger.info('Iter buffer size: %s', get_iter_buffer_size(args))
    _logger.info('Batch size: %s', args.batch_size)
    _logger.info('Keep best only: %s | Best metric: %s', args.keep_best_only, args.best_metric)
    _logger.info('Device: %s', args.device)
    _logger.info('=' * 80)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prepare_args(args)
    except Exception as exc:
        _logger.error('%s', exc)
        return 1

    log_configuration(args)
    run_iterative_training(args)

    _logger.info('Training completed successfully!')
    if args.eval_after_train:
        _logger.info('=' * 80)
        _logger.info('Running post-training evaluation on test set')
        _logger.info('=' * 80)
        run_post_eval(args, args.output_map, session=args.eval_session)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
