import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import train_ace_dinov2_iterative as train_iter


class FakeTrainer:
    def __init__(self, options):
        self.options = options
        self.buffer_calls = []
        self.reset_calls = []
        self.run_epoch_calls = []
        self.save_calls = []
        self.epoch = None

    def create_training_buffer(self, buffer_size=None):
        self.buffer_calls.append(buffer_size)

    def reset_optimizer_scheduler(self, keep_optimizer_state=False, buffer_size=None):
        self.reset_calls.append((keep_optimizer_state, buffer_size))

    def run_epoch(self):
        self.run_epoch_calls.append(self.epoch)

    def save_model(self, output_path):
        self.save_calls.append(Path(output_path))


class TrainACEDINOv2IterativeTest(unittest.TestCase):
    def _make_args(self, tmpdir, **overrides):
        base = dict(
            scene=Path(tmpdir) / 'dataset' / 'scene1' / 'train',
            output_map=Path(tmpdir) / 'scene1_model.pt',
            experiment_root=Path(tmpdir) / 'output',
            dinov2_path=Path('/tmp/dinov2_dummy.pth'),
            freeze_backbone=True,
            device='cuda:0',
            num_head_blocks=4,
            use_homogeneous=True,
            training_buffer_size=100,
            buffer_batch_size=1,
            buffer_image_width=None,
            samples_per_image=512,
            epochs=2,
            batch_size=10,
            learning_rate_min=1e-4,
            learning_rate_max=1e-3,
            image_resolution=518,
            use_aug=True,
            aug_rotation=15,
            aug_scale=1.5,
            repro_loss_type='dyntanh',
            repro_loss_soft_clamp=50,
            repro_loss_soft_clamp_min=1,
            repro_loss_schedule='circle',
            repro_loss_hard_clamp=1000,
            depth_min=0.1,
            depth_max=1000,
            depth_target=10,
            use_half=True,
            eval_after_train=True,
            eval_session='post_train',
            post_train_eval_device='cuda:0',
            iterations=3,
            iter_buffer_size=40,
            reset_optimizer_each_iter=False,
            eval_each_iteration=False,
            _scene_display_name='scene1',
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_build_parser_adds_iterative_arguments_with_expected_defaults(self):
        parser = train_iter.build_parser()
        args = parser.parse_args(['scene_path', 'model.pt'])

        self.assertEqual(args.iterations, 1)
        self.assertIsNone(args.iter_buffer_size)
        self.assertFalse(args.reset_optimizer_each_iter)
        self.assertTrue(args.eval_each_iteration)

    def test_build_trainer_options_expands_total_buffer_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(tmpdir, iterations=4, iter_buffer_size=25, training_buffer_size=60)

            trainer_args = train_iter.build_trainer_options(args)

            self.assertEqual(trainer_args.training_buffer_size, 100)
            self.assertEqual(args.training_buffer_size, 60)
            self.assertIsNot(trainer_args, args)

    def test_run_iterative_training_reuses_optimizer_state_and_saves_final_model(self):
        created = []

        def trainer_factory(options):
            trainer = FakeTrainer(options)
            created.append(trainer)
            return trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(
                tmpdir,
                iterations=3,
                iter_buffer_size=40,
                eval_each_iteration=False,
                eval_after_train=False,
            )

            train_iter.run_iterative_training(args, trainer_factory=trainer_factory, eval_fn=lambda *a, **k: None)

        trainer = created[0]
        self.assertEqual(trainer.options.training_buffer_size, 120)
        self.assertEqual(trainer.buffer_calls, [40, 40, 40])
        self.assertEqual(trainer.reset_calls, [(True, 40), (True, 40)])
        self.assertEqual(trainer.run_epoch_calls, [0, 1, 0, 1, 0, 1])
        self.assertEqual(trainer.save_calls, [args.output_map])

    def test_run_iterative_training_emits_iteration_checkpoints_and_evals(self):
        created = []
        eval_calls = []

        def trainer_factory(options):
            trainer = FakeTrainer(options)
            created.append(trainer)
            return trainer

        def eval_fn(args, checkpoint_path, session):
            eval_calls.append((Path(checkpoint_path), session))

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(
                tmpdir,
                iterations=2,
                iter_buffer_size=30,
                eval_each_iteration=True,
                eval_after_train=False,
            )

            train_iter.run_iterative_training(args, trainer_factory=trainer_factory, eval_fn=eval_fn)

        trainer = created[0]
        self.assertEqual(
            trainer.save_calls,
            [
                args.output_map.parent / f'iter01_{args.output_map.name}',
                args.output_map.parent / f'iter02_{args.output_map.name}',
                args.output_map,
            ],
        )
        self.assertEqual(
            eval_calls,
            [
                (args.output_map.parent / f'iter01_{args.output_map.name}', 'iter01'),
                (args.output_map.parent / f'iter02_{args.output_map.name}', 'iter02'),
            ],
        )


if __name__ == '__main__':
    unittest.main()
