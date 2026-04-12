import unittest
from types import SimpleNamespace

import torch
from torch import nn

from trainer_dinov2 import TrainerACEDINOv2


class TrainerDINOv2IterativeSupportTest(unittest.TestCase):
    def _make_trainer_stub(self, training_buffer_size=10, current_buffer_size=None, batch_size=4, epochs=2):
        trainer = TrainerACEDINOv2.__new__(TrainerACEDINOv2)
        trainer.options = SimpleNamespace(
            training_buffer_size=training_buffer_size,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate_min=1e-4,
            learning_rate_max=1e-3,
            use_half=False,
        )
        trainer.regressor = nn.Linear(2, 2)
        trainer.device = torch.device("cpu")
        trainer.iteration = 0
        trainer.epoch = 0
        trainer.training_generator = torch.Generator().manual_seed(0)
        trainer._current_buffer_size = (
            training_buffer_size if current_buffer_size is None else current_buffer_size
        )

        buffer_size = trainer._current_buffer_size
        trainer.training_buffer = {
            "features": torch.randn(buffer_size, 2),
            "target_px": torch.randn(buffer_size, 2),
            "gt_poses_inv": torch.randn(buffer_size, 3, 4),
            "intrinsics": torch.randn(buffer_size, 3, 3),
            "intrinsics_inv": torch.randn(buffer_size, 3, 3),
        }
        return trainer

    def test_reset_optimizer_scheduler_preserves_optimizer_when_requested(self):
        trainer = self._make_trainer_stub(training_buffer_size=16, batch_size=4, epochs=3)

        trainer._init_optimizer_scheduler()
        original_optimizer = trainer.optimizer

        trainer.reset_optimizer_scheduler(keep_optimizer_state=True, buffer_size=8)

        self.assertIs(trainer.optimizer, original_optimizer)
        self.assertEqual(trainer.scheduler.total_steps, 3 * (8 // 4))

    def test_reset_optimizer_scheduler_recreates_optimizer_when_requested(self):
        trainer = self._make_trainer_stub(training_buffer_size=16, batch_size=4, epochs=2)

        trainer._init_optimizer_scheduler()
        original_optimizer = trainer.optimizer

        trainer.reset_optimizer_scheduler(keep_optimizer_state=False, buffer_size=12)

        self.assertIsNot(trainer.optimizer, original_optimizer)
        self.assertEqual(trainer.scheduler.total_steps, 2 * (12 // 4))

    def test_run_epoch_uses_current_buffer_size_for_iteration_batches(self):
        trainer = self._make_trainer_stub(training_buffer_size=10, current_buffer_size=6, batch_size=4, epochs=2)
        observed_batch_sizes = []

        def fake_training_step(features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
            observed_batch_sizes.append(features_bC.shape[0])

        trainer.training_step = fake_training_step

        trainer.run_epoch()

        self.assertEqual(observed_batch_sizes, [4])
        self.assertEqual(trainer.iteration, 1)


if __name__ == "__main__":
    unittest.main()
