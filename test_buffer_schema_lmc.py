#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import torch

from trainer_dinov2_lmc import TrainerACEDINOv2LMC


class BufferSchemaLMCTest(unittest.TestCase):
    def _make_trainer_stub(self, use_half=False, buffer_on_cpu=True, feature_dim=1024):
        trainer = TrainerACEDINOv2LMC.__new__(TrainerACEDINOv2LMC)
        trainer.options = SimpleNamespace(
            use_half=use_half,
            buffer_on_cpu=buffer_on_cpu,
            s1_buffer_keep_ratio=0.25,
            s1_buffer_refill_ratio=None,
        )
        trainer.device = torch.device("cpu")
        trainer.regressor = SimpleNamespace(feature_dim=feature_dim)
        return trainer

    def _make_buffer(self, n=32, feature_dim=1024, feature_dtype=torch.float32):
        return {
            "features": torch.empty((n, feature_dim), dtype=feature_dtype),
            "target_px": torch.empty((n, 2), dtype=torch.float32),
            "gt_poses_inv": torch.empty((n, 3, 4), dtype=torch.float32),
            "intrinsics": torch.empty((n, 3, 3), dtype=torch.float32),
            "intrinsics_inv": torch.empty((n, 3, 3), dtype=torch.float32),
        }

    def test_validate_fused_buffer_schema_accepts_expected_layout(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        buffer_dict = self._make_buffer()

        trainer._validate_training_buffer_schema(
            buffer_dict=buffer_dict,
            schema_name="fused_buffer",
            expected_size=32,
        )

    def test_validate_raw_buffer_schema_rejects_wrong_feature_dtype(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        buffer_dict = self._make_buffer(feature_dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "features.*dtype"):
            trainer._validate_training_buffer_schema(
                buffer_dict=buffer_dict,
                schema_name="raw_buffer",
                expected_size=32,
            )

    def test_validate_buffer_schema_rejects_wrong_target_shape(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        buffer_dict = self._make_buffer()
        buffer_dict["target_px"] = torch.empty((32, 3), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "target_px.*shape"):
            trainer._validate_training_buffer_schema(
                buffer_dict=buffer_dict,
                schema_name="fused_buffer",
                expected_size=32,
            )

    def test_validate_buffer_schema_rejects_missing_key(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        buffer_dict = self._make_buffer()
        del buffer_dict["intrinsics_inv"]

        with self.assertRaisesRegex(ValueError, "missing keys"):
            trainer._validate_training_buffer_schema(
                buffer_dict=buffer_dict,
                schema_name="raw_buffer",
                expected_size=32,
            )

    def test_resolve_partial_refill_counts_from_keep_ratio(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)

        keep_count, refill_count = trainer._resolve_s1_partial_refill_counts(total_size=32)

        self.assertEqual(keep_count, 8)
        self.assertEqual(refill_count, 24)

    def test_resolve_partial_refill_counts_rejects_inconsistent_ratios(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        trainer.options.s1_buffer_keep_ratio = 0.6
        trainer.options.s1_buffer_refill_ratio = 0.3

        with self.assertRaisesRegex(ValueError, "sum to 1"):
            trainer._resolve_s1_partial_refill_counts(total_size=32)

    def test_merge_partial_raw_buffers_preserves_schema_and_size(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        kept_buffer = self._make_buffer(n=8)
        refill_buffer = self._make_buffer(n=24)

        merged = trainer._merge_training_buffers(
            schema_name="raw_buffer",
            buffers=[kept_buffer, refill_buffer],
            expected_size=32,
        )

        trainer._validate_training_buffer_schema(
            buffer_dict=merged,
            schema_name="raw_buffer",
            expected_size=32,
        )
        self.assertEqual(merged["features"].shape[0], 32)

    def test_resolve_partial_refill_counts_rejects_zero_keep_or_refill(self):
        trainer = self._make_trainer_stub(use_half=False, buffer_on_cpu=True)
        trainer.options.s1_buffer_keep_ratio = 1.0

        with self.assertRaisesRegex(ValueError, "both keep/refill counts > 0"):
            trainer._resolve_s1_partial_refill_counts(total_size=32)


if __name__ == "__main__":
    unittest.main()
