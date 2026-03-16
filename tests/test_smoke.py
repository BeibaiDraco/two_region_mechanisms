from __future__ import annotations

import unittest

import torch

from src.models import build_model
from src.tasks import build_task


class SmokeTest(unittest.TestCase):
    def test_binary_additive_forward(self):
        task = build_task({"name": "BinaryCategorizationTask", "params": {"seed": 1}})
        masks = task.default_input_masks()
        model = build_model({
            "name": "TwoRegionAdditiveRNN",
            "params": {
                "input_dim": task.input_dim,
                "output_dim": task.output_dim,
                "hidden_size_a": 16,
                "hidden_size_b": 16,
                "input_mask_a": masks["a"].tolist(),
                "input_mask_b": masks["b"].tolist(),
            },
        })
        batch = task.sample_batch(4, split="train", device="cpu")
        rollout = model(batch.inputs)
        self.assertEqual(rollout["outputs"].shape, (task.seq_len, 4, 1))

    def test_context_gated_forward(self):
        task = build_task({"name": "ContextDependentDecisionTask", "params": {"seed": 2}})
        masks = task.default_input_masks()
        model = build_model({
            "name": "TwoRegionGatedRNN",
            "params": {
                "input_dim": task.input_dim,
                "output_dim": task.output_dim,
                "hidden_size_a": 12,
                "hidden_size_b": 14,
                "input_mask_a": masks["a"].tolist(),
                "input_mask_b": masks["b"].tolist(),
            },
        })
        batch = task.sample_batch(3, split="val", device="cpu")
        rollout = model(batch.inputs)
        self.assertEqual(rollout["outputs"].shape[1], 3)
        self.assertTrue(torch.isfinite(rollout["outputs"]).all())


if __name__ == "__main__":
    unittest.main()
