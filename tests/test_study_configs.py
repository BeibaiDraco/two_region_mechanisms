from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis.eval_cache import load_eval_cache, save_eval_cache
from src.tasks import build_task
from src.utils.config import load_config


class StudyConfigTest(unittest.TestCase):
    def test_new_configs_exist(self):
        repo = Path(__file__).resolve().parents[1]
        delayed = load_config(repo / "configs" / "tasks" / "delayed.yaml")
        redundant = load_config(repo / "configs" / "tasks" / "redundant.yaml")
        lowrank = load_config(repo / "configs" / "models" / "lowrank.yaml")

        self.assertEqual(delayed["name"], "DelayedCategorizationTask")
        self.assertEqual(redundant["name"], "RedundantInputControlTask")
        self.assertEqual(lowrank["name"], "TwoRegionLowRankCommRNN")

    def test_eval_cache_round_trip(self):
        task = build_task({"name": "BinaryCategorizationTask", "params": {"seed": 3}})
        batch = task.sample_batch(5, split="val", device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval_cache.pt"
            save_eval_cache(batch, path, split="val", n_trials=5)
            loaded = load_eval_cache(path, device="cpu")

        self.assertEqual(tuple(batch.inputs.shape), tuple(loaded.inputs.shape))
        self.assertEqual(set(batch.meta.keys()), set(loaded.meta.keys()))


if __name__ == "__main__":
    unittest.main()
