from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseTask


class BinaryCategorizationTask(BaseTask):
    name = "BinaryCategorizationTask"

    def __init__(
        self,
        seq_len: int = 50,
        fix_len: int = 10,
        stim_len: int = 25,
        response_len: int = 15,
        noise_std: float = 0.20,
        coherence_levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        assert fix_len + stim_len + response_len == seq_len
        self._seq_len = seq_len
        self.fix_len = fix_len
        self.stim_len = stim_len
        self.response_len = response_len
        self.noise_std = noise_std
        self.coherence_levels = coherence_levels

    @property
    def input_dim(self) -> int:
        return 2  # evidence, go

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def epochs(self):
        return {
            "fixation": (0, self.fix_len),
            "stimulus": (self.fix_len, self.fix_len + self.stim_len),
            "response": (self.fix_len + self.stim_len, self.seq_len),
        }

    def default_input_masks(self) -> Dict[str, np.ndarray]:
        return {
            "a": np.array([1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0], dtype=np.float32),
        }

    def _allocate_meta(self, batch_size: int):
        return {
            "coherence": np.zeros((batch_size,), dtype=np.float32),
            "label": np.zeros((batch_size,), dtype=np.float32),
        }

    def _generate_trial(self, rng: np.random.Generator):
        inputs = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.seq_len, 1), dtype=np.float32)
        mask = np.zeros((self.seq_len, 1), dtype=np.float32)

        coherence = float(rng.choice(self.coherence_levels))
        coherence *= float(rng.choice([-1.0, 1.0]))
        stim_start, stim_end = self.epochs["stimulus"]
        resp_start, resp_end = self.epochs["response"]

        inputs[stim_start:stim_end, 0] = coherence + rng.normal(0.0, self.noise_std, size=(stim_end - stim_start))
        inputs[resp_start:resp_end, 1] = 1.0

        label = 1.0 if coherence >= 0 else -1.0
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0

        meta = {"coherence": coherence, "label": label}
        return {"inputs": inputs, "targets": targets, "mask": mask, "meta": meta}


class DelayedCategorizationTask(BinaryCategorizationTask):
    name = "DelayedCategorizationTask"

    def __init__(
        self,
        fix_len: int = 10,
        stim_len: int = 20,
        delay_len: int = 20,
        response_len: int = 15,
        noise_std: float = 0.20,
        coherence_levels: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
        seed: int = 0,
    ):
        self.delay_len = delay_len
        super().__init__(
            seq_len=fix_len + stim_len + delay_len + response_len,
            fix_len=fix_len,
            stim_len=stim_len,
            response_len=response_len + delay_len,
            noise_std=noise_std,
            coherence_levels=coherence_levels,
            seed=seed,
        )
        self.response_len = response_len
        self._seq_len = fix_len + stim_len + delay_len + response_len
        self.fix_len = fix_len
        self.stim_len = stim_len

    @property
    def epochs(self):
        stim_end = self.fix_len + self.stim_len
        delay_end = stim_end + self.delay_len
        return {
            "fixation": (0, self.fix_len),
            "stimulus": (self.fix_len, stim_end),
            "delay": (stim_end, delay_end),
            "response": (delay_end, self.seq_len),
        }

    def _generate_trial(self, rng: np.random.Generator):
        trial = super()._generate_trial(rng)
        inputs = trial["inputs"]
        targets = trial["targets"]
        mask = trial["mask"]

        # Clear inherited response timing and rebuild.
        mask[:] = 0.0
        targets[:] = 0.0
        resp_start, resp_end = self.epochs["response"]
        inputs[:, 1] = 0.0
        inputs[resp_start:resp_end, 1] = 1.0
        label = trial["meta"]["label"]
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0
        return trial
