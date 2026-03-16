from __future__ import annotations

import numpy as np

from .base import BaseTask


class RedundantInputControlTask(BaseTask):
    name = "RedundantInputControlTask"

    def __init__(
        self,
        fix_len: int = 10,
        stim_len: int = 22,
        response_len: int = 16,
        noise_std: float = 0.18,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.fix_len = fix_len
        self.stim_len = stim_len
        self.response_len = response_len
        self.noise_std = noise_std
        self._seq_len = fix_len + stim_len + response_len

    @property
    def input_dim(self) -> int:
        return 3  # evidence_to_A, evidence_to_B, go

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def epochs(self):
        stim_start = self.fix_len
        stim_end = stim_start + self.stim_len
        return {
            "fixation": (0, self.fix_len),
            "stimulus": (stim_start, stim_end),
            "response": (stim_end, self.seq_len),
        }

    def default_input_masks(self):
        return {
            "a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 1.0, 1.0], dtype=np.float32),
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

        coherence = float(rng.choice([0.2, 0.4, 0.6, 0.8]) * rng.choice([-1.0, 1.0]))
        stim_start, stim_end = self.epochs["stimulus"]
        resp_start, resp_end = self.epochs["response"]

        signal = coherence + rng.normal(0.0, self.noise_std, size=(stim_end - stim_start))
        inputs[stim_start:stim_end, 0] = signal
        inputs[stim_start:stim_end, 1] = signal
        inputs[resp_start:resp_end, 2] = 1.0

        label = 1.0 if coherence >= 0 else -1.0
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0

        return {"inputs": inputs, "targets": targets, "mask": mask, "meta": {"coherence": coherence, "label": label}}
