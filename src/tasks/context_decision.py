from __future__ import annotations

import numpy as np

from .base import BaseTask


class ContextDependentDecisionTask(BaseTask):
    name = "ContextDependentDecisionTask"

    def __init__(
        self,
        cue_len: int = 8,
        stim_len: int = 24,
        delay_len: int = 10,
        response_len: int = 16,
        noise_std: float = 0.25,
        coherence_levels: tuple[float, ...] = (0.15, 0.3, 0.5, 0.7),
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.cue_len = cue_len
        self.stim_len = stim_len
        self.delay_len = delay_len
        self.response_len = response_len
        self.noise_std = noise_std
        self.coherence_levels = coherence_levels
        self._seq_len = cue_len + stim_len + delay_len + response_len

    @property
    def input_dim(self) -> int:
        return 5  # stim1, stim2, ctx1, ctx2, go

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def epochs(self):
        stim_start = self.cue_len
        stim_end = stim_start + self.stim_len
        delay_end = stim_end + self.delay_len
        return {
            "cue": (0, self.cue_len),
            "stimulus": (stim_start, stim_end),
            "delay": (stim_end, delay_end),
            "response": (delay_end, self.seq_len),
        }

    def default_input_masks(self):
        return {
            "a": np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "b": np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }

    def _allocate_meta(self, batch_size: int):
        return {
            "coh1": np.zeros((batch_size,), dtype=np.float32),
            "coh2": np.zeros((batch_size,), dtype=np.float32),
            "context": np.zeros((batch_size,), dtype=np.int64),
            "label": np.zeros((batch_size,), dtype=np.float32),
        }

    def _generate_trial(self, rng: np.random.Generator):
        inputs = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.seq_len, 1), dtype=np.float32)
        mask = np.zeros((self.seq_len, 1), dtype=np.float32)

        ctx = int(rng.integers(0, 2))
        coh1 = float(rng.choice(self.coherence_levels) * rng.choice([-1.0, 1.0]))
        coh2 = float(rng.choice(self.coherence_levels) * rng.choice([-1.0, 1.0]))
        cue_start, cue_end = self.epochs["cue"]
        stim_start, stim_end = self.epochs["stimulus"]
        resp_start, resp_end = self.epochs["response"]

        inputs[cue_start:cue_end, 2 + ctx] = 1.0
        inputs[stim_start:stim_end, 0] = coh1 + rng.normal(0.0, self.noise_std, size=(stim_end - stim_start))
        inputs[stim_start:stim_end, 1] = coh2 + rng.normal(0.0, self.noise_std, size=(stim_end - stim_start))
        inputs[resp_start:resp_end, 4] = 1.0

        label = 1.0 if (coh1 if ctx == 0 else coh2) >= 0 else -1.0
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0

        meta = {"coh1": coh1, "coh2": coh2, "context": ctx, "label": label}
        return {"inputs": inputs, "targets": targets, "mask": mask, "meta": meta}
