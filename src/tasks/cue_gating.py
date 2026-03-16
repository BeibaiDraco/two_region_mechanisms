from __future__ import annotations

import numpy as np

from .base import BaseTask


class CueGatingTask(BaseTask):
    name = "CueGatingTask"
    RULE_NAMES = ("report", "ignore", "invert", "hold")

    def __init__(
        self,
        cue_len: int = 8,
        stim_len: int = 22,
        delay_len: int = 18,
        response_len: int = 16,
        noise_std: float = 0.20,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.cue_len = cue_len
        self.stim_len = stim_len
        self.delay_len = delay_len
        self.response_len = response_len
        self.noise_std = noise_std
        self._seq_len = cue_len + stim_len + delay_len + response_len

    @property
    def input_dim(self) -> int:
        return 6  # evidence, report, ignore, invert, hold, go

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
            "a": np.array([1, 1, 1, 1, 1, 0], dtype=np.float32),
            "b": np.array([0, 1, 1, 1, 1, 1], dtype=np.float32),
        }

    def _allocate_meta(self, batch_size: int):
        return {
            "rule": np.zeros((batch_size,), dtype=np.int64),
            "label": np.zeros((batch_size,), dtype=np.float32),
        }

    def _generate_trial(self, rng: np.random.Generator):
        inputs = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.seq_len, 1), dtype=np.float32)
        mask = np.zeros((self.seq_len, 1), dtype=np.float32)

        rule = int(rng.integers(0, 4))
        evidence = float(rng.choice([0.2, 0.4, 0.6, 0.8]) * rng.choice([-1.0, 1.0]))
        cue_start, cue_end = self.epochs["cue"]
        stim_start, stim_end = self.epochs["stimulus"]
        resp_start, resp_end = self.epochs["response"]

        inputs[cue_start:cue_end, 1 + rule] = 1.0

        if rule == 3:  # hold
            pulse_len = max(3, self.stim_len // 4)
            mem_val = evidence
            inputs[stim_start:stim_start + pulse_len, 0] = mem_val
            inputs[stim_start + pulse_len:stim_end, 0] = rng.normal(0.0, self.noise_std * 1.5, size=(stim_end - stim_start - pulse_len))
            label = 1.0 if mem_val >= 0 else -1.0
        else:
            inputs[stim_start:stim_end, 0] = evidence + rng.normal(0.0, self.noise_std, size=(stim_end - stim_start))
            if rule == 0:
                label = 1.0 if evidence >= 0 else -1.0
            elif rule == 1:
                label = 0.0
            else:  # invert
                label = -1.0 if evidence >= 0 else 1.0

        inputs[resp_start:resp_end, 5] = 1.0
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0

        return {"inputs": inputs, "targets": targets, "mask": mask, "meta": {"rule": rule, "label": label}}
