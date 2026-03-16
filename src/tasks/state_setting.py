from __future__ import annotations

import numpy as np

from .base import BaseTask


class StateSettingTask(BaseTask):
    name = "StateSettingTask"
    MODE_NAMES = ("integrate", "memory", "transient")

    def __init__(
        self,
        cue_len: int = 8,
        signal_len: int = 24,
        delay_len: int = 20,
        response_len: int = 16,
        noise_std: float = 0.20,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.cue_len = cue_len
        self.signal_len = signal_len
        self.delay_len = delay_len
        self.response_len = response_len
        self.noise_std = noise_std
        self._seq_len = cue_len + signal_len + delay_len + response_len

    @property
    def input_dim(self) -> int:
        return 5  # signal, mode_integrate, mode_memory, mode_transient, go

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def epochs(self):
        sig_start = self.cue_len
        sig_end = sig_start + self.signal_len
        delay_end = sig_end + self.delay_len
        return {
            "cue": (0, self.cue_len),
            "signal": (sig_start, sig_end),
            "delay": (sig_end, delay_end),
            "response": (delay_end, self.seq_len),
        }

    def default_input_masks(self):
        return {
            "a": np.array([1, 1, 1, 1, 0], dtype=np.float32),
            "b": np.array([0, 1, 1, 1, 1], dtype=np.float32),
        }

    def _allocate_meta(self, batch_size: int):
        return {
            "mode": np.zeros((batch_size,), dtype=np.int64),
            "label": np.zeros((batch_size,), dtype=np.float32),
        }

    def _generate_trial(self, rng: np.random.Generator):
        inputs = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.seq_len, 1), dtype=np.float32)
        mask = np.zeros((self.seq_len, 1), dtype=np.float32)

        mode = int(rng.integers(0, 3))
        cue_start, cue_end = self.epochs["cue"]
        sig_start, sig_end = self.epochs["signal"]
        resp_start, resp_end = self.epochs["response"]
        inputs[cue_start:cue_end, 1 + mode] = 1.0

        if mode == 0:  # integrate
            coherence = float(rng.choice([0.15, 0.3, 0.45, 0.6]) * rng.choice([-1.0, 1.0]))
            signal = coherence + rng.normal(0.0, self.noise_std, size=(sig_end - sig_start))
            inputs[sig_start:sig_end, 0] = signal
            label = 1.0 if signal.sum() >= 0 else -1.0
        elif mode == 1:  # memory
            mem_val = float(rng.choice([0.4, 0.7]) * rng.choice([-1.0, 1.0]))
            pulse_len = max(3, self.signal_len // 5)
            inputs[sig_start:sig_start + pulse_len, 0] = mem_val
            inputs[sig_start + pulse_len:sig_end, 0] = rng.normal(0.0, self.noise_std * 1.3, size=(sig_end - sig_start - pulse_len))
            label = 1.0 if mem_val >= 0 else -1.0
        else:  # transient
            split = self.signal_len // 2
            first = float(rng.choice([0.3, 0.6]) * rng.choice([-1.0, 1.0]))
            second = float(rng.choice([0.3, 0.6]) * rng.choice([-1.0, 1.0]))
            inputs[sig_start:sig_start + split, 0] = first
            inputs[sig_start + split:sig_end, 0] = second
            inputs[sig_start:sig_end, 0] += rng.normal(0.0, self.noise_std * 0.7, size=(sig_end - sig_start))
            label = 1.0 if second >= 0 else -1.0

        inputs[resp_start:resp_end, 4] = 1.0
        targets[resp_start:resp_end, 0] = label
        mask[resp_start:resp_end, 0] = 1.0

        return {"inputs": inputs, "targets": targets, "mask": mask, "meta": {"mode": mode, "label": label}}
