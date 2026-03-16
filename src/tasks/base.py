from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass
class TaskBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    meta: Dict[str, torch.Tensor]


class BaseTask:
    name = "BaseTask"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.train_rng = np.random.default_rng(seed)
        self.val_rng = np.random.default_rng(seed + 999)

    @property
    def input_dim(self) -> int:
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        return 1

    @property
    def seq_len(self) -> int:
        raise NotImplementedError

    @property
    def epochs(self) -> Dict[str, Tuple[int, int]]:
        raise NotImplementedError

    def default_input_masks(self) -> Dict[str, np.ndarray]:
        return {
            "a": np.ones(self.input_dim, dtype=np.float32),
            "b": np.ones(self.input_dim, dtype=np.float32),
        }

    def sample_batch(self, batch_size: int, split: str = "train", device: str = "cpu") -> TaskBatch:
        rng = self.train_rng if split == "train" else self.val_rng
        inputs = np.zeros((self.seq_len, batch_size, self.input_dim), dtype=np.float32)
        targets = np.zeros((self.seq_len, batch_size, self.output_dim), dtype=np.float32)
        mask = np.zeros((self.seq_len, batch_size, self.output_dim), dtype=np.float32)
        meta = self._allocate_meta(batch_size)

        for batch_idx in range(batch_size):
            trial = self._generate_trial(rng)
            inputs[:, batch_idx, :] = trial["inputs"]
            targets[:, batch_idx, :] = trial["targets"]
            mask[:, batch_idx, :] = trial["mask"]
            for key, value in trial["meta"].items():
                meta[key][batch_idx] = value

        meta["seq_len"] = np.full((batch_size,), self.seq_len, dtype=np.int64)
        return TaskBatch(
            inputs=torch.tensor(inputs, device=device),
            targets=torch.tensor(targets, device=device),
            mask=torch.tensor(mask, device=device),
            meta={k: torch.tensor(v, device=device) for k, v in meta.items()},
        )

    def _allocate_meta(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {}

    def _generate_trial(self, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def compute_accuracy(self, outputs: torch.Tensor, batch: TaskBatch, zero_tol: float = 0.35) -> float:
        # outputs: [T, B, 1]
        out = outputs[..., 0].detach()
        tgt = batch.targets[..., 0]
        mask = batch.mask[..., 0]
        trial_scores = []
        for b in range(out.shape[1]):
            active = mask[:, b] > 0
            if active.sum() == 0:
                continue
            pred = out[active, b].mean().item()
            target = tgt[active, b].mean().item()
            if abs(target) < 1e-6:
                correct = abs(pred) < zero_tol
            else:
                correct = np.sign(pred) == np.sign(target)
            trial_scores.append(float(correct))
        return float(np.mean(trial_scores)) if trial_scores else 0.0

    def epoch_steps(self, name: str) -> list[int]:
        if name not in self.epochs:
            raise KeyError(f"Unknown epoch {name!r} for {self.name}")
        start, end = self.epochs[name]
        return list(range(start, end))
