from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from src.tasks.base import TaskBatch


def default_eval_cache_path(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    return run_dir / "metrics" / "eval_cache.pt"


def _batch_to_payload(batch: TaskBatch, split: str, n_trials: int) -> Dict[str, Any]:
    return {
        "split": split,
        "n_trials": int(n_trials),
        "inputs": batch.inputs.detach().cpu(),
        "targets": batch.targets.detach().cpu(),
        "mask": batch.mask.detach().cpu(),
        "meta": {key: value.detach().cpu() for key, value in batch.meta.items()},
    }


def save_eval_cache(batch: TaskBatch, path: str | Path, split: str = "val", n_trials: int | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = batch.inputs.shape[1] if n_trials is None else n_trials
    torch.save(_batch_to_payload(batch, split=split, n_trials=batch_size), path)
    return path


def load_eval_cache(path: str | Path, device: str = "cpu") -> TaskBatch:
    payload = torch.load(Path(path), map_location=device)
    return TaskBatch(
        inputs=payload["inputs"].to(device),
        targets=payload["targets"].to(device),
        mask=payload["mask"].to(device),
        meta={key: value.to(device) for key, value in payload["meta"].items()},
    )


def load_or_create_eval_batch(
    task,
    n_trials: int,
    device: str = "cpu",
    split: str = "val",
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> TaskBatch:
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not refresh:
            return load_eval_cache(cache_path, device=device)

    batch = task.sample_batch(n_trials, split=split, device=device)
    if cache_path is not None:
        save_eval_cache(batch, cache_path, split=split, n_trials=n_trials)
    return batch
