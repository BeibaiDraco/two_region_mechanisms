from __future__ import annotations

from typing import Dict, Optional


def build_intervention(task, kind: str, epoch: Optional[str] = None) -> Dict:
    active_steps = None if epoch is None else task.epoch_steps(epoch)
    return {"type": kind, "active_steps": active_steps}


def available_epoch_names(task) -> list[str]:
    return list(task.epochs.keys())
