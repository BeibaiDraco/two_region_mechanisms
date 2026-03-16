from __future__ import annotations

from typing import Any, Dict

from .categorization import BinaryCategorizationTask, DelayedCategorizationTask
from .context_decision import ContextDependentDecisionTask
from .cue_gating import CueGatingTask
from .state_setting import StateSettingTask
from .redundant_control import RedundantInputControlTask

TASK_REGISTRY = {
    "BinaryCategorizationTask": BinaryCategorizationTask,
    "DelayedCategorizationTask": DelayedCategorizationTask,
    "ContextDependentDecisionTask": ContextDependentDecisionTask,
    "CueGatingTask": CueGatingTask,
    "StateSettingTask": StateSettingTask,
    "RedundantInputControlTask": RedundantInputControlTask,
}


def build_task(task_cfg: Dict[str, Any]):
    name = task_cfg["name"]
    params = task_cfg.get("params", {})
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {name}")
    return TASK_REGISTRY[name](**params)
