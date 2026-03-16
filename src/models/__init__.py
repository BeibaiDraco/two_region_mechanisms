from __future__ import annotations

from typing import Any, Dict

from .two_region_rnn import (
    TwoRegionAdditiveRNN,
    TwoRegionGatedRNN,
    TwoRegionLowRankCommRNN,
    TwoRegionReciprocalContextRNN,
)

MODEL_REGISTRY = {
    "TwoRegionAdditiveRNN": TwoRegionAdditiveRNN,
    "TwoRegionGatedRNN": TwoRegionGatedRNN,
    "TwoRegionLowRankCommRNN": TwoRegionLowRankCommRNN,
    "TwoRegionReciprocalContextRNN": TwoRegionReciprocalContextRNN,
}


def build_model(model_cfg: Dict[str, Any]):
    name = model_cfg["name"]
    params = model_cfg.get("params", {})
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}")
    return MODEL_REGISTRY[name](**params)
