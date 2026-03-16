from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.models import build_model
from src.tasks import build_task
from src.utils.config import load_config
from src.utils.io import load_checkpoint


def load_run_artifacts(run_dir: str | Path, checkpoint_name: str = "best.pt", map_location: str = "cpu"):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    task = build_task(cfg["task"])
    # Inject task-driven model defaults.
    masks = task.default_input_masks()
    model_cfg = dict(cfg["model"])
    params = dict(model_cfg.get("params", {}))
    params.setdefault("input_dim", task.input_dim)
    params.setdefault("output_dim", task.output_dim)
    params.setdefault("input_mask_a", masks["a"].tolist())
    params.setdefault("input_mask_b", masks["b"].tolist())
    model_cfg["params"] = params
    model = build_model(model_cfg)

    ckpt_path = run_dir / "checkpoints" / checkpoint_name
    checkpoint = load_checkpoint(ckpt_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(map_location)
    model.eval()
    return cfg, task, model, checkpoint


@torch.no_grad()
def run_model_rollout(model, batch, intervention: Optional[Dict] = None):
    model.eval()
    return model(batch.inputs, intervention=intervention, add_noise=False)
