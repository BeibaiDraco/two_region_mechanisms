#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models import build_model
from src.tasks import build_task
from src.train import Trainer
from src.utils.config import load_config, merge_overrides
from src.utils.seeds import set_seed


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", nargs="*", default=None, help="Overrides like train.steps=100 model.params.hidden_size_a=32")
    args = parser.parse_args()

    cfg = merge_overrides(load_config(args.config), args.override)
    set_seed(int(cfg.get("seed", 0)))

    device = resolve_device(str(cfg.get("device", "auto")))
    cfg["device"] = device
    cfg.setdefault("train", {})
    cfg["train"]["device"] = device

    task = build_task(cfg["task"])
    model_cfg = dict(cfg["model"])
    params = dict(model_cfg.get("params", {}))
    masks = task.default_input_masks()
    params.setdefault("input_dim", task.input_dim)
    params.setdefault("output_dim", task.output_dim)
    params.setdefault("input_mask_a", masks["a"].tolist())
    params.setdefault("input_mask_b", masks["b"].tolist())
    model_cfg["params"] = params

    model = build_model(model_cfg).to(device)
    run_dir = Path(cfg.get("output_dir", "outputs/default_run"))

    trainer = Trainer(model=model, task=task, cfg=cfg, run_dir=run_dir)
    trainer.train()
    print(f"Training finished. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
