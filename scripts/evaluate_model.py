#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis.mechanism_classification import classify_mechanism
from src.analysis.manifolds import pca_project
from src.analysis.rollout import load_run_artifacts
from src.utils.io import save_json
from src.utils.plotting import plot_hidden_projection, plot_readout_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg, task, model, _ = load_run_artifacts(args.run_dir, checkpoint_name=args.checkpoint, map_location="cpu")
    batch = task.sample_batch(args.batch_size, split="val", device="cpu")

    analysis_cfg = cfg.get("analysis", {})
    result = classify_mechanism(model, task, batch, thresholds=analysis_cfg)
    summary = result["summary"]

    run_dir = Path(args.run_dir)
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    q_c = result["effect"]["q_c"].cpu().numpy()
    q_0 = result["effect"]["q_0"].cpu().numpy()
    targets = batch.targets[..., 0].cpu().numpy()
    plot_readout_trajectories(q_c, q_0, targets, figures_dir / "coupled_vs_decoupled.png")

    r_a = result["coupled"]["r_a"][:, 0, :].cpu().numpy()
    r_b = result["coupled"]["r_b"][:, 0, :].cpu().numpy()
    pca_a = pca_project(r_a, n_components=2)
    pca_b = pca_project(r_b, n_components=2)
    plot_hidden_projection(pca_a["projection"], pca_b["projection"], figures_dir / "pca_hidden_states.png")

    save_json(summary, metrics_dir / "eval_metrics.json")
    print(pd.Series(summary))


if __name__ == "__main__":
    main()
