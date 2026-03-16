#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis.interventions import available_epoch_names, build_intervention
from src.analysis.readouts import default_readout
from src.analysis.rollout import load_run_artifacts, run_model_rollout
from src.utils.io import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg, task, model, _ = load_run_artifacts(args.run_dir, checkpoint_name=args.checkpoint, map_location="cpu")
    batch = task.sample_batch(args.batch_size, split="val", device="cpu")
    base = run_model_rollout(model, batch, intervention=None)
    base_acc = task.compute_accuracy(base["outputs"], batch)

    rows = []
    for epoch_name in available_epoch_names(task):
        intervention = build_intervention(task, "decouple", epoch=epoch_name)
        roll = run_model_rollout(model, batch, intervention=intervention)
        acc = task.compute_accuracy(roll["outputs"], batch)
        delta_q = (default_readout(base) - default_readout(roll)).abs().max().item()
        rows.append({
            "epoch": epoch_name,
            "base_accuracy": base_acc,
            "ablated_accuracy": acc,
            "accuracy_drop": base_acc - acc,
            "max_abs_delta_q": delta_q,
        })

    df = pd.DataFrame(rows)
    run_dir = Path(args.run_dir)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "ablations.csv", index=False)
    save_json({"rows": rows}, metrics_dir / "ablations.json")
    print(df)


if __name__ == "__main__":
    main()
