#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import load_json


def flatten_profile(run_dir: Path) -> dict:
    cfg = load_config(run_dir / "config.yaml")
    study = cfg.get("study", {})
    row = {
        "run_name": run_dir.name,
        "task": study.get("task_key"),
        "architecture": study.get("architecture_key"),
        "seed": study.get("seed", cfg.get("seed")),
    }

    metrics_dir = run_dir / "metrics"
    training = metrics_dir / "training_summary.csv"
    mechanism = metrics_dir / "mechanism_summary.json"
    final_profile = metrics_dir / "final_profile.json"
    threshold = metrics_dir / "threshold_robustness_summary.csv"

    if training.exists():
        row.update(pd.read_csv(training).iloc[0].to_dict())
    if mechanism.exists():
        row.update(load_json(mechanism))
    if final_profile.exists():
        final = load_json(final_profile)
        row["strategy_label"] = final.get("strategy_label")
        task_profile = final.get("task_profile", {})
        for key, value in task_profile.items():
            if isinstance(value, (dict, list)):
                continue
            row[f"task_profile__{key}"] = value
    if threshold.exists():
        row.update({f"threshold__{k}": v for k, v in pd.read_csv(threshold).iloc[0].to_dict().items()})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "runs"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "summaries"))
    args = parser.parse_args()

    run_dirs = sorted(path for path in Path(args.runs_root).iterdir() if path.is_dir() and (path / "config.yaml").exists())
    rows = [flatten_profile(run_dir) for run_dir in run_dirs if (run_dir / "metrics" / "final_profile.json").exists()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No completed study runs found")
        return

    by_run = pd.DataFrame(rows).sort_values(["task", "architecture", "seed"])
    by_run.to_csv(out_dir / "by_run.csv", index=False)
    by_run.to_csv(out_dir / "by_task_arch_seed.csv", index=False)

    numeric_cols = by_run.select_dtypes(include="number").columns.tolist()
    mean_df = by_run.groupby(["task", "architecture"], dropna=False)[numeric_cols].mean().reset_index()
    if "strategy_label" in by_run.columns:
        label_df = (
            by_run.groupby(["task", "architecture"])["strategy_label"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
            .reset_index()
        )
        mean_df = mean_df.merge(label_df, on=["task", "architecture"], how="left")
    mean_df.to_csv(out_dir / "by_task_arch_mean.csv", index=False)
    print(f"Saved summaries to {out_dir}")


if __name__ == "__main__":
    main()
