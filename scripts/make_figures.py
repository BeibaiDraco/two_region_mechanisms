#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis.summary_stats import class_counts, collect_mechanism_summaries
from src.utils.plotting import plot_bar, plot_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    df = collect_mechanism_summaries(root)
    if df.empty:
        raise SystemExit(f"No mechanism_summary.json files found under {root}")

    counts = class_counts(df)
    plot_bar(counts, root / "mechanism_class_counts_overall.png", title="Overall mechanism class counts", ylabel="count")

    if "run_dir" in df.columns:
        df["run_name"] = df["run_dir"].apply(lambda s: Path(s).name)
    df.to_csv(root / "mechanism_summary_table.csv", index=False)

    heatmap_cols = [c for c in ["D_J_mean", "M_J_mean", "E_J_mean", "max_abs_delta_q_mean", "max_abs_direct_mean", "max_abs_modulation_mean"] if c in df.columns]
    if heatmap_cols and "run_name" in df.columns:
        matrix = df[heatmap_cols].to_numpy(dtype=float)
        plot_heatmap(matrix, df["run_name"].tolist(), heatmap_cols, root / "summary_heatmap.png", title="Run summary heatmap")

    print(df.head())


if __name__ == "__main__":
    main()
