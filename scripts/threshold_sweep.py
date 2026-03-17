#!/usr/bin/env python
"""Sweep D/M thresholds for each run, generate heatmaps and a robustness summary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.analysis.mechanism_classification import (
    classify_mechanism,
    threshold_sweep,
)
from src.analysis.eval_cache import load_or_create_eval_batch
from src.analysis.rollout import load_run_artifacts
from src.utils.io import save_json
from src.utils.plotting import (
    plot_class_grid,
    plot_sweep_comparison,
    plot_sweep_heatmap,
)


def collect_trial_arrays(model, task, analysis_cfg, num_batches=8, batch_size=32, eval_cache=None, refresh_eval_cache=False):
    """Run model, collect per-trial max/mean arrays (no sweep yet)."""
    all_d_max, all_d_mean = [], []
    all_m_max, all_m_mean = [], []

    if eval_cache:
        batches = [
            load_or_create_eval_batch(
                task,
                n_trials=batch_size,
                device="cpu",
                split="val",
                cache_path=eval_cache,
                refresh=refresh_eval_cache,
            )
        ]
    else:
        batches = [task.sample_batch(batch_size, split="val", device="cpu") for _ in range(num_batches)]

    for batch in batches:
        result = classify_mechanism(model, task, batch, thresholds=analysis_cfg)
        d, m = result["direct"], result["modulation"]
        all_d_max.append(d["D_per_trial_max"])
        all_d_mean.append(d["D_per_trial_mean"])
        all_m_max.append(m["M_per_trial_max"])
        all_m_mean.append(m["M_per_trial_mean"])

    return {
        "d_max": np.concatenate(all_d_max),
        "d_mean": np.concatenate(all_d_mean),
        "m_max": np.concatenate(all_m_max),
        "m_mean": np.concatenate(all_m_mean),
    }


def robustness_label(sweep_max: dict) -> str:
    """Assign a robustness label based on how often the class flips across the grid."""
    classes = sweep_max["class_grid"].ravel()
    total = len(classes)
    counts = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    dominant_class = max(counts, key=counts.get)
    dominant_frac = counts[dominant_class] / total
    if dominant_frac >= 0.80:
        return f"robust {dominant_class}"
    elif dominant_frac >= 0.50:
        return f"leaning {dominant_class}"
    else:
        return "threshold-sensitive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-min", type=float, default=0.01)
    parser.add_argument("--d-max", type=float, default=0.15)
    parser.add_argument("--d-steps", type=int, default=20)
    parser.add_argument("--m-min", type=float, default=0.005)
    parser.add_argument("--m-max", type=float, default=0.12)
    parser.add_argument("--m-steps", type=int, default=20)
    parser.add_argument("--mean-factor", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--eval-cache", type=str, default=None)
    parser.add_argument("--refresh-eval-cache", action="store_true")
    args = parser.parse_args()

    d_range = np.linspace(args.d_min, args.d_max, args.d_steps)
    m_range = np.linspace(args.m_min, args.m_max, args.m_steps)

    summary_rows = []

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        exp_name = run_dir.name
        print(f"\n{'='*60}")
        print(f"  Sweeping: {exp_name}")
        print(f"{'='*60}")

        cfg, task, model, _ = load_run_artifacts(
            run_dir, checkpoint_name=args.checkpoint, map_location="cpu"
        )
        analysis_cfg = cfg.get("analysis", {})
        current_d = analysis_cfg.get("direct_threshold", 0.06)
        current_m = analysis_cfg.get("modulation_threshold", 0.04)

        arrays = collect_trial_arrays(
            model, task, analysis_cfg,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            eval_cache=args.eval_cache,
            refresh_eval_cache=args.refresh_eval_cache,
        )

        sweep_max = threshold_sweep(
            arrays["d_max"], arrays["d_mean"],
            arrays["m_max"], arrays["m_mean"],
            d_range, m_range, rule="max",
        )
        sweep_maxmean = threshold_sweep(
            arrays["d_max"], arrays["d_mean"],
            arrays["m_max"], arrays["m_mean"],
            d_range, m_range, rule="max_mean", mean_factor=args.mean_factor,
        )

        fig_dir = run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_class_grid(
            sweep_max["class_grid"], d_range, m_range,
            fig_dir / "sweep_class_max.png",
            title=f"{exp_name}: class label (max rule)",
            current_d=current_d, current_m=current_m,
        )
        plot_class_grid(
            sweep_maxmean["class_grid"], d_range, m_range,
            fig_dir / "sweep_class_maxmean.png",
            title=f"{exp_name}: class label (max+mean rule)",
            current_d=current_d, current_m=current_m,
        )
        plot_sweep_comparison(
            sweep_max["class_grid"], sweep_maxmean["class_grid"],
            d_range, m_range,
            fig_dir / "sweep_comparison.png",
            title=f"{exp_name}: max vs max+mean",
            current_d=current_d, current_m=current_m,
        )
        plot_sweep_heatmap(
            sweep_max["frac_D_trials"], d_range, m_range,
            fig_dir / "sweep_frac_D_trials.png",
            title=f"{exp_name}: frac trials D>thresh (max rule)",
            cmap="YlOrRd", current_d=current_d, current_m=current_m,
        )
        plot_sweep_heatmap(
            sweep_max["frac_M_trials"], d_range, m_range,
            fig_dir / "sweep_frac_M_trials.png",
            title=f"{exp_name}: frac trials M>thresh (max rule)",
            cmap="YlGnBu", current_d=current_d, current_m=current_m,
        )

        sweep_dir = run_dir / "metrics" / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sweep_dir / "sweep_max.npz",
            d_range=d_range, m_range=m_range,
            class_grid=sweep_max["class_grid"],
            frac_mixed=sweep_max["frac_mixed"],
            frac_direct_only=sweep_max["frac_direct_only"],
            frac_D_trials=sweep_max["frac_D_trials"],
            frac_M_trials=sweep_max["frac_M_trials"],
        )
        np.savez_compressed(
            sweep_dir / "sweep_maxmean.npz",
            d_range=d_range, m_range=m_range,
            class_grid=sweep_maxmean["class_grid"],
            frac_mixed=sweep_maxmean["frac_mixed"],
            frac_direct_only=sweep_maxmean["frac_direct_only"],
            frac_D_trials=sweep_maxmean["frac_D_trials"],
            frac_M_trials=sweep_maxmean["frac_M_trials"],
        )

        label_max = robustness_label(sweep_max)
        label_maxmean = robustness_label(sweep_maxmean)

        classes_max = sweep_max["class_grid"].ravel()
        n = len(classes_max)
        pct_mixed_max = float((classes_max == "(1,1)").sum()) / n * 100
        pct_direct_max = float((classes_max == "(1,0)").sum()) / n * 100
        pct_mod_max = float((classes_max == "(0,1)").sum()) / n * 100
        pct_none_max = float((classes_max == "(0,0)").sum()) / n * 100

        classes_mm = sweep_maxmean["class_grid"].ravel()
        pct_mixed_mm = float((classes_mm == "(1,1)").sum()) / n * 100
        pct_direct_mm = float((classes_mm == "(1,0)").sum()) / n * 100

        print(f"  Max rule:      {label_max}")
        print(f"    (1,1)={pct_mixed_max:.0f}%  (1,0)={pct_direct_max:.0f}%  "
              f"(0,1)={pct_mod_max:.0f}%  (0,0)={pct_none_max:.0f}%")
        print(f"  Max+mean rule: {label_maxmean}")
        print(f"    (1,1)={pct_mixed_mm:.0f}%  (1,0)={pct_direct_mm:.0f}%")

        row = {
            "experiment": exp_name,
            "task": cfg["task"]["name"],
            "model": cfg["model"]["name"],
            "robustness_max": label_max,
            "robustness_maxmean": label_maxmean,
            "pct_mixed_max": round(pct_mixed_max, 1),
            "pct_direct_only_max": round(pct_direct_max, 1),
            "pct_mod_only_max": round(pct_mod_max, 1),
            "pct_none_max": round(pct_none_max, 1),
            "pct_mixed_maxmean": round(pct_mixed_mm, 1),
            "pct_direct_only_maxmean": round(pct_direct_mm, 1),
            "current_d_thresh": current_d,
            "current_m_thresh": current_m,
        }
        summary_rows.append(row)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dirs[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "threshold_robustness_summary.csv", index=False)

    print(f"\n{'='*60}")
    print("  ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(summary_df[["experiment", "robustness_max", "robustness_maxmean",
                       "pct_mixed_max", "pct_direct_only_max",
                       "pct_none_max"]].to_string(index=False))
    print(f"\nSaved to {out_dir / 'threshold_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
