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

from src.analysis.mechanism_classification import classify_mechanism
from src.analysis.rollout import load_run_artifacts
from src.utils.io import save_json
from src.utils.plotting import (
    plot_bar,
    plot_three_histograms,
    plot_three_traces,
    plot_trial_histogram,
    plot_proxy_traces,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg, task, model, _ = load_run_artifacts(
        args.run_dir, checkpoint_name=args.checkpoint, map_location="cpu"
    )
    analysis_cfg = cfg.get("analysis", {})

    run_dir = Path(args.run_dir)
    metrics_dir = run_dir / "metrics"
    raw_dir = metrics_dir / "raw"
    figures_dir = run_dir / "figures"
    for d in [metrics_dir, raw_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    experiment_name = run_dir.name
    task_name = cfg["task"]["name"]
    model_name = cfg["model"]["name"]
    seed = cfg.get("seed", "?")

    # ── Collect per-trial, per-timepoint data across batches ──────────
    summaries = []
    all_e_trace, all_d_trace, all_m_trace = [], [], []
    all_d_per_trial_max, all_d_per_trial_mean, all_d_per_trial_auc = [], [], []
    all_m_per_trial_max, all_m_per_trial_mean, all_m_per_trial_auc = [], [], []
    all_e_per_trial_max, all_e_per_trial_mean, all_e_per_trial_auc = [], [], []
    trial_rows = []

    for batch_idx in range(args.num_batches):
        batch = task.sample_batch(args.batch_size, split="val", device="cpu")
        result = classify_mechanism(model, task, batch, thresholds=analysis_cfg)
        summaries.append(result["summary"])

        e, d, m = result["effect"], result["direct"], result["modulation"]

        all_e_trace.append(e["trace"])
        all_d_trace.append(d["trace"])
        all_m_trace.append(m["trace"])

        n_trials = e["trace"].shape[1]
        for t in range(n_trials):
            trial_rows.append({
                "batch_idx": batch_idx,
                "trial_idx": t,
                "E_max": float(e["E_per_trial_max"][t]),
                "E_mean": float(e["E_per_trial_mean"][t]),
                "E_auc": float(e["E_per_trial_auc"][t]),
                "D_max": float(d["D_per_trial_max"][t]),
                "D_mean": float(d["D_per_trial_mean"][t]),
                "D_auc": float(d["D_per_trial_auc"][t]),
                "M_max": float(m["M_per_trial_max"][t]),
                "M_mean": float(m["M_per_trial_mean"][t]),
                "M_auc": float(m["M_per_trial_auc"][t]),
                "divergence_time": e["per_trial_divergence_time"][t],
            })

        all_d_per_trial_max.append(d["D_per_trial_max"])
        all_d_per_trial_mean.append(d["D_per_trial_mean"])
        all_d_per_trial_auc.append(d["D_per_trial_auc"])
        all_m_per_trial_max.append(m["M_per_trial_max"])
        all_m_per_trial_mean.append(m["M_per_trial_mean"])
        all_m_per_trial_auc.append(m["M_per_trial_auc"])
        all_e_per_trial_max.append(e["E_per_trial_max"])
        all_e_per_trial_mean.append(e["E_per_trial_mean"])
        all_e_per_trial_auc.append(e["E_per_trial_auc"])

    # ── Save per-trial CSV ────────────────────────────────────────────
    trial_df = pd.DataFrame(trial_rows)
    trial_df.insert(0, "experiment", experiment_name)
    trial_df.insert(1, "task", task_name)
    trial_df.insert(2, "model", model_name)
    trial_df.insert(3, "seed", seed)
    trial_df.to_csv(raw_dir / "per_trial_metrics.csv", index=False)

    # ── Save per-timepoint traces as .npz ─────────────────────────────
    e_trace_all = np.concatenate(all_e_trace, axis=1)
    d_trace_all = np.concatenate(all_d_trace, axis=1)
    m_trace_all = np.concatenate(all_m_trace, axis=1)
    np.savez_compressed(
        raw_dir / "traces.npz",
        E_trace=e_trace_all,
        D_trace=d_trace_all,
        M_trace=m_trace_all,
    )

    # ── Concatenate per-trial arrays ──────────────────────────────────
    d_max = np.concatenate(all_d_per_trial_max)
    d_mean = np.concatenate(all_d_per_trial_mean)
    m_max = np.concatenate(all_m_per_trial_max)
    m_mean = np.concatenate(all_m_per_trial_mean)
    m_auc = np.concatenate(all_m_per_trial_auc)
    e_max = np.concatenate(all_e_per_trial_max)

    # ── Batch-level summary (backward compatible) ─────────────────────
    df = pd.DataFrame(summaries)
    agg = {
        "experiment": experiment_name,
        "task": task_name,
        "model": model_name,
        "seed": seed,
        "D_J_mean": float(df["D_J"].mean()),
        "M_J_mean": float(df["M_J"].mean()),
        "E_J_mean": float(df["E_J"].mean()),
        "mechanism_class_mode": str(df["mechanism_class"].mode().iloc[0]),
        "mechanism_class_counts": df["mechanism_class"].value_counts().to_dict(),
        "max_abs_delta_q_mean": float(df["max_abs_delta_q"].mean()),
        "max_abs_direct_mean": float(df["max_abs_direct"].mean()),
        "max_abs_modulation_mean": float(df["max_abs_modulation"].mean()),
        "auc_delta_q_mean": float(df["auc_delta_q"].mean()),
        "auc_direct_mean": float(df["auc_direct"].mean()),
        "auc_modulation_mean": float(df["auc_modulation"].mean()),
        "n_trials_total": len(trial_df),
        "trial_D_max_median": float(np.median(d_max)),
        "trial_D_max_p90": float(np.percentile(d_max, 90)),
        "trial_M_max_median": float(np.median(m_max)),
        "trial_M_max_p90": float(np.percentile(m_max, 90)),
        "trial_M_mean_median": float(np.median(m_mean)),
        "trial_M_mean_p90": float(np.percentile(m_mean, 90)),
        "trial_E_max_median": float(np.median(e_max)),
        "trial_E_max_p90": float(np.percentile(e_max, 90)),
        "frac_trials_D_above": float((d_max > analysis_cfg.get("direct_threshold", 0.08)).mean()),
        "frac_trials_M_above": float((m_max > analysis_cfg.get("modulation_threshold", 0.05)).mean()),
        "frac_trials_E_above": float((e_max > analysis_cfg.get("effect_threshold", 0.15)).mean()),
    }
    save_json(agg, metrics_dir / "mechanism_summary.json")

    # ── Print compact summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {experiment_name}  ({task_name} / {model_name})")
    print(f"{'='*60}")
    print(f"  Class: {agg['mechanism_class_mode']}  ({agg['mechanism_class_counts']})")
    print(f"  D_J={agg['D_J_mean']:.2f}  M_J={agg['M_J_mean']:.2f}  E_J={agg['E_J_mean']:.2f}")
    print(f"  --- Batch-level (max over trials) ---")
    print(f"  max|E|={agg['max_abs_delta_q_mean']:.4f}  max|D|={agg['max_abs_direct_mean']:.4f}  max|M|={agg['max_abs_modulation_mean']:.4f}")
    print(f"  AUC(E)={agg['auc_delta_q_mean']:.4f}  AUC(D)={agg['auc_direct_mean']:.4f}  AUC(M)={agg['auc_modulation_mean']:.4f}")
    print(f"  --- Trial-level distribution (n={agg['n_trials_total']}) ---")
    print(f"  D_max: median={agg['trial_D_max_median']:.4f}  p90={agg['trial_D_max_p90']:.4f}  frac_above={agg['frac_trials_D_above']:.2f}")
    print(f"  M_max: median={agg['trial_M_max_median']:.4f}  p90={agg['trial_M_max_p90']:.4f}  frac_above={agg['frac_trials_M_above']:.2f}")
    print(f"  M_mean: median={agg['trial_M_mean_median']:.4f}  p90={agg['trial_M_mean_p90']:.4f}")
    print(f"  E_max: median={agg['trial_E_max_median']:.4f}  p90={agg['trial_E_max_p90']:.4f}  frac_above={agg['frac_trials_E_above']:.2f}")
    print()

    # ── Thresholds for plot annotations ───────────────────────────────
    d_thresh = analysis_cfg.get("direct_threshold", 0.06)
    m_thresh = analysis_cfg.get("modulation_threshold", 0.04)
    e_thresh = analysis_cfg.get("effect_threshold", 0.12)

    # ── Histograms ────────────────────────────────────────────────────
    plot_three_histograms(
        d_max, m_max, m_mean,
        figures_dir / "hist_D_M_trial.png",
        title=f"{experiment_name}: trial-level D & M",
        d_thresh=d_thresh, m_thresh=m_thresh,
    )
    plot_trial_histogram(
        d_max, figures_dir / "hist_trial_max_D.png",
        title=f"{experiment_name}: trial-max |D|",
        xlabel="trial-max |D|", threshold=d_thresh,
    )
    plot_trial_histogram(
        m_max, figures_dir / "hist_trial_max_M.png",
        title=f"{experiment_name}: trial-max |M|",
        xlabel="trial-max |M|", threshold=m_thresh,
    )
    plot_trial_histogram(
        m_mean, figures_dir / "hist_trial_mean_M.png",
        title=f"{experiment_name}: trial-mean |M|",
        xlabel="trial-mean |M|", threshold=m_thresh,
    )
    plot_trial_histogram(
        e_max, figures_dir / "hist_trial_max_E.png",
        title=f"{experiment_name}: trial-max |E|",
        xlabel="trial-max |delta_q|", threshold=e_thresh,
    )
    plot_trial_histogram(
        m_auc, figures_dir / "hist_trial_auc_M.png",
        title=f"{experiment_name}: trial AUC |M|",
        xlabel="time-integrated |M|",
    )

    # ── Time traces ───────────────────────────────────────────────────
    plot_three_traces(
        e_trace_all, d_trace_all, m_trace_all,
        figures_dir / "traces_E_D_M.png",
        title=f"{experiment_name}: proxy time traces",
        thresholds={"effect_threshold": e_thresh, "direct_threshold": d_thresh,
                     "modulation_threshold": m_thresh},
    )
    plot_proxy_traces(
        d_trace_all, figures_dir / "traces_D.png",
        title=f"{experiment_name}: |D(t)| per trial",
        ylabel="|D(t)|", threshold=d_thresh,
    )
    plot_proxy_traces(
        m_trace_all, figures_dir / "traces_M.png",
        title=f"{experiment_name}: |M(t)| per trial",
        ylabel="|M(t)|", threshold=m_thresh,
    )
    plot_proxy_traces(
        e_trace_all, figures_dir / "traces_E.png",
        title=f"{experiment_name}: |delta_q(t)| per trial",
        ylabel="|delta_q(t)|", threshold=e_thresh,
    )

    # ── Legacy bar chart ──────────────────────────────────────────────
    plot_bar(
        agg["mechanism_class_counts"],
        figures_dir / "mechanism_class_counts.png",
        title="Mechanism class counts", ylabel="count",
    )

    print(f"Saved raw metrics to {raw_dir}/")
    print(f"Saved figures to {figures_dir}/")


if __name__ == "__main__":
    main()
