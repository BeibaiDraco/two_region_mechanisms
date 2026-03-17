#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tasks import build_task
from src.utils.config import load_config


def save_label_grid(df: pd.DataFrame, out_path: Path) -> None:
    label_df = (
        df.groupby(["task", "architecture"])["strategy_label"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")
        .reset_index()
    )
    pivot = label_df.pivot(index="task", columns="architecture", values="strategy_label").fillna("")
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(pivot.columns)), max(4, 0.8 * len(pivot.index))))
    ax.axis("off")
    table = ax.table(cellText=pivot.values, rowLabels=pivot.index, colLabels=pivot.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_mechanism_map(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for architecture, subdf in df.groupby("architecture"):
        ax.scatter(subdf["D_J_mean"], subdf["M_J_mean"], label=architecture, alpha=0.8)
    ax.set_xlabel("D_J_mean")
    ax.set_ylabel("M_J_mean")
    ax.legend()
    ax.set_title("D vs M mechanism map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _epoch_spans(run_dir: Path):
    cfg = load_config(run_dir / "config.yaml")
    task = build_task(cfg["task"])
    return task.epochs


def _mean_run_traces(run_dir: Path) -> dict[str, np.ndarray]:
    trace_path = run_dir / "metrics" / "raw" / "traces.npz"
    payload = np.load(trace_path)
    return {
        "E": payload["E_trace"].mean(axis=1),
        "D": payload["D_trace"].mean(axis=1),
        "M": payload["M_trace"].mean(axis=1),
    }


def _peak_normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values)))
    if peak <= 1e-12:
        return values.copy()
    return values / peak


def save_emd_summary_grid(
    df: pd.DataFrame,
    runs_root: Path,
    out_path: Path,
    normalize: bool = False,
) -> None:
    task_order = list(df["task"].dropna().unique())
    arch_order = list(df["architecture"].dropna().unique())
    fig, axes = plt.subplots(
        len(task_order),
        len(arch_order),
        figsize=(4.0 * len(arch_order), 2.8 * len(task_order)),
        squeeze=False,
        sharey=False,
    )
    colors = {"E": "#111111", "D": "#d62728", "M": "#1f77b4"}

    for row_idx, task_name in enumerate(task_order):
        for col_idx, arch_name in enumerate(arch_order):
            ax = axes[row_idx][col_idx]
            subdf = df[(df["task"] == task_name) & (df["architecture"] == arch_name)]
            if subdf.empty:
                ax.axis("off")
                continue

            run_traces = {"E": [], "D": [], "M": []}
            for run_name in subdf["run_name"]:
                traces = _mean_run_traces(runs_root / run_name)
                for key, values in traces.items():
                    run_traces[key].append(_peak_normalize(values) if normalize else values)

            first_run_dir = runs_root / subdf.iloc[0]["run_name"]
            for _, (start, end) in _epoch_spans(first_run_dir).items():
                ax.axvspan(start, end, color="#dddddd", alpha=0.12, linewidth=0)

            for key in ["E", "D", "M"]:
                values = np.stack(run_traces[key], axis=0)
                mean = values.mean(axis=0)
                std = values.std(axis=0)
                time = np.arange(mean.shape[0])
                ax.plot(time, mean, color=colors[key], lw=1.8, label=key)
                ax.fill_between(time, mean - std, mean + std, color=colors[key], alpha=0.12)

            if row_idx == 0:
                ax.set_title(arch_name)
            if col_idx == 0:
                ax.set_ylabel(task_name.replace("Task", ""))
            if row_idx == len(task_order) - 1:
                ax.set_xlabel("time")
            ax.set_xlim(left=0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    title = "Study E/D/M Summary Trajectories"
    if normalize:
        title += " (Peak-Normalized)"
    else:
        title += " (Raw)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_onsets(df: pd.DataFrame, out_path: Path) -> None:
    onset_cols = [col for col in ["task_profile__D_onset", "task_profile__M_onset", "task_profile__q_onset"] if col in df.columns]
    if not onset_cols:
        return
    melted = df.melt(id_vars=["task", "architecture"], value_vars=onset_cols, var_name="metric", value_name="onset")
    melted = melted.dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, subdf in melted.groupby("metric"):
        ax.scatter(subdf["task"] + " / " + subdf["architecture"], subdf["onset"], label=metric, alpha=0.8)
    ax.set_ylabel("Onset")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=str, default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "summaries" / "by_run.csv"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "figures"))
    parser.add_argument("--runs-root", type=str, default=str(ROOT / "outputs" / "studies" / "task_arch_grid" / "runs"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    out_dir = Path(args.output_dir)
    runs_root = Path(args.runs_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if {"run_name", "task", "architecture"}.issubset(df.columns):
        save_emd_summary_grid(df, runs_root, out_dir / "emd_summary_grid.png", normalize=False)
        save_emd_summary_grid(df, runs_root, out_dir / "emd_summary_grid_peak_normalized.png", normalize=True)
    if "strategy_label" in df.columns:
        save_label_grid(df, out_dir / "task_arch_profile_grid.png")
    if {"D_J_mean", "M_J_mean"}.issubset(df.columns):
        save_mechanism_map(df, out_dir / "mechanism_map.png")
    save_onsets(df, out_dir / "onset_plot.png")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
