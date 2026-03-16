from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(train_history: pd.DataFrame, val_history: pd.DataFrame, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    if not train_history.empty:
        plt.plot(train_history["step"], train_history["loss"], label="train loss")
        plt.plot(train_history["step"], train_history["accuracy"], label="train acc")
    if not val_history.empty:
        plt.plot(val_history["step"], val_history["loss"], label="val loss")
        plt.plot(val_history["step"], val_history["accuracy"], label="val acc")
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_readout_trajectories(
    q_c: np.ndarray,
    q_0: np.ndarray,
    targets: np.ndarray,
    save_path: str | Path,
    num_trials: int = 6,
    title: str = "Coupled vs decoupled readout",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_trials = min(num_trials, q_c.shape[1])
    plt.figure(figsize=(8, 5))
    for idx in range(num_trials):
        plt.plot(q_c[:, idx], alpha=0.9, label="coupled" if idx == 0 else None)
        plt.plot(q_0[:, idx], linestyle="--", alpha=0.9, label="decoupled" if idx == 0 else None)
        plt.plot(targets[:, idx], linestyle=":", alpha=0.6, label="target" if idx == 0 else None)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_hidden_projection(
    proj_a: np.ndarray,
    proj_b: np.ndarray,
    save_path: str | Path,
    title: str = "PCA trajectories",
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    if proj_a.shape[-1] >= 2:
        plt.plot(proj_a[:, 0], proj_a[:, 1], label="region A")
    if proj_b.shape[-1] >= 2:
        plt.plot(proj_b[:, 0], proj_b[:, 1], label="region B")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_bar(values: Dict[str, float], save_path: str | Path, title: str, xlabel: str = "", ylabel: str = "") -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(values.keys())
    nums = [values[k] for k in labels]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, nums)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(1.1 * len(col_labels) + 2, 0.8 * len(row_labels) + 2))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Distribution plots for mechanism proxy diagnostics
# ---------------------------------------------------------------------------

def plot_trial_histogram(
    values: np.ndarray,
    save_path: str | Path,
    title: str,
    xlabel: str,
    threshold: Optional[float] = None,
    bins: int = 30,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(values, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.75)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
        ax.legend()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("trial count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_proxy_traces(
    trace: np.ndarray,
    save_path: str | Path,
    title: str,
    ylabel: str,
    num_trials: int = 20,
    threshold: Optional[float] = None,
) -> None:
    """Plot |proxy(t)| for a sample of individual trials plus the trial-mean."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    T, B = trace.shape
    sample_idx = np.random.choice(B, size=min(num_trials, B), replace=False)
    sample_idx.sort()

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in sample_idx:
        ax.plot(trace[:, i], color="steelblue", alpha=0.25, linewidth=0.8)
    ax.plot(trace.mean(axis=1), color="black", linewidth=2, label="trial mean")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("timestep")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_three_histograms(
    d_max: np.ndarray,
    m_max: np.ndarray,
    m_mean: np.ndarray,
    save_path: str | Path,
    title: str,
    d_thresh: Optional[float] = None,
    m_thresh: Optional[float] = None,
    bins: int = 30,
) -> None:
    """Side-by-side histograms: trial-max D, trial-max M, trial-mean M."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    for ax, vals, label, thresh in [
        (axes[0], d_max, "trial-max |D|", d_thresh),
        (axes[1], m_max, "trial-max |M|", m_thresh),
        (axes[2], m_mean, "trial-mean |M|", m_thresh),
    ]:
        ax.hist(vals, bins=bins, edgecolor="black", linewidth=0.5, alpha=0.75)
        if thresh is not None:
            ax.axvline(thresh, color="red", linestyle="--", linewidth=1.2, label=f"threshold={thresh}")
            ax.legend(fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("trial count")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_three_traces(
    e_trace: np.ndarray,
    d_trace: np.ndarray,
    m_trace: np.ndarray,
    save_path: str | Path,
    title: str,
    num_trials: int = 15,
    thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Stacked time-trace panels for E, D, M."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or {}

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    labels = ["|delta_q(t)|", "|D(t)|", "|M(t)|"]
    traces = [e_trace, d_trace, m_trace]
    thresh_keys = ["effect_threshold", "direct_threshold", "modulation_threshold"]

    for ax, trace, ylabel, tk in zip(axes, traces, labels, thresh_keys):
        T, B = trace.shape
        sample_idx = np.random.choice(B, size=min(num_trials, B), replace=False)
        sample_idx.sort()
        for i in sample_idx:
            ax.plot(trace[:, i], color="steelblue", alpha=0.2, linewidth=0.7)
        ax.plot(trace.mean(axis=1), color="black", linewidth=1.8, label="trial mean")
        if tk in thresholds:
            ax.axhline(thresholds[tk], color="red", linestyle="--", linewidth=1,
                        label=f"threshold={thresholds[tk]}")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("timestep")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Threshold sweep heatmaps
# ---------------------------------------------------------------------------

def plot_sweep_heatmap(
    matrix: np.ndarray,
    d_range: np.ndarray,
    m_range: np.ndarray,
    save_path: str | Path,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    current_d: Optional[float] = None,
    current_m: Optional[float] = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        matrix, origin="lower", aspect="auto",
        extent=[m_range[0], m_range[-1], d_range[0], d_range[-1]],
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if current_d is not None and current_m is not None:
        ax.plot(current_m, current_d, marker="x", color="black",
                markersize=10, markeredgewidth=2.5)

    ax.set_xlabel("M threshold", fontsize=11)
    ax.set_ylabel("D threshold", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_class_grid(
    class_grid: np.ndarray,
    d_range: np.ndarray,
    m_range: np.ndarray,
    save_path: str | Path,
    title: str,
    current_d: Optional[float] = None,
    current_m: Optional[float] = None,
) -> None:
    """Color-coded heatmap of mechanism class labels."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    class_to_int = {"(0,0)": 0, "(1,0)": 1, "(0,1)": 2, "(1,1)": 3}
    int_grid = np.vectorize(lambda c: class_to_int.get(c, -1))(class_grid).astype(float)

    cmap = plt.cm.colors.ListedColormap(["#d9d9d9", "#fc8d62", "#66c2a5", "#8da0cb"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        int_grid, origin="lower", aspect="auto",
        extent=[m_range[0], m_range[-1], d_range[0], d_range[-1]],
        cmap=cmap, norm=norm,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["(0,0)", "(1,0)", "(0,1)", "(1,1)"])

    if current_d is not None and current_m is not None:
        ax.plot(current_m, current_d, marker="x", color="black",
                markersize=10, markeredgewidth=2.5)

    ax.set_xlabel("M threshold", fontsize=11)
    ax.set_ylabel("D threshold", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sweep_comparison(
    grid_max: np.ndarray,
    grid_maxmean: np.ndarray,
    d_range: np.ndarray,
    m_range: np.ndarray,
    save_path: str | Path,
    title: str,
    current_d: Optional[float] = None,
    current_m: Optional[float] = None,
) -> None:
    """Side-by-side class grids for max vs max+mean rules."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    class_to_int = {"(0,0)": 0, "(1,0)": 1, "(0,1)": 2, "(1,1)": 3}
    cmap = plt.cm.colors.ListedColormap(["#d9d9d9", "#fc8d62", "#66c2a5", "#8da0cb"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, grid, subtitle in [(axes[0], grid_max, "max rule"), (axes[1], grid_maxmean, "max+mean rule")]:
        int_grid = np.vectorize(lambda c: class_to_int.get(c, -1))(grid).astype(float)
        im = ax.imshow(
            int_grid, origin="lower", aspect="auto",
            extent=[m_range[0], m_range[-1], d_range[0], d_range[-1]],
            cmap=cmap, norm=norm,
        )
        if current_d is not None and current_m is not None:
            ax.plot(current_m, current_d, marker="x", color="black",
                    markersize=10, markeredgewidth=2.5)
        ax.set_xlabel("M threshold", fontsize=10)
        ax.set_ylabel("D threshold", fontsize=10)
        ax.set_title(subtitle, fontsize=11)

    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], fraction=0.023, pad=0.04)
    cbar.ax.set_yticklabels(["(0,0)", "(1,0)", "(0,1)", "(1,1)"])
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
