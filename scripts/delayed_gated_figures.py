#!/usr/bin/env python
"""Canonical delayed_gated figures."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/experiments/delayed_gated")
FIGS = OUT / "figures"
DATA = np.load(OUT / "delayed_gated_minimal.npz")

q_raw = DATA["q"]
d = DATA["d"]
m = DATA["m"]
r = DATA["r"]
time = DATA["time"]
abs_q = np.abs(q_raw)
final_abs_q = np.abs(DATA["final_q"])
difficulty = DATA["difficulty"]

fix_mask = DATA["fix_mask"]
stim_mask = DATA["stim_mask"]
delay_mask = DATA["delay_mask"]
resp_mask = DATA["response_mask"]

N, T = q_raw.shape
fix_idx = np.where(fix_mask > 0)[0]
stim_idx = np.where(stim_mask > 0)[0]
delay_idx = np.where(delay_mask > 0)[0]
resp_idx = np.where(resp_mask > 0)[0]

median_faq = np.median(final_abs_q)
hi = final_abs_q >= median_faq
lo = ~hi

EPOCH_COLORS = {"fix": "#bdbdbd", "stim": "#aec7e8", "delay": "#ffbb78", "resp": "#98df8a"}
EPOCH_SPANS = [(fix_idx[0], fix_idx[-1]+1, "fix"),
               (stim_idx[0], stim_idx[-1]+1, "stim"),
               (delay_idx[0], delay_idx[-1]+1, "delay"),
               (resp_idx[0], resp_idx[-1]+1, "resp")]
EPOCH_LABELS = {"fix": "fixation", "stim": "stimulus", "delay": "delay", "resp": "response"}


def shade_epochs(ax, label_y=None):
    for s, e, name in EPOCH_SPANS:
        ax.axvspan(s, e, color=EPOCH_COLORS[name], alpha=0.12, zorder=0)
    if label_y is not None:
        for s, e, name in EPOCH_SPANS:
            ax.text((s+e)/2, label_y, EPOCH_LABELS[name], ha="center",
                    fontsize=7, color="#555555", style="italic")


def band(arr, axis=0):
    return np.mean(arr, axis=axis), np.percentile(arr, 25, axis=axis), np.percentile(arr, 75, axis=axis)


# =====================================================================
# Figure 1: Canonical delayed_gated 4-panel
# =====================================================================

fig, axes = plt.subplots(4, 1, figsize=(8, 9.5), sharex=True)

panels = [
    (abs_q, "|q(t)|", 0.06),
    (d,     "|D(t)|", 0.06),
    (m,     "|M(t)|", 0.06),
    (r,     "|Δq(t)|", 0.06),
]

for idx, (ax, (arr, ylabel, _)) in enumerate(zip(axes, panels)):
    shade_epochs(ax, label_y=ax.get_ylim()[1] if idx == 0 else None)

    # All trials (light gray)
    mn_all, p25_all, p75_all = band(arr)
    ax.fill_between(time, p25_all, p75_all, color="#cccccc", alpha=0.4, zorder=1)
    ax.plot(time, mn_all, color="#999999", linewidth=1, linestyle="--", label="all", zorder=2)

    # High / low split
    for mask, color, label in [(hi, "#d62728", "high |q$_f$|"), (lo, "#1f77b4", "low |q$_f$|")]:
        mn, p25, p75 = band(arr[mask])
        ax.fill_between(time, p25, p75, color=color, alpha=0.12, zorder=3)
        ax.plot(time, mn, color=color, linewidth=1.8, label=label, zorder=4)

    ax.set_ylabel(ylabel, fontsize=11)
    if idx == 0:
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        shade_epochs(ax, label_y=ax.get_ylim()[1] * 0.92)

axes[-1].set_xlabel("timestep", fontsize=11)
fig.suptitle("delayed_gated: mechanism proxy traces (high vs low final |q|)",
             fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig(FIGS / "canonical_delayed_gated.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved canonical_delayed_gated.png")


# =====================================================================
# Figure 2: Within-coherence M traces
# =====================================================================

coh_values = np.sort(np.unique(difficulty))
n_coh = len(coh_values)

fig, axes = plt.subplots(n_coh, 2, figsize=(12, 2.8 * n_coh), sharex=True)

for ci, cval in enumerate(coh_values):
    cmask = difficulty == cval
    faq_c = final_abs_q[cmask]
    med_c = np.median(faq_c)
    hi_c = cmask & (final_abs_q >= med_c)
    lo_c = cmask & (final_abs_q < med_c)
    if lo_c.sum() == 0:
        lo_c = cmask & (final_abs_q <= med_c)

    for ji, (arr, ylabel, title_prefix) in enumerate([(m, "|M(t)|", "M"), (d, "|D(t)|", "D")]):
        ax = axes[ci, ji]
        shade_epochs(ax)
        mn_hi = arr[hi_c].mean(axis=0)
        mn_lo = arr[lo_c].mean(axis=0)
        p25_hi, p75_hi = np.percentile(arr[hi_c], 25, axis=0), np.percentile(arr[hi_c], 75, axis=0)
        p25_lo, p75_lo = np.percentile(arr[lo_c], 25, axis=0), np.percentile(arr[lo_c], 75, axis=0)

        ax.fill_between(time, p25_hi, p75_hi, color="#d62728", alpha=0.1)
        ax.fill_between(time, p25_lo, p75_lo, color="#1f77b4", alpha=0.1)
        ax.plot(time, mn_hi, color="#d62728", linewidth=1.6,
                label=f"hi |q$_f$| (n={hi_c.sum()})")
        ax.plot(time, mn_lo, color="#1f77b4", linewidth=1.6,
                label=f"lo |q$_f$| (n={lo_c.sum()})")

        if ci == 0:
            ax.set_title(f"{title_prefix} proxy", fontsize=11)
            ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel(f"coh={cval:.1f}", fontsize=10)

axes[-1, 0].set_xlabel("timestep", fontsize=10)
axes[-1, 1].set_xlabel("timestep", fontsize=10)
fig.suptitle("Within-coherence: high vs low final |q| — M and D traces",
             fontsize=12, y=1.005)
fig.tight_layout()
fig.savefig(FIGS / "canonical_within_coherence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved canonical_within_coherence.png")
