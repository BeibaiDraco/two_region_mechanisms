#!/usr/bin/env python
"""Generate all missing slide-ready figures for the 5 experiments."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPS = Path("outputs/experiments")

def _bar_colors(vals):
    return ["#d62728" if v > 0 else "#1f77b4" for v in vals]

# =====================================================================
# 1. BINARY — D vs M unique contribution
# =====================================================================
def binary_unique():
    p = json.load(open(EXPS / "binary_additive/metrics/binary_additive_final_profile.json"))
    d_u = p["stim_D_unique_of_M"]
    m_u = p["stim_M_unique_of_D"]
    d_on = p["D_onset"]
    m_on = p["M_onset"]
    q_on = p["q_onset"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # Bar plot
    ax = axes[0]
    labels = ["D unique\n(ctrl M+diff)", "M unique\n(ctrl D+diff)"]
    vals = [d_u, m_u]
    bars = ax.bar(labels, vals, color=_bar_colors(vals), width=0.5, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("partial correlation with response |q|", fontsize=9)
    ax.set_title("Unique contribution\n(stimulus -> response)", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03 * np.sign(v),
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.8, 1.0)

    # Onset timeline
    ax = axes[1]
    events = [("D onset", d_on, "#d62728"), ("M onset", m_on, "#1f77b4"), ("q onset", q_on, "#333333")]
    for i, (name, t, c) in enumerate(events):
        ax.barh(i, t, height=0.5, color=c, alpha=0.8)
        ax.text(t + 0.5, i, f"t={t:.0f}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([e[0] for e in events], fontsize=10)
    ax.set_xlabel("timestep", fontsize=10)
    ax.set_title("Onset ordering", fontsize=10)
    ax.invert_yaxis()

    fig.suptitle("Binary additive: direct-dominant", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "binary_additive/figures/slide_unique_contribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  binary: slide_unique_contribution.png")


# =====================================================================
# 2. DELAYED — timing + unique contribution summary
# =====================================================================
def delayed_unique():
    p = json.load(open(EXPS / "delayed_gated/metrics/delayed_gated_final_profile.json"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # Onset timeline
    ax = axes[0]
    events = [("M onset", p["M_onset"], "#1f77b4"),
              ("D onset", p["D_onset"], "#d62728"),
              ("q onset", p["q_onset"], "#333333")]
    for i, (name, t, c) in enumerate(events):
        ax.barh(i, t, height=0.5, color=c, alpha=0.8)
        ax.text(t + 0.5, i, f"t={t:.0f}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([e[0] for e in events], fontsize=10)
    ax.set_xlabel("timestep", fontsize=10)
    ax.set_title("Onset ordering", fontsize=10)
    ax.invert_yaxis()

    # Stim->response unique
    ax = axes[1]
    labels = ["D unique", "M unique"]
    vals = [p["stim_D_unique"], p["stim_M_unique"]]
    bars = ax.bar(labels, vals, color=_bar_colors(vals), width=0.5, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("stimulus -> response", fontsize=10)
    ax.set_ylabel("partial corr", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03 * np.sign(v),
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.8, 1.0)

    # Delay->response unique
    ax = axes[2]
    vals2 = [p["delay_D_unique"], p["delay_M_unique"]]
    bars = ax.bar(labels, vals2, color=_bar_colors(vals2), width=0.5, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("delay -> response (strict)", fontsize=10)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03 * np.sign(v),
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.8, 1.0)

    fig.suptitle("Delayed gated: M-first, delay-supported", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "delayed_gated/figures/slide_unique_contribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  delayed: slide_unique_contribution.png")


# =====================================================================
# 3. CONTEXT — routing figures
# =====================================================================
def context_routing():
    R = np.load(EXPS / "context_gated/context_gated_routing.npz")
    rel_c = R["rel_slope_coupled"]
    irrel_c = R["irrel_slope_coupled"]
    rel_d = R["rel_slope_decoupled"]
    irrel_d = R["irrel_slope_decoupled"]
    time = R["time"]
    T = len(time)
    eps = 1e-6
    norm_sel = (rel_c - np.abs(irrel_c)) / (np.abs(rel_c) + np.abs(irrel_c) + eps)

    epochs = {"cue": (0, 8), "stimulus": (8, 32), "delay": (32, 42), "response": (42, 58)}
    EC = {"cue": "#c7b8ea", "stimulus": "#aec7e8", "delay": "#ffbb78", "response": "#98df8a"}

    def shade(ax):
        for n, (s, e) in epochs.items():
            ax.axvspan(s, e, alpha=0.12, color=EC[n])

    # Figure B: time-resolved routing slopes
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax = axes[0]
    shade(ax)
    ax.plot(time, rel_c, color="#d62728", lw=2, label="relevant slope")
    ax.plot(time, irrel_c, color="#1f77b4", lw=2, label="irrelevant slope")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel("slope (coupled)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    shade(ax)
    ax.plot(time, norm_sel, color="#2ca02c", lw=2)
    ax.set_ylabel("normalized selectivity", fontsize=10)
    ax.set_xlabel("timestep", fontsize=10)
    ax.set_ylim(-0.2, 1.1)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    for ax in axes:
        for n, (s, e) in epochs.items():
            ax.text((s+e)/2, ax.get_ylim()[1]*0.93, n, ha="center", fontsize=7,
                    color="#555", style="italic")

    fig.suptitle("Context gated: time-resolved routing", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "context_gated/figures/slide_routing_timecourse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  context: slide_routing_timecourse.png")

    # Figure C: epoch routing summary
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep_names = ["cue", "stimulus", "delay", "response"]
    x = np.arange(len(ep_names))
    w = 0.3
    rel_vals = [rel_c[s:e].mean() for n, (s, e) in epochs.items()]
    irrel_vals = [np.abs(irrel_c[s:e]).mean() for n, (s, e) in epochs.items()]
    ax.bar(x - w/2, rel_vals, w, color="#d62728", label="relevant slope", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, irrel_vals, w, color="#1f77b4", label="|irrelevant| slope", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_names, fontsize=10)
    ax.set_ylabel("slope magnitude", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title("Context gated: routing by epoch (coupled)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "context_gated/figures/slide_routing_epoch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  context: slide_routing_epoch.png")

    # Figure E: coupled vs decoupled routing
    p = json.load(open(EXPS / "context_gated/metrics/context_gated_final_profile.json"))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["coupled", "decoupled"]
    vals = [p["response_routing_selectivity"], p["decoupled_routing_selectivity"]]
    bars = ax.bar(labels, vals, color=["#d62728", "#1f77b4"], width=0.4, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, max(v, 0) + 0.05,
                f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("response routing selectivity", fontsize=10)
    ax.set_title("Context gated: A->B is necessary for routing", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.2, 2.5)
    fig.tight_layout()
    fig.savefig(EXPS / "context_gated/figures/slide_routing_coupled_decoupled.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  context: slide_routing_coupled_decoupled.png")


# =====================================================================
# 4. STATE-SETTING — compact mode-spread summary
# =====================================================================
def state_setting_summary():
    p = json.load(open(EXPS / "state_setting_reciprocal/metrics/state_setting_reciprocal_final_profile.json"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Alpha vs beta separation
    ax = axes[0]
    seps = p["mode_separation_by_epoch"]
    ep_names = list(seps.keys())
    x = np.arange(len(ep_names))
    w = 0.3
    alpha_vals = [seps[e]["alpha_sep"] for e in ep_names]
    beta_vals = [seps[e]["beta_sep"] for e in ep_names]
    ax.bar(x - w/2, alpha_vals, w, color="#2ca02c", label="alpha (readout)", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, beta_vals, w, color="#ff7f0e", label="beta (orthogonal)", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_names, fontsize=9)
    ax.set_ylabel("mode separation (std)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title("Hidden-state separation", fontsize=10)

    # Mode spread coupled vs decoupled
    ax = axes[1]
    labels = ["coupled", "decoupled"]
    vals = [p["response_mode_spread_coupled"], p["response_mode_spread_decoupled"]]
    bars = ax.bar(labels, vals, color=["#d62728", "#1f77b4"], width=0.4, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("response mode spread", fontsize=9)
    ax.set_title("Mode differentiation", fontsize=10)

    # Triplet spread
    ax = axes[2]
    vals2 = [p["triplet_q_spread_coupled"], p["triplet_q_spread_decoupled"]]
    bars = ax.bar(labels, vals2, color=["#d62728", "#1f77b4"], width=0.4, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vals2):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("within-triplet spread", fontsize=9)
    ax.set_title("Same input, different cue", fontsize=10)

    fig.suptitle("State-setting: off-axis mode control", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "state_setting_reciprocal/figures/slide_mode_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  state_setting: slide_mode_summary.png")


# =====================================================================
# 5. REDUNDANT — decoupling cost + unique contribution
# =====================================================================
def redundant_figures():
    p = json.load(open(EXPS / "redundant_lowrank/metrics/redundant_lowrank_final_profile.json"))
    cost = p["decoupling_cost_by_difficulty"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Decoupling cost by difficulty
    ax = axes[0]
    diffs = [r["difficulty"] for r in cost]
    coupled = [r["coupled_mean_abs_q"] for r in cost]
    decoupled = [r["decoupled_mean_abs_q"] for r in cost]
    x = np.arange(len(diffs))
    w = 0.3
    ax.bar(x - w/2, coupled, w, color="#d62728", label="coupled", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, decoupled, w, color="#1f77b4", label="decoupled", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.1f}" for d in diffs], fontsize=10)
    ax.set_xlabel("|coherence|", fontsize=10)
    ax.set_ylabel("response |q|", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_title("Decoupling cost by difficulty", fontsize=10)
    for i, r in enumerate(cost):
        ax.text(i, max(coupled[i], decoupled[i]) + 0.03,
                f"{r['relative_cost']:.0%}", ha="center", fontsize=8, color="#555")

    # Unique contribution
    ax = axes[1]
    labels = ["D unique\n(ctrl M+diff)", "M unique\n(ctrl D+diff)"]
    vals = [p["stim_D_unique_of_M"], p["stim_M_unique_of_D"]]
    bars = ax.bar(labels, vals, color=_bar_colors(vals), width=0.5, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("partial corr with response |q|", fontsize=9)
    ax.set_title("Unique contribution\n(stimulus -> response)", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03 * max(np.sign(v), 0.1),
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.2, 1.0)

    fig.suptitle("Redundant lowrank: direct backup", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(EXPS / "redundant_lowrank/figures/slide_cost_and_unique.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  redundant: slide_cost_and_unique.png")


# =====================================================================
if __name__ == "__main__":
    print("Generating slide figures...")
    binary_unique()
    delayed_unique()
    context_routing()
    state_setting_summary()
    redundant_figures()
    print("\nDone.")
