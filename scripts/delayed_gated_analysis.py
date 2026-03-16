#!/usr/bin/env python
"""Full delayed_gated mechanism analysis: Steps 1-5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("outputs/experiments/delayed_gated")
DATA = np.load(OUT / "delayed_gated_minimal.npz")

q_raw = DATA["q"]          # (N, T) — signed readout
d = DATA["d"]               # (N, T) — |D(t)|
m = DATA["m"]               # (N, T) — |M(t)|
r = DATA["r"]               # (N, T) — |delta_q(t)|
time = DATA["time"]          # (T,)
abs_q = np.abs(q_raw)

fix_mask = DATA["fix_mask"]
stim_mask = DATA["stim_mask"]
delay_mask = DATA["delay_mask"]
resp_mask = DATA["response_mask"]

final_q = DATA["final_q"]
correct = DATA["correct"]
difficulty = DATA["difficulty"]
target = DATA["target"]
coherence = DATA["coherence"]

N, T = q_raw.shape
final_abs_q = np.abs(final_q)

# Epoch index ranges
fix_idx = np.where(fix_mask > 0)[0]
stim_idx = np.where(stim_mask > 0)[0]
delay_idx = np.where(delay_mask > 0)[0]
resp_idx = np.where(resp_mask > 0)[0]

FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)


# =====================================================================
# Step 1: 4-panel trace plots
# =====================================================================
print("=== Step 1: Trace plots ===")

def trace_stats(arr):
    return np.mean(arr, axis=0), np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)

def shade_epochs(ax):
    alpha = 0.08
    ax.axvspan(fix_idx[0], fix_idx[-1]+1, color="gray", alpha=alpha)
    ax.axvspan(stim_idx[0], stim_idx[-1]+1, color="blue", alpha=alpha)
    ax.axvspan(delay_idx[0], delay_idx[-1]+1, color="orange", alpha=alpha)
    ax.axvspan(resp_idx[0], resp_idx[-1]+1, color="green", alpha=alpha)

def plot_four_panels(traces_dict, save_name, suptitle):
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    labels = ["|q(t)|", "|D(t)|", "|M(t)|", "|delta_q(t)|"]
    keys = ["abs_q", "d", "m", "r"]
    for ax, key, ylabel in zip(axes, keys, labels):
        shade_epochs(ax)
        for group_name, style in traces_dict.items():
            mn, p25, p75, color, ls = style[key]
            ax.plot(time, mn, color=color, linestyle=ls, linewidth=1.8, label=group_name)
            ax.fill_between(time, p25, p75, color=color, alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("timestep")
    axes[0].text(np.mean(fix_idx), axes[0].get_ylim()[1]*0.95, "fix", ha="center", fontsize=7, color="gray")
    axes[0].text(np.mean(stim_idx), axes[0].get_ylim()[1]*0.95, "stim", ha="center", fontsize=7, color="blue")
    axes[0].text(np.mean(delay_idx), axes[0].get_ylim()[1]*0.95, "delay", ha="center", fontsize=7, color="orange")
    axes[0].text(np.mean(resp_idx), axes[0].get_ylim()[1]*0.95, "resp", ha="center", fontsize=7, color="green")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / save_name, dpi=150)
    plt.close(fig)

# All trials
mn_aq, p25_aq, p75_aq = trace_stats(abs_q)
mn_d, p25_d, p75_d = trace_stats(d)
mn_m, p25_m, p75_m = trace_stats(m)
mn_r, p25_r, p75_r = trace_stats(r)

all_traces = {"all trials": {
    "abs_q": (mn_aq, p25_aq, p75_aq, "black", "-"),
    "d": (mn_d, p25_d, p75_d, "black", "-"),
    "m": (mn_m, p25_m, p75_m, "black", "-"),
    "r": (mn_r, p25_r, p75_r, "black", "-"),
}}
plot_four_panels(all_traces, "step1_traces_all.png", "delayed_gated: mean ± IQR")

# Median split on final |q|
median_faq = np.median(final_abs_q)
hi = final_abs_q >= median_faq
lo = ~hi

def group_stats(arr, mask):
    return trace_stats(arr[mask])

split_traces = {}
for name, mask, color in [("high |q_final|", hi, "crimson"), ("low |q_final|", lo, "steelblue")]:
    split_traces[name] = {
        "abs_q": (*group_stats(abs_q, mask), color, "-"),
        "d": (*group_stats(d, mask), color, "-"),
        "m": (*group_stats(m, mask), color, "-"),
        "r": (*group_stats(r, mask), color, "-"),
    }
plot_four_panels(split_traces, "step1_traces_median_split.png",
                 "delayed_gated: high vs low final |q| (median split)")

# Save epoch_traces.npz
save_dict = {"time": time, "fix_mask": fix_mask, "stim_mask": stim_mask,
             "delay_mask": delay_mask, "response_mask": resp_mask}
for prefix, arr in [("abs_q", abs_q), ("d", d), ("m", m), ("r", r)]:
    mn, p25, p75 = trace_stats(arr)
    save_dict[f"mean_{prefix}_all"] = mn
    save_dict[f"p25_{prefix}_all"] = p25
    save_dict[f"p75_{prefix}_all"] = p75
    for suffix, mask in [("hi", hi), ("lo", lo)]:
        mn_s, p25_s, p75_s = group_stats(arr, mask)
        save_dict[f"mean_{prefix}_{suffix}"] = mn_s
        save_dict[f"p25_{prefix}_{suffix}"] = p25_s
        save_dict[f"p75_{prefix}_{suffix}"] = p75_s
np.savez_compressed(OUT / "delayed_gated_epoch_traces.npz", **save_dict)
print(f"  Saved epoch_traces.npz  (hi={hi.sum()}, lo={lo.sum()})")


# =====================================================================
# Step 2: Within-coherence splits
# =====================================================================
print("\n=== Step 2: Within-coherence splits ===")

coh_values = np.sort(np.unique(difficulty))
by_coh = {}
by_coh_meta = {}

for ci, cval in enumerate(coh_values):
    cmask = difficulty == cval
    n_c = cmask.sum()
    faq_c = final_abs_q[cmask]
    med_c = np.median(faq_c)
    hi_c = cmask & (final_abs_q >= med_c)
    lo_c = cmask & (final_abs_q < med_c)
    if lo_c.sum() == 0:
        lo_c = cmask & (final_abs_q <= med_c)

    by_coh_meta[f"coh{ci}"] = {"coh_value": cval, "n_total": int(n_c),
                                 "n_hi": int(hi_c.sum()), "n_lo": int(lo_c.sum())}
    for prefix, arr in [("abs_q", abs_q), ("d", d), ("m", m)]:
        by_coh[f"mean_{prefix}_coh{ci}_hi"] = arr[hi_c].mean(axis=0) if hi_c.sum() > 0 else np.zeros(T)
        by_coh[f"mean_{prefix}_coh{ci}_lo"] = arr[lo_c].mean(axis=0) if lo_c.sum() > 0 else np.zeros(T)
    print(f"  coh={cval:.1f}: n={n_c}, hi={hi_c.sum()}, lo={lo_c.sum()}")

by_coh["coherence_values"] = coh_values
by_coh["time"] = time
by_coh["fix_mask"] = fix_mask
by_coh["stim_mask"] = stim_mask
by_coh["delay_mask"] = delay_mask
by_coh["response_mask"] = resp_mask
np.savez_compressed(OUT / "delayed_gated_by_coherence.npz", **by_coh)

# Plot within-coherence panels
fig, axes = plt.subplots(len(coh_values), 3, figsize=(14, 3 * len(coh_values)), sharex=True)
for ci, cval in enumerate(coh_values):
    for ji, (prefix, ylabel) in enumerate([("abs_q", "|q|"), ("d", "|D|"), ("m", "|M|")]):
        ax = axes[ci, ji]
        shade_epochs(ax)
        ax.plot(time, by_coh[f"mean_{prefix}_coh{ci}_hi"], color="crimson", label="hi |q_f|")
        ax.plot(time, by_coh[f"mean_{prefix}_coh{ci}_lo"], color="steelblue", label="lo |q_f|")
        if ci == 0:
            ax.set_title(ylabel)
        if ji == 0:
            ax.set_ylabel(f"coh={cval:.1f}")
        if ci == 0 and ji == 2:
            ax.legend(fontsize=7)
axes[-1, 1].set_xlabel("timestep")
fig.suptitle("Within-coherence: high vs low final |q|", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS / "step2_within_coherence.png", dpi=150)
plt.close(fig)
print("  Saved by_coherence.npz + step2 figure")


# =====================================================================
# Step 3 + 4: Per-trial epoch features + persistence
# =====================================================================
print("\n=== Step 3+4: Per-trial features ===")

def epoch_auc(arr_1d, idx):
    return float(np.trapz(arr_1d[idx]))

def epoch_peak(arr_1d, idx):
    return float(arr_1d[idx].max()) if len(idx) > 0 else 0.0

def onset_time(arr_1d, threshold_frac=0.1):
    """First time arr exceeds threshold_frac * max."""
    peak = arr_1d.max()
    if peak < 1e-8:
        return -1
    above = np.where(arr_1d > threshold_frac * peak)[0]
    return int(above[0]) if len(above) > 0 else -1

def peak_time(arr_1d):
    return int(np.argmax(arr_1d))

def duration_above_frac(arr_1d, frac, ref=None):
    """Number of timesteps where arr > frac * ref (ref defaults to trial peak)."""
    if ref is None:
        ref = arr_1d.max()
    if ref < 1e-8:
        return 0
    return int((arr_1d > frac * ref).sum())

def duration_above_fixed(arr_1d, threshold):
    return int((arr_1d > threshold).sum())

rows = []
for i in range(N):
    row = {
        "trial_id": i,
        "final_abs_q": final_abs_q[i],
        "coherence": coherence[i],
        "difficulty": difficulty[i],
        "target": target[i],
        "correct": correct[i],
    }
    for prefix, arr in [("d", d), ("m", m)]:
        tr = arr[i]
        row[f"stim_{prefix}_auc"] = epoch_auc(tr, stim_idx)
        row[f"delay_{prefix}_auc"] = epoch_auc(tr, delay_idx)
        row[f"resp_{prefix}_auc"] = epoch_auc(tr, resp_idx)
        row[f"stim_{prefix}_peak"] = epoch_peak(tr, stim_idx)
        row[f"delay_{prefix}_peak"] = epoch_peak(tr, delay_idx)
        row[f"resp_{prefix}_peak"] = epoch_peak(tr, resp_idx)
        row[f"{prefix}_onset"] = onset_time(tr)
        row[f"{prefix}_peak_time"] = peak_time(tr)
        row[f"{prefix}_peak_total"] = float(tr.max())
        row[f"{prefix}_auc_total"] = float(np.trapz(tr))
        row[f"{prefix}_duration_25pct"] = duration_above_frac(tr, 0.25)
        row[f"{prefix}_duration_50pct"] = duration_above_frac(tr, 0.50)
        row[f"{prefix}_duration_fixed_002"] = duration_above_fixed(tr, 0.02)

    # M-specific delay persistence
    row["m_auc_delay"] = epoch_auc(m[i], delay_idx)
    row["m_delay_duration_25pct"] = duration_above_frac(m[i, delay_idx[0]:delay_idx[-1]+1], 0.25,
                                                         ref=m[i].max())
    # q timing
    row["q_onset"] = onset_time(abs_q[i])
    row["q_peak_time"] = peak_time(abs_q[i])

    rows.append(row)

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT / "delayed_gated_trial_features.csv", index=False)
print(f"  Saved trial_features.csv ({len(feat_df)} rows, {len(feat_df.columns)} cols)")

# Correlations with final_abs_q
print("\n  --- Correlations with final_abs_q ---")
corr_features = [c for c in feat_df.columns if c not in
                 ["trial_id", "final_abs_q", "coherence", "difficulty", "target", "correct"]]
corr_rows = []
for feat in corr_features:
    vals = feat_df[feat].values
    valid = np.isfinite(vals) & (vals != -1)
    if valid.sum() < 10:
        continue
    r_val, p_val = stats.pearsonr(vals[valid], final_abs_q[valid])
    corr_rows.append({"feature": feat, "corr_with_final_abs_q": round(r_val, 4),
                       "pvalue": round(p_val, 6), "n": int(valid.sum())})
corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_final_abs_q", key=abs, ascending=False)
print(corr_df.head(15).to_string(index=False))

# Within-coherence partial correlations
print("\n  --- Within-coherence correlations (residualized) ---")
from sklearn.linear_model import LinearRegression
resid_faq = final_abs_q - LinearRegression().fit(difficulty.reshape(-1,1), final_abs_q).predict(difficulty.reshape(-1,1))
partial_rows = []
for feat in ["delay_m_auc", "delay_m_peak", "delay_d_auc", "delay_d_peak",
             "stim_m_auc", "stim_d_auc", "m_auc_total", "d_auc_total",
             "m_duration_50pct", "d_duration_50pct"]:
    if feat not in feat_df.columns:
        continue
    vals = feat_df[feat].values
    resid_feat = vals - LinearRegression().fit(difficulty.reshape(-1,1), vals).predict(difficulty.reshape(-1,1))
    r_val, p_val = stats.pearsonr(resid_feat, resid_faq)
    partial_rows.append({"feature": feat, "partial_corr": round(r_val, 4), "pvalue": round(p_val, 6)})
partial_df = pd.DataFrame(partial_rows).sort_values("partial_corr", key=abs, ascending=False)
print(partial_df.to_string(index=False))


# =====================================================================
# Step 5: Lag analysis
# =====================================================================
print("\n=== Step 5: Lag analysis ===")

lag_rows = []

def windowed_mean(arr_2d, idx):
    """Mean of arr_2d[:, idx] per trial."""
    return arr_2d[:, idx].mean(axis=1)

# Feature windows -> target windows
feature_windows = {
    "stim_d": ("d", stim_idx),
    "stim_m": ("m", stim_idx),
    "delay_d": ("d", delay_idx),
    "delay_m": ("m", delay_idx),
    "early_stim_d": ("d", stim_idx[:len(stim_idx)//2]),
    "early_stim_m": ("m", stim_idx[:len(stim_idx)//2]),
    "late_stim_d": ("d", stim_idx[len(stim_idx)//2:]),
    "late_stim_m": ("m", stim_idx[len(stim_idx)//2:]),
    "early_delay_m": ("m", delay_idx[:len(delay_idx)//2]),
    "late_delay_m": ("m", delay_idx[len(delay_idx)//2:]),
}

target_windows = {
    "resp_abs_q": ("abs_q", resp_idx),
    "delay_abs_q": ("abs_q", delay_idx),
    "late_delay_abs_q": ("abs_q", delay_idx[len(delay_idx)//2:]),
    "final_abs_q": None,
}

arrays = {"d": d, "m": m, "abs_q": abs_q}

for feat_name, (arr_key, feat_idx) in feature_windows.items():
    feat_vals = windowed_mean(arrays[arr_key], feat_idx)
    for tgt_name, tgt_spec in target_windows.items():
        if tgt_spec is None:
            tgt_vals = final_abs_q
        else:
            tgt_arr_key, tgt_idx = tgt_spec
            tgt_vals = windowed_mean(arrays[tgt_arr_key], tgt_idx)
        r_val, p_val = stats.pearsonr(feat_vals, tgt_vals)
        lag_rows.append({
            "feature": feat_name,
            "target_window": tgt_name,
            "corr": round(r_val, 4),
            "pvalue": round(p_val, 6),
            "n_trials": N,
        })

lag_df = pd.DataFrame(lag_rows)
lag_df.to_csv(OUT / "delayed_gated_lag_summary.csv", index=False)
print(f"  Saved lag_summary.csv ({len(lag_df)} rows)")
print("\n  --- Top lag correlations (predicting final_abs_q) ---")
sub = lag_df[lag_df["target_window"] == "final_abs_q"].sort_values("corr", key=abs, ascending=False)
print(sub.to_string(index=False))

print("\n  --- Top lag correlations (predicting resp_abs_q) ---")
sub2 = lag_df[lag_df["target_window"] == "resp_abs_q"].sort_values("corr", key=abs, ascending=False)
print(sub2.to_string(index=False))

print("\n=== All done ===")
