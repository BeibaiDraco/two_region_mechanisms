#!/usr/bin/env python
"""State-setting task: rule-aligned temporal analysis.

1. Export hidden states + W_out
2. Time-varying rule-aligned transfer curves (coupled vs decoupled)
3. Readout-aligned vs orthogonal hidden-state decomposition
4. Paired-triplet analysis (same input, different mode)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.readouts import default_readout
from src.analysis.rollout import load_run_artifacts

OUT = Path("outputs/experiments/state_setting_reciprocal")
FIGS = OUT / "figures"; FIGS.mkdir(parents=True, exist_ok=True)

MODE_NAMES = {0: "integrate", 1: "memory", 2: "transient"}
RULE_LABELS = {0: "cumsum(signal)", 1: "early_pulse", 2: "late_signal"}

cfg, task, model, _ = load_run_artifacts(str(OUT), checkpoint_name="best.pt", map_location="cpu")
epochs = task.epochs
sig_start, sig_end = epochs["signal"]
T = task.seq_len

# ── 1. Standard eval with hidden states ───────────────────────────────
print("=== Collecting standard batch (256 trials) ===")
batch = task.sample_batch(256, split="val", device="cpu")
model.eval()
with torch.no_grad():
    coupled = model(batch.inputs, intervention=None, add_noise=False)
    decoupled = model(batch.inputs, intervention={"type": "decouple"}, add_noise=False)

q_c = default_readout(coupled).cpu().numpy().T       # (N, T)
q_d = default_readout(decoupled).cpu().numpy().T
xb_c = coupled["x_b"].cpu().numpy().transpose(1, 0, 2)   # (N, T, H)
xb_d = decoupled["x_b"].cpu().numpy().transpose(1, 0, 2)
W_out = model.w_out.weight.detach().cpu().numpy()         # (1, H)
b_out = model.w_out.bias.detach().cpu().numpy()            # (1,)

meta = {k: v.cpu().numpy() for k, v in batch.meta.items()}
mode = meta["mode"]
label = meta["label"]
rule_val = meta["rule_relevant_value"]
sig_sum_total = meta["signal_sum_total"]
sig_sum_first = meta["signal_sum_first_half"]
sig_sum_second = meta["signal_sum_second_half"]
early_pulse = meta["early_pulse_value"]

N = q_c.shape[0]
sign = np.sign(label)

# Save hidden states
np.savez_compressed(OUT / "state_setting_hidden.npz",
    xb_coupled=xb_c.astype(np.float32),
    xb_decoupled=xb_d.astype(np.float32),
    q_coupled=q_c.astype(np.float32),
    q_decoupled=q_d.astype(np.float32),
    W_out=W_out, b_out=b_out,
    mode=mode, label=label,
    rule_relevant_value=rule_val,
    signal_sum_total=sig_sum_total,
    signal_sum_first_half=sig_sum_first,
    signal_sum_second_half=sig_sum_second,
    early_pulse_value=early_pulse,
    time=np.arange(T, dtype=np.float32))
print(f"  Saved state_setting_hidden.npz (xb shape={xb_c.shape})")

# ── 2. Time-varying rule-aligned transfer curves ──────────────────────
print("\n=== Time-varying rule-aligned transfer ===")

# For each mode, at each timestep, regress q(t) on the rule-relevant value
# Integrate: cumulative sum of signal up to min(t, sig_end)
# Memory: early pulse value (constant)
# Transient: second-half signal sum (constant)

input_signal = batch.inputs[:, :, 0].cpu().numpy().T  # (N, T)
cumsum_signal = np.cumsum(input_signal[:, sig_start:sig_end], axis=1)
# Pad to full T: before signal is 0, after signal is final cumsum
cumsum_full = np.zeros((N, T))
for i in range(N):
    cumsum_full[i, sig_start:sig_end] = cumsum_signal[i]
    cumsum_full[i, sig_end:] = cumsum_signal[i, -1]

rule_latent_t = np.zeros((N, T))  # time-varying rule-relevant variable
for i in range(N):
    mi = int(mode[i])
    if mi == 0:  # integrate: cumulative sum up to t
        rule_latent_t[i] = cumsum_full[i]
    elif mi == 1:  # memory: early pulse (constant after pulse)
        rule_latent_t[i, :] = early_pulse[i]
    else:  # transient: second-half sum (constant)
        rule_latent_t[i, :] = sig_sum_second[i]

slopes_c = np.zeros((3, T))
slopes_d = np.zeros((3, T))
r2_c = np.zeros((3, T))
r2_d = np.zeros((3, T))

for mi in range(3):
    mask = mode == mi
    for t in range(T):
        x = rule_latent_t[mask, t].reshape(-1, 1)
        if np.std(x) < 1e-8:
            continue
        yc = q_c[mask, t]
        yd = q_d[mask, t]
        lrc = LinearRegression().fit(x, yc)
        lrd = LinearRegression().fit(x, yd)
        slopes_c[mi, t] = lrc.coef_[0]
        slopes_d[mi, t] = lrd.coef_[0]
        r2_c[mi, t] = max(0, lrc.score(x, yc))
        r2_d[mi, t] = max(0, lrd.score(x, yd))

# Print epoch summaries
print(f"\n  {'Mode':>12s} {'Epoch':>10s} {'C:slope':>8s} {'C:R2':>6s} {'D:slope':>8s} {'D:R2':>6s}")
for mi in range(3):
    for ename, (s, e) in epochs.items():
        sc = slopes_c[mi, s:e].mean()
        sd = slopes_d[mi, s:e].mean()
        rc = r2_c[mi, s:e].mean()
        rd = r2_d[mi, s:e].mean()
        print(f"  {MODE_NAMES[mi]:>12s} {ename:>10s} {sc:8.4f} {rc:6.3f} {sd:8.4f} {rd:6.3f}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
time = np.arange(T)
for mi, ax in enumerate(axes):
    for ename, (s, e) in epochs.items():
        ax.axvspan(s, e, alpha=0.08,
                   color={"cue":"#c7b8ea","signal":"#aec7e8","delay":"#ffbb78","response":"#98df8a"}[ename])
    ax.plot(time, slopes_c[mi], color="#d62728", lw=1.8, label="coupled")
    ax.plot(time, slopes_d[mi], color="#1f77b4", lw=1.8, label="decoupled")
    ax.set_ylabel(f"{MODE_NAMES[mi]}\nslope to rule", fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    if mi == 0: ax.legend(fontsize=8)
axes[-1].set_xlabel("timestep")
fig.suptitle("State-setting: time-varying rule-aligned transfer (q vs rule_relevant)", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "rule_aligned_transfer.png", dpi=150)
plt.close(fig)
print("  Saved rule_aligned_transfer.png")

np.savez_compressed(OUT / "state_setting_timevarying_transfer.npz",
    slopes_coupled=slopes_c, slopes_decoupled=slopes_d,
    r2_coupled=r2_c, r2_decoupled=r2_d,
    time=time.astype(np.float32))

# ── 3. Readout-aligned vs orthogonal decomposition ───────────────────
print("\n=== Readout vs orthogonal hidden-state decomposition ===")

w_dir = W_out[0] / (np.linalg.norm(W_out[0]) + 1e-8)  # (H,)

# For each trial and timestep: project xb onto w_dir and orthogonal complement
alpha_c = np.einsum("nth,h->nt", xb_c, w_dir)  # readout-aligned
orth_c = xb_c - alpha_c[:, :, None] * w_dir[None, None, :]
beta_norm_c = np.linalg.norm(orth_c, axis=2)

alpha_d = np.einsum("nth,h->nt", xb_d, w_dir)
orth_d = xb_d - alpha_d[:, :, None] * w_dir[None, None, :]
beta_norm_d = np.linalg.norm(orth_d, axis=2)

print(f"\n  {'Mode':>12s} {'Epoch':>10s} {'C:alpha':>8s} {'C:beta':>8s} {'D:alpha':>8s} {'D:beta':>8s}")
for mi in range(3):
    mask = mode == mi
    for ename, (s, e) in epochs.items():
        ac = alpha_c[mask, s:e].mean()
        bc = beta_norm_c[mask, s:e].mean()
        ad = alpha_d[mask, s:e].mean()
        bd = beta_norm_d[mask, s:e].mean()
        print(f"  {MODE_NAMES[mi]:>12s} {ename:>10s} {ac:8.4f} {bc:8.4f} {ad:8.4f} {bd:8.4f}")

# Mode separation in alpha vs beta
print(f"\n  Mode separation (std of per-mode means):")
print(f"  {'Epoch':>10s} {'C:alpha_sep':>12s} {'C:beta_sep':>12s} {'D:alpha_sep':>12s} {'D:beta_sep':>12s}")
for ename, (s, e) in epochs.items():
    ac_modes = [alpha_c[mode==mi, s:e].mean() for mi in range(3)]
    bc_modes = [beta_norm_c[mode==mi, s:e].mean() for mi in range(3)]
    ad_modes = [alpha_d[mode==mi, s:e].mean() for mi in range(3)]
    bd_modes = [beta_norm_d[mode==mi, s:e].mean() for mi in range(3)]
    print(f"  {ename:>10s} {np.std(ac_modes):12.4f} {np.std(bc_modes):12.4f} "
          f"{np.std(ad_modes):12.4f} {np.std(bd_modes):12.4f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
colors = ["#d62728", "#1f77b4", "#2ca02c"]
for mi in range(3):
    mask = mode == mi
    axes[0].plot(time, alpha_c[mask].mean(axis=0), color=colors[mi], lw=1.5,
                 label=f"{MODE_NAMES[mi]} (C)")
    axes[0].plot(time, alpha_d[mask].mean(axis=0), color=colors[mi], lw=1.5,
                 ls="--", label=f"{MODE_NAMES[mi]} (D)")
    axes[1].plot(time, beta_norm_c[mask].mean(axis=0), color=colors[mi], lw=1.5)
    axes[1].plot(time, beta_norm_d[mask].mean(axis=0), color=colors[mi], lw=1.5, ls="--")
for ax in axes:
    for ename, (s, e) in epochs.items():
        ax.axvspan(s, e, alpha=0.08,
                   color={"cue":"#c7b8ea","signal":"#aec7e8","delay":"#ffbb78","response":"#98df8a"}[ename])
axes[0].set_ylabel("alpha (readout-aligned)", fontsize=10)
axes[0].legend(fontsize=7, ncol=3, loc="upper left")
axes[1].set_ylabel("beta_norm (orthogonal)", fontsize=10)
axes[-1].set_xlabel("timestep")
fig.suptitle("Hidden state: readout-aligned vs orthogonal (solid=coupled, dashed=decoupled)", fontsize=10)
fig.tight_layout()
fig.savefig(FIGS / "readout_vs_orthogonal.png", dpi=150)
plt.close(fig)
print("  Saved readout_vs_orthogonal.png")

# ── 4. Paired-triplet analysis ────────────────────────────────────────
print("\n=== Paired-triplet analysis (same input, 3 modes) ===")
triplet_batch = task.sample_paired_triplets(80, split="val", device="cpu")
with torch.no_grad():
    trip_coupled = model(triplet_batch.inputs, intervention=None, add_noise=False)
    trip_decoupled = model(triplet_batch.inputs, intervention={"type": "decouple"}, add_noise=False)

tq_c = default_readout(trip_coupled).cpu().numpy().T
tq_d = default_readout(trip_decoupled).cpu().numpy().T
tmeta = {k: v.cpu().numpy() for k, v in triplet_batch.meta.items()}
tmode = tmeta["mode"]
ttriplet = tmeta["triplet_id"]
trule = tmeta["rule_relevant_value"]
tlabel = tmeta["label"]

# Sign-align
tsign = np.sign(tlabel)
tq_c_aligned = tq_c * tsign[:, None]
tq_d_aligned = tq_d * tsign[:, None]

# Per-mode mean traces (sign-aligned)
print(f"\n  Triplet sign-aligned traces (coupled):")
print(f"  {'Mode':>12s} {'cue':>8s} {'signal':>8s} {'delay':>8s} {'response':>8s}")
for mi in range(3):
    mask = tmode == mi
    vals = [tq_c_aligned[mask, s:e].mean() for ename, (s, e) in epochs.items()]
    print(f"  {MODE_NAMES[mi]:>12s} " + " ".join(f"{v:8.4f}" for v in vals))

# Within-triplet mode differences
print(f"\n  Within-triplet pairwise response |q| difference (coupled):")
resp_s, resp_e = epochs["response"]
for ti in range(min(5, 80)):
    idxs = np.where(ttriplet == ti)[0]
    if len(idxs) != 3:
        continue
    resp_vals = {int(tmode[i]): float(np.abs(tq_c[i, resp_s:resp_e]).mean()) for i in idxs}
    print(f"    triplet {ti}: " + "  ".join(f"{MODE_NAMES[mi]}={resp_vals.get(mi, 0):.4f}" for mi in range(3)))

# Aggregate: mode spread within triplets
spreads_c = []
spreads_d = []
for ti in range(80):
    idxs = np.where(ttriplet == ti)[0]
    if len(idxs) != 3:
        continue
    rc = [np.abs(tq_c[i, resp_s:resp_e]).mean() for i in idxs]
    rd = [np.abs(tq_d[i, resp_s:resp_e]).mean() for i in idxs]
    spreads_c.append(np.std(rc))
    spreads_d.append(np.std(rd))
print(f"\n  Within-triplet mode spread (std of 3 response |q|):")
print(f"    Coupled:   mean={np.mean(spreads_c):.4f}  std={np.std(spreads_c):.4f}")
print(f"    Decoupled: mean={np.mean(spreads_d):.4f}  std={np.std(spreads_d):.4f}")

# Plot triplet traces
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for mi, ax in enumerate(axes):
    mask = tmode == mi
    for ename, (s, e) in epochs.items():
        ax.axvspan(s, e, alpha=0.08,
                   color={"cue":"#c7b8ea","signal":"#aec7e8","delay":"#ffbb78","response":"#98df8a"}[ename])
    mn_c = tq_c_aligned[mask].mean(axis=0)
    mn_d = tq_d_aligned[mask].mean(axis=0)
    ax.plot(time, mn_c, color="#d62728", lw=1.8, label="coupled")
    ax.plot(time, mn_d, color="#1f77b4", lw=1.8, label="decoupled")
    ax.set_ylabel(f"{MODE_NAMES[mi]}", fontsize=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    if mi == 0: ax.legend(fontsize=8)
axes[-1].set_xlabel("timestep")
fig.suptitle("Paired triplets: same input, different mode (sign-aligned q)", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS / "triplet_traces.png", dpi=150)
plt.close(fig)
print("  Saved triplet_traces.png")

print("\n=== Done ===")
