#!/usr/bin/env python
"""Generate final per-task mechanism profiles and cross-task summary table."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.mechanism_classification import (
    estimate_realized_effect, estimate_direct_forcing, estimate_modulation)
from src.analysis.readouts import default_readout
from src.analysis.rollout import load_run_artifacts
from src.analysis.task_profiles import (
    epoch_indices, robust_onset, unique_contribution, windowed_mean,
    robust_divergence_time, context_selectivity_index,
    difficulty_conditioned_effect, trace_band)
from src.utils.io import save_json

N_TRIALS = 256

def _collect(run_dir):
    cfg, task, model, _ = load_run_artifacts(run_dir, checkpoint_name="best.pt", map_location="cpu")
    batch = task.sample_batch(N_TRIALS, split="val", device="cpu")
    model.eval()
    with torch.no_grad():
        coupled = model(batch.inputs, intervention=None, add_noise=False)
        decoupled = model(batch.inputs, intervention={"type": "decouple"}, add_noise=False)
    return cfg, task, model, batch, coupled, decoupled

def _base_profile(cfg, task, batch, coupled, decoupled):
    q_c = default_readout(coupled).cpu().numpy().T
    q_d = default_readout(decoupled).cpu().numpy().T
    d = estimate_direct_forcing(task._hack_model, batch, coupled)["trace"].T
    m = estimate_modulation(task._hack_model, batch, coupled, decoupled)["trace"].T
    e = estimate_realized_effect(coupled, decoupled)["trace"].T
    return q_c, q_d, d, m, e

def _onsets(d, m, abs_q, bl):
    d_on = np.array([robust_onset(d[i], bl) for i in range(d.shape[0])])
    m_on = np.array([robust_onset(m[i], bl) for i in range(m.shape[0])])
    q_on = np.array([robust_onset(abs_q[i], bl) for i in range(abs_q.shape[0])])
    f = lambda a: round(float(a[a>=0].mean()), 1) if (a>=0).any() else -1
    return f(d_on), f(m_on), f(q_on)

def _accuracy(q, batch):
    resp_q = q[:, -1]  # simplified
    lbl = batch.meta["label"].cpu().numpy()
    return float((np.sign(resp_q) == np.sign(lbl)).mean())


# =====================================================================
# Per-task profile generators
# =====================================================================

def profile_binary(run_dir):
    cfg, task, model, batch, coupled, decoupled = _collect(run_dir)
    task._hack_model = model
    q_c, q_d, d, m, e = _base_profile(cfg, task, batch, coupled, decoupled)
    ep = epoch_indices(task.epochs)
    bl = ep[list(task.epochs.keys())[0]]
    coh = batch.meta["coherence"].cpu().numpy()
    diff = np.abs(coh)
    resp_idx = ep["response"]
    stim_idx = ep["stimulus"]
    faq = np.abs(q_c[:, resp_idx].mean(axis=1))

    d_on, m_on, q_on = _onsets(d, m, np.abs(q_c), bl)
    uc = unique_contribution(
        windowed_mean(d, stim_idx), windowed_mean(m, stim_idx),
        windowed_mean(np.abs(q_c), resp_idx), diff)

    label = "direct_dominant" if uc["d_unique_of_m"] > 0.3 and uc["m_unique_of_d"] < 0.1 else "weak_mixed"

    return {
        "task": "BinaryCategorizationTask", "model": cfg["model"]["name"],
        "primary_question": "Direct transmission vs stimulus-period modulation",
        "D_onset": d_on, "M_onset": m_on, "q_onset": q_on,
        "stim_D_unique_of_M": uc["d_unique_of_m"],
        "stim_M_unique_of_D": uc["m_unique_of_d"],
        "D_onset_before_M": d_on < m_on if d_on >= 0 and m_on >= 0 else None,
        "strategy_label": label,
    }

def profile_delayed(run_dir):
    cfg, task, model, batch, coupled, decoupled = _collect(run_dir)
    task._hack_model = model
    q_c, q_d, d, m, e = _base_profile(cfg, task, batch, coupled, decoupled)
    ep = epoch_indices(task.epochs)
    bl = ep[list(task.epochs.keys())[0]]
    coh = batch.meta["coherence"].cpu().numpy()
    diff = np.abs(coh)
    resp_idx = ep["response"]
    stim_idx = ep["stimulus"]
    delay_idx = ep["delay"]

    d_on, m_on, q_on = _onsets(d, m, np.abs(q_c), bl)

    uc_stim = unique_contribution(
        windowed_mean(d, stim_idx), windowed_mean(m, stim_idx),
        windowed_mean(np.abs(q_c), resp_idx), diff)
    uc_delay = unique_contribution(
        windowed_mean(d, delay_idx), windowed_mean(m, delay_idx),
        windowed_mean(np.abs(q_c), resp_idx), diff)
    late_delay = delay_idx[len(delay_idx)//2:]
    uc_stim_ld = unique_contribution(
        windowed_mean(d, stim_idx), windowed_mean(m, stim_idx),
        windowed_mean(np.abs(q_c), late_delay), diff)

    m_first = m_on < d_on if m_on >= 0 and d_on >= 0 else None
    m_strong = uc_delay["m_unique_of_d"] > 0.3

    if m_first and m_strong:
        label = "M_first_delay_supported"
    elif m_strong:
        label = "delay_conditioning"
    else:
        label = "temporally_mixed"

    return {
        "task": "DelayedCategorizationTask", "model": cfg["model"]["name"],
        "primary_question": "Direct write-in vs delay-period modulation vs temporal staging",
        "D_onset": d_on, "M_onset": m_on, "q_onset": q_on,
        "stim_D_unique": uc_stim["d_unique_of_m"],
        "stim_M_unique": uc_stim["m_unique_of_d"],
        "delay_D_unique": uc_delay["d_unique_of_m"],
        "delay_M_unique": uc_delay["m_unique_of_d"],
        "stim_to_late_delay_D_unique": uc_stim_ld["d_unique_of_m"],
        "stim_to_late_delay_M_unique": uc_stim_ld["m_unique_of_d"],
        "M_onset_before_D": m_first,
        "strategy_label": label,
    }

def profile_context(run_dir):
    cfg, task, model, batch, coupled, decoupled = _collect(run_dir)
    q_c = default_readout(coupled).cpu().numpy().T
    q_d = default_readout(decoupled).cpu().numpy().T
    ep = epoch_indices(task.epochs)
    meta = {k: v.cpu().numpy() for k, v in batch.meta.items()}
    ctx = meta["context"]
    coh1, coh2 = meta["coh1"], meta["coh2"]
    rel = np.where(ctx == 0, coh1, coh2)
    irrel = np.where(ctx == 0, coh2, coh1)

    resp_idx = ep["response"]
    T = q_c.shape[1]

    # Response routing
    X = np.column_stack([rel, irrel])
    fq_c = q_c[:, resp_idx].mean(axis=1)
    fq_d = q_d[:, resp_idx].mean(axis=1)
    lr_c = LinearRegression().fit(X, fq_c)
    lr_d = LinearRegression().fit(X, fq_d)

    rel_slope_c = lr_c.coef_[0]
    irrel_slope_c = lr_c.coef_[1]
    sel_c = rel_slope_c - abs(irrel_slope_c)
    rel_slope_d = lr_d.coef_[0]
    irrel_slope_d = lr_d.coef_[1]
    sel_d = rel_slope_d - abs(irrel_slope_d)

    # Time-resolved
    rel_slopes = np.zeros(T)
    irrel_slopes = np.zeros(T)
    for t in range(T):
        lr = LinearRegression().fit(X, q_c[:, t])
        rel_slopes[t] = lr.coef_[0]
        irrel_slopes[t] = lr.coef_[1]

    # Epoch-averaged routing
    routing_by_epoch = {}
    for ename, (s, e_) in task.epochs.items():
        r = rel_slopes[s:e_].mean()
        i = irrel_slopes[s:e_].mean()
        eps = 1e-6
        ns = (r - abs(i)) / (abs(r) + abs(i) + eps) if abs(r) + abs(i) > 0.01 else 0
        routing_by_epoch[ename] = {"relevant": round(float(r), 4),
                                    "irrelevant": round(float(i), 4),
                                    "norm_selectivity": round(float(ns), 4)}

    label = "routing_modulatory" if sel_c > 1.0 else "weak_routing"

    return {
        "task": "ContextDependentDecisionTask", "model": cfg["model"]["name"],
        "primary_question": "Context regime setting vs evidence routing vs irrelevant suppression",
        "response_relevant_slope": round(float(rel_slope_c), 4),
        "response_irrelevant_leakage": round(float(abs(irrel_slope_c)), 4),
        "response_routing_selectivity": round(float(sel_c), 4),
        "decoupled_routing_selectivity": round(float(sel_d), 4),
        "selectivity_gain_from_coupling": round(float(sel_c - sel_d), 4),
        "routing_by_epoch": routing_by_epoch,
        "strategy_label": label,
    }

def profile_state_setting(run_dir):
    cfg, task, model, batch, coupled, decoupled = _collect(run_dir)
    q_c = default_readout(coupled).cpu().numpy().T
    q_d = default_readout(decoupled).cpu().numpy().T
    xb_c = coupled["x_b"].cpu().numpy().transpose(1, 0, 2)
    xb_d = decoupled["x_b"].cpu().numpy().transpose(1, 0, 2)
    W_out = model.w_out.weight.detach().cpu().numpy()[0]
    w_dir = W_out / (np.linalg.norm(W_out) + 1e-8)

    meta = {k: v.cpu().numpy() for k, v in batch.meta.items()}
    mode = meta["mode"]
    ep = epoch_indices(task.epochs)
    resp_idx = ep["response"]

    # Alpha/beta
    alpha_c = np.einsum("nth,h->nt", xb_c, w_dir)
    beta_c = np.linalg.norm(xb_c - alpha_c[:, :, None] * w_dir[None, None, :], axis=2)
    alpha_d = np.einsum("nth,h->nt", xb_d, w_dir)
    beta_d = np.linalg.norm(xb_d - alpha_d[:, :, None] * w_dir[None, None, :], axis=2)

    # Mode separation per epoch
    sep = {}
    for ename, (s, e_) in task.epochs.items():
        a_modes = [alpha_c[mode == mi, s:e_].mean() for mi in range(3)]
        b_modes = [beta_c[mode == mi, s:e_].mean() for mi in range(3)]
        sep[ename] = {"alpha_sep": round(float(np.std(a_modes)), 4),
                       "beta_sep": round(float(np.std(b_modes)), 4)}

    # Response mode spread
    c_resp = [np.abs(q_c[mode == mi][:, resp_idx]).mean() for mi in range(3)]
    d_resp = [np.abs(q_d[mode == mi][:, resp_idx]).mean() for mi in range(3)]

    # Triplets
    trip_batch = task.sample_paired_triplets(80, split="val", device="cpu")
    with torch.no_grad():
        tc = model(trip_batch.inputs, intervention=None, add_noise=False)
        td = model(trip_batch.inputs, intervention={"type": "decouple"}, add_noise=False)
    txb_c = tc["x_b"].cpu().numpy().transpose(1, 0, 2)
    txb_d = td["x_b"].cpu().numpy().transpose(1, 0, 2)
    tmode = trip_batch.meta["mode"].cpu().numpy()
    ttriplet = trip_batch.meta["triplet_id"].cpu().numpy()
    tq_c = default_readout(tc).cpu().numpy().T
    tq_d = default_readout(td).cpu().numpy().T

    tbeta_c = np.linalg.norm(txb_c - np.einsum("nth,h->nt", txb_c, w_dir)[:, :, None] * w_dir[None, None, :], axis=2)
    tbeta_d = np.linalg.norm(txb_d - np.einsum("nth,h->nt", txb_d, w_dir)[:, :, None] * w_dir[None, None, :], axis=2)

    spreads_c, spreads_d = [], []
    for ti in range(80):
        idxs = np.where(ttriplet == ti)[0]
        if len(idxs) != 3: continue
        resp_s, resp_e = task.epochs["response"]
        rc = [np.abs(tq_c[i, resp_s:resp_e]).mean() for i in idxs]
        rd = [np.abs(tq_d[i, resp_s:resp_e]).mean() for i in idxs]
        spreads_c.append(np.std(rc))
        spreads_d.append(np.std(rd))

    label = "off_axis_mode_control" if sep["response"]["beta_sep"] > 2 * sep["response"]["alpha_sep"] else "regime_setting_modulatory"

    return {
        "task": "StateSettingTask", "model": cfg["model"]["name"],
        "primary_question": "Cue-dependent regime setting: readout-axis vs off-axis hidden state",
        "mode_separation_by_epoch": sep,
        "response_beta_separation": sep["response"]["beta_sep"],
        "response_alpha_separation": sep["response"]["alpha_sep"],
        "response_mode_spread_coupled": round(float(np.std(c_resp)), 4),
        "response_mode_spread_decoupled": round(float(np.std(d_resp)), 4),
        "triplet_q_spread_coupled": round(float(np.mean(spreads_c)), 4),
        "triplet_q_spread_decoupled": round(float(np.mean(spreads_d)), 4),
        "strategy_label": label,
    }

def profile_redundant(run_dir):
    cfg, task, model, batch, coupled, decoupled = _collect(run_dir)
    task._hack_model = model
    q_c, q_d, d, m, e = _base_profile(cfg, task, batch, coupled, decoupled)
    ep = epoch_indices(task.epochs)
    bl = ep[list(task.epochs.keys())[0]]
    coh = batch.meta["coherence"].cpu().numpy()
    diff = np.abs(coh)
    resp_idx = ep["response"]
    stim_idx = ep["stimulus"]

    d_on, m_on, q_on = _onsets(d, m, np.abs(q_c), bl)
    uc = unique_contribution(
        windowed_mean(d, stim_idx), windowed_mean(m, stim_idx),
        windowed_mean(np.abs(q_c), resp_idx), diff)

    cost_table = difficulty_conditioned_effect(q_c, q_d, diff, resp_idx)
    rel_costs = [r["relative_cost"] for r in cost_table]
    mean_rel = np.mean(rel_costs)

    label = "direct_backup" if uc["d_unique_of_m"] > 0.3 and uc["m_unique_of_d"] < 0.1 else "weak_use"

    return {
        "task": "RedundantInputControlTask", "model": cfg["model"]["name"],
        "primary_question": "Backup/direct use when B has redundant evidence",
        "D_onset": d_on, "M_onset": m_on, "q_onset": q_on,
        "stim_D_unique_of_M": uc["d_unique_of_m"],
        "stim_M_unique_of_D": uc["m_unique_of_d"],
        "mean_relative_decoupling_cost": round(float(mean_rel), 4),
        "decoupling_cost_by_difficulty": cost_table,
        "strategy_label": label,
    }


# =====================================================================
# Main
# =====================================================================

RUNS = {
    "binary_additive": ("outputs/experiments/binary_additive", profile_binary),
    "delayed_gated": ("outputs/experiments/delayed_gated", profile_delayed),
    "context_gated": ("outputs/experiments/context_gated", profile_context),
    "state_setting_reciprocal": ("outputs/experiments/state_setting_reciprocal", profile_state_setting),
    "redundant_lowrank": ("outputs/experiments/redundant_lowrank", profile_redundant),
}

summary_rows = []
for name, (run_dir, fn) in RUNS.items():
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    profile = fn(run_dir)
    out_dir = Path(run_dir)
    save_json(profile, out_dir / "metrics" / f"{name}_final_profile.json")
    print(json.dumps(profile, indent=2, default=str))

    row = {"experiment": name, "task": profile["task"],
           "strategy_label": profile["strategy_label"],
           "primary_question": profile["primary_question"]}
    for k, v in profile.items():
        if k in row or isinstance(v, (dict, list)):
            continue
        row[k] = v
    summary_rows.append(row)

df = pd.DataFrame(summary_rows)
df.to_csv("outputs/experiments/final_cross_task_summary.csv", index=False)
print(f"\n{'='*60}")
print("  FINAL CROSS-TASK SUMMARY")
print(f"{'='*60}")
print(df[["experiment", "strategy_label"]].to_string(index=False))
print(f"\nSaved final_cross_task_summary.csv")
