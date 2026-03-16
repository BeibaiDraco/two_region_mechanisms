from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .interventions import build_intervention
from .readouts import default_readout


def _per_trial_stats(score_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute per-trial summary statistics from a (T, B) score tensor."""
    abs_scores = score_tensor.abs().cpu().numpy()
    return {
        "per_trial_max": abs_scores.max(axis=0),
        "per_trial_mean": abs_scores.mean(axis=0),
        "per_trial_auc": np.trapz(abs_scores, axis=0),
    }


@torch.no_grad()
def estimate_realized_effect(coupled: dict, decoupled: dict) -> Dict[str, object]:
    q_c = default_readout(coupled)
    q_0 = default_readout(decoupled)
    delta = q_c - q_0
    max_abs = float(delta.abs().max().cpu())
    mean_abs = float(delta.abs().mean().cpu())

    divergence_times = []
    threshold = max(1e-6, 0.1 * max_abs)
    for b in range(delta.shape[1]):
        above = torch.where(delta[:, b].abs() > threshold)[0]
        divergence_times.append(int(above[0].item()) if len(above) else -1)

    valid_times = [t for t in divergence_times if t >= 0]
    mean_ttd = float(np.mean(valid_times)) if valid_times else -1.0

    trial_stats = _per_trial_stats(delta)

    return {
        "q_c": q_c,
        "q_0": q_0,
        "delta_q": delta,
        "trace": delta.abs().cpu().numpy(),
        "max_abs_delta_q": max_abs,
        "mean_abs_delta_q": mean_abs,
        "auc_delta_q": float(np.trapz(delta.abs().cpu().numpy(), axis=0).mean()),
        "time_to_divergence_mean": mean_ttd,
        "per_trial_divergence_time": divergence_times,
        **{f"E_{k}": v for k, v in trial_stats.items()},
    }


@torch.no_grad()
def estimate_direct_forcing(model, batch, coupled: dict) -> Dict[str, object]:
    seq_len, batch_size = batch.inputs.shape[0], batch.inputs.shape[1]
    scores = []
    for t in range(seq_len):
        x_a = coupled["x_a"][t - 1] if t > 0 else torch.zeros(batch_size, model.hidden_size_a, device=batch.inputs.device)
        x_b = coupled["x_b"][t - 1] if t > 0 else torch.zeros(batch_size, model.hidden_size_b, device=batch.inputs.device)
        u_t = batch.inputs[t]

        step_full = model.step(x_a, x_b, u_t, step_idx=t, intervention=None, add_noise=False)
        step_dec = model.step(x_a, x_b, u_t, step_idx=t, intervention={"type": "decouple"}, add_noise=False)
        q_full = step_full["output"][..., 0]
        q_dec = step_dec["output"][..., 0]
        scores.append(q_full - q_dec)

    score_tensor = torch.stack(scores, dim=0)
    trial_stats = _per_trial_stats(score_tensor)

    return {
        "score_tensor": score_tensor,
        "trace": score_tensor.abs().cpu().numpy(),
        "max_abs_direct": float(score_tensor.abs().max().cpu()),
        "mean_abs_direct": float(score_tensor.abs().mean().cpu()),
        "auc_direct": float(np.trapz(score_tensor.abs().cpu().numpy(), axis=0).mean()),
        **{f"D_{k}": v for k, v in trial_stats.items()},
    }


@torch.no_grad()
def estimate_modulation(model, batch, coupled: dict, decoupled: dict) -> Dict[str, object]:
    seq_len, batch_size = batch.inputs.shape[0], batch.inputs.shape[1]
    scores = []
    for t in range(seq_len):
        if t > 0:
            x_a_base = decoupled["x_a"][t - 1]
            x_b_base = decoupled["x_b"][t - 1]
            x_b_c = coupled["x_b"][t - 1]
        else:
            x_a_base = torch.zeros(batch_size, model.hidden_size_a, device=batch.inputs.device)
            x_b_base = torch.zeros(batch_size, model.hidden_size_b, device=batch.inputs.device)
            x_b_c = torch.zeros(batch_size, model.hidden_size_b, device=batch.inputs.device)

        delta = x_b_c - x_b_base
        grad = model.readout_gradient(x_b_base)
        denom = grad.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = (delta * grad).sum(dim=-1, keepdim=True) / denom * grad
        delta_private = delta - aligned
        x_b_private = torch.clamp(x_b_base + delta_private, min=-5.0, max=5.0)

        u_t = batch.inputs[t]
        step_base = model.step(x_a_base, x_b_base, u_t, step_idx=t, intervention={"type": "decouple"}, add_noise=False)
        step_priv = model.step(x_a_base, x_b_private, u_t, step_idx=t, intervention={"type": "decouple"}, add_noise=False)

        q_base = step_base["output"][..., 0]
        q_priv = step_priv["output"][..., 0]
        scores.append(q_priv - q_base)

    score_tensor = torch.stack(scores, dim=0)
    trial_stats = _per_trial_stats(score_tensor)

    return {
        "score_tensor": score_tensor,
        "trace": score_tensor.abs().cpu().numpy(),
        "max_abs_modulation": float(score_tensor.abs().max().cpu()),
        "mean_abs_modulation": float(score_tensor.abs().mean().cpu()),
        "auc_modulation": float(np.trapz(score_tensor.abs().cpu().numpy(), axis=0).mean()),
        **{f"M_{k}": v for k, v in trial_stats.items()},
    }


@torch.no_grad()
def classify_mechanism(
    model,
    task,
    batch,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    thresholds = thresholds or {}
    effect_threshold = float(thresholds.get("effect_threshold", 0.15))
    direct_threshold = float(thresholds.get("direct_threshold", 0.08))
    modulation_threshold = float(thresholds.get("modulation_threshold", 0.05))

    coupled = model(batch.inputs, intervention=None, add_noise=False)
    decoupled = model(batch.inputs, intervention={"type": "decouple"}, add_noise=False)

    effect = estimate_realized_effect(coupled, decoupled)
    direct = estimate_direct_forcing(model, batch, coupled)
    modulation = estimate_modulation(model, batch, coupled, decoupled)

    D = int(direct["max_abs_direct"] > direct_threshold)
    M = int(modulation["max_abs_modulation"] > modulation_threshold)
    E = int(effect["max_abs_delta_q"] > effect_threshold)

    summary = {
        "D_J": D,
        "M_J": M,
        "E_J": E,
        "mechanism_class": f"({D},{M})",
        "max_abs_delta_q": effect["max_abs_delta_q"],
        "mean_abs_delta_q": effect["mean_abs_delta_q"],
        "auc_delta_q": effect["auc_delta_q"],
        "time_to_divergence_mean": effect["time_to_divergence_mean"],
        "max_abs_direct": direct["max_abs_direct"],
        "mean_abs_direct": direct["mean_abs_direct"],
        "auc_direct": direct["auc_direct"],
        "max_abs_modulation": modulation["max_abs_modulation"],
        "mean_abs_modulation": modulation["mean_abs_modulation"],
        "auc_modulation": modulation["auc_modulation"],
    }
    return {
        "summary": summary,
        "coupled": coupled,
        "decoupled": decoupled,
        "effect": effect,
        "direct": direct,
        "modulation": modulation,
    }


# ---------------------------------------------------------------------------
# Threshold sweep + max+mean classifier
# ---------------------------------------------------------------------------

def classify_from_arrays(
    d_trial_max: np.ndarray,
    d_trial_mean: np.ndarray,
    m_trial_max: np.ndarray,
    m_trial_mean: np.ndarray,
    d_thresh: float,
    m_thresh: float,
    rule: str = "max",
    mean_factor: float = 0.5,
) -> dict:
    """Classify using pre-computed per-trial arrays at a given threshold pair.

    Rules:
      "max"      — D=1 if any trial max > d_thresh (original batch-level rule)
      "max_mean" — D=1 if trial max > d_thresh AND trial mean > mean_factor * d_thresh
                   (requires sustained effect, not just a spike)
    """
    if rule == "max":
        D_batch = int(d_trial_max.max() > d_thresh)
        M_batch = int(m_trial_max.max() > m_thresh)
        frac_D = float((d_trial_max > d_thresh).mean())
        frac_M = float((m_trial_max > m_thresh).mean())
    elif rule == "max_mean":
        d_pass = (d_trial_max > d_thresh) & (d_trial_mean > mean_factor * d_thresh)
        m_pass = (m_trial_max > m_thresh) & (m_trial_mean > mean_factor * m_thresh)
        D_batch = int(d_pass.any())
        M_batch = int(m_pass.any())
        frac_D = float(d_pass.mean())
        frac_M = float(m_pass.mean())
    else:
        raise ValueError(f"Unknown rule: {rule}")

    return {
        "D_J": D_batch,
        "M_J": M_batch,
        "mechanism_class": f"({D_batch},{M_batch})",
        "frac_trials_D": frac_D,
        "frac_trials_M": frac_M,
    }


def threshold_sweep(
    d_trial_max: np.ndarray,
    d_trial_mean: np.ndarray,
    m_trial_max: np.ndarray,
    m_trial_mean: np.ndarray,
    d_range: np.ndarray,
    m_range: np.ndarray,
    rule: str = "max",
    mean_factor: float = 0.5,
) -> dict:
    """Sweep over D/M threshold pairs, returning 2-D arrays for heatmaps.

    Returns arrays indexed as [i_d, i_m] where i_d indexes d_range, i_m indexes m_range.
    """
    nd, nm = len(d_range), len(m_range)
    class_grid = np.empty((nd, nm), dtype=object)
    frac_mixed = np.zeros((nd, nm))
    frac_direct_only = np.zeros((nd, nm))
    frac_mod_only = np.zeros((nd, nm))
    frac_none = np.zeros((nd, nm))
    frac_D_trials = np.zeros((nd, nm))
    frac_M_trials = np.zeros((nd, nm))

    for i, dt in enumerate(d_range):
        for j, mt in enumerate(m_range):
            r = classify_from_arrays(
                d_trial_max, d_trial_mean, m_trial_max, m_trial_mean,
                dt, mt, rule=rule, mean_factor=mean_factor,
            )
            class_grid[i, j] = r["mechanism_class"]
            frac_D_trials[i, j] = r["frac_trials_D"]
            frac_M_trials[i, j] = r["frac_trials_M"]

            D, M = r["D_J"], r["M_J"]
            frac_mixed[i, j] = float(D == 1 and M == 1)
            frac_direct_only[i, j] = float(D == 1 and M == 0)
            frac_mod_only[i, j] = float(D == 0 and M == 1)
            frac_none[i, j] = float(D == 0 and M == 0)

    return {
        "d_range": d_range,
        "m_range": m_range,
        "class_grid": class_grid,
        "frac_mixed": frac_mixed,
        "frac_direct_only": frac_direct_only,
        "frac_mod_only": frac_mod_only,
        "frac_none": frac_none,
        "frac_D_trials": frac_D_trials,
        "frac_M_trials": frac_M_trials,
    }
