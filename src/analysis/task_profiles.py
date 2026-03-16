"""Shared engine for task-specific mechanism profiling.

Every task uses the same primitives (D, M, E, q traces), but each task
defines its own epochs, splits, and headline metrics via an analysis contract.
This module provides the reusable building blocks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# ── Epoch helpers ─────────────────────────────────────────────────────

def epoch_indices(epochs: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {name: np.arange(s, e) for name, (s, e) in epochs.items()}


def build_epoch_masks(epochs: Dict[str, Tuple[int, int]], T: int) -> Dict[str, np.ndarray]:
    masks = {}
    for name, (s, e) in epochs.items():
        m = np.zeros(T, dtype=np.float32)
        m[s:e] = 1.0
        masks[name] = m
    return masks


# ── Onset detection (fixed: event-aligned, baseline-subtracted, persistent) ──

def robust_onset(
    arr_1d: np.ndarray,
    baseline_idx: np.ndarray,
    threshold_frac: float = 0.15,
    min_consecutive: int = 3,
) -> int:
    """Event-aligned onset with baseline subtraction and persistence criterion.

    Returns first timestep where signal exceeds baseline_mean + threshold_frac * peak
    for at least min_consecutive timesteps.
    """
    if len(baseline_idx) == 0:
        baseline = 0.0
    else:
        baseline = float(arr_1d[baseline_idx].mean())

    centered = arr_1d - baseline
    peak = centered.max()
    if peak < 1e-8:
        return -1

    thresh = threshold_frac * peak
    above = centered > thresh

    count = 0
    for t in range(len(above)):
        if above[t]:
            count += 1
            if count >= min_consecutive:
                return t - min_consecutive + 1
        else:
            count = 0
    return -1


def peak_time(arr_1d: np.ndarray) -> int:
    return int(np.argmax(arr_1d))


# ── Per-trial feature extraction ─────────────────────────────────────

def epoch_auc(arr_1d: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return 0.0
    return float(np.trapezoid(arr_1d[idx]))


def epoch_peak(arr_1d: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return 0.0
    return float(arr_1d[idx].max())


def epoch_mean(arr_1d: np.ndarray, idx: np.ndarray) -> float:
    if len(idx) == 0:
        return 0.0
    return float(arr_1d[idx].mean())


def extract_trial_features(
    d: np.ndarray,
    m: np.ndarray,
    abs_q: np.ndarray,
    epoch_idx: Dict[str, np.ndarray],
    epoch_names: List[str],
    baseline_idx: np.ndarray,
) -> Dict[str, float]:
    """Extract features for one trial. d, m, abs_q are 1-D (T,)."""
    feat = {}
    for prefix, arr in [("d", d), ("m", m)]:
        for ep in epoch_names:
            idx = epoch_idx[ep]
            feat[f"{ep}_{prefix}_auc"] = epoch_auc(arr, idx)
            feat[f"{ep}_{prefix}_peak"] = epoch_peak(arr, idx)
            feat[f"{ep}_{prefix}_mean"] = epoch_mean(arr, idx)
        feat[f"{prefix}_onset"] = robust_onset(arr, baseline_idx)
        feat[f"{prefix}_peak_time"] = peak_time(arr)
        feat[f"{prefix}_auc_total"] = float(np.trapezoid(arr))
    feat["q_onset"] = robust_onset(abs_q, baseline_idx)
    feat["q_peak_time"] = peak_time(abs_q)
    return feat


def extract_all_trials(
    d: np.ndarray,
    m: np.ndarray,
    abs_q: np.ndarray,
    epoch_idx: Dict[str, np.ndarray],
    epoch_names: List[str],
    meta: Dict[str, np.ndarray],
    final_abs_q: np.ndarray,
    baseline_idx: np.ndarray,
) -> List[Dict]:
    """Extract features for all trials. d, m, abs_q are (N, T)."""
    N = d.shape[0]
    rows = []
    for i in range(N):
        row = {"trial_id": i, "final_abs_q": float(final_abs_q[i])}
        row.update(extract_trial_features(d[i], m[i], abs_q[i], epoch_idx, epoch_names, baseline_idx))
        for k, v in meta.items():
            val = v[i]
            if hasattr(val, 'item'):
                try:
                    row[k] = float(val)
                except (ValueError, TypeError):
                    row[k] = str(val)
            else:
                row[k] = val
        rows.append(row)
    return rows


# ── Trace statistics ──────────────────────────────────────────────────

def trace_band(arr: np.ndarray):
    """Return mean, p25, p75 over axis 0."""
    return np.mean(arr, axis=0), np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)


# ── Correlation helpers ───────────────────────────────────────────────

def _safe_pearsonr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return (np.nan, np.nan)
    return stats.pearsonr(a, b)


def partial_corr(feature: np.ndarray, target: np.ndarray, covariate: np.ndarray) -> Tuple[float, float]:
    """Pearson correlation after residualizing both on covariate."""
    cov = covariate.reshape(-1, 1)
    r_feat = feature - LinearRegression().fit(cov, feature).predict(cov)
    r_tgt = target - LinearRegression().fit(cov, target).predict(cov)
    return _safe_pearsonr(r_feat, r_tgt)


def unique_contribution(
    d_feat: np.ndarray,
    m_feat: np.ndarray,
    target: np.ndarray,
    covariate: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Unique predictive contribution of D and M for target.

    Reports:
    - raw D and M correlations
    - D after partialing out M (and covariate)
    - M after partialing out D (and covariate)
    """
    r_d, _ = _safe_pearsonr(d_feat, target)
    r_m, _ = _safe_pearsonr(m_feat, target)

    controls_for_d = m_feat.reshape(-1, 1)
    controls_for_m = d_feat.reshape(-1, 1)
    if covariate is not None:
        controls_for_d = np.column_stack([controls_for_d, covariate])
        controls_for_m = np.column_stack([controls_for_m, covariate])

    def _resid(y, X):
        return y - LinearRegression().fit(X, y).predict(X)

    r_d_unique, p_d = _safe_pearsonr(_resid(d_feat, controls_for_d), _resid(target, controls_for_d))
    r_m_unique, p_m = _safe_pearsonr(_resid(m_feat, controls_for_m), _resid(target, controls_for_m))

    return {
        "raw_d": round(float(r_d), 4) if not np.isnan(r_d) else np.nan,
        "raw_m": round(float(r_m), 4) if not np.isnan(r_m) else np.nan,
        "d_unique_of_m": round(float(r_d_unique), 4) if not np.isnan(r_d_unique) else np.nan,
        "d_unique_p": round(float(p_d), 6) if not np.isnan(p_d) else np.nan,
        "m_unique_of_d": round(float(r_m_unique), 4) if not np.isnan(r_m_unique) else np.nan,
        "m_unique_p": round(float(p_m), 6) if not np.isnan(p_m) else np.nan,
    }


def strict_lag_prediction(
    predictor: np.ndarray,
    outcome: np.ndarray,
    predictor_idx: np.ndarray,
    outcome_idx: np.ndarray,
) -> Tuple[float, float]:
    """Correlate predictor window mean with outcome window mean.
    Caller is responsible for ensuring predictor_idx ends before outcome_idx starts.
    """
    p_vals = predictor[:, predictor_idx].mean(axis=1)
    o_vals = outcome[:, outcome_idx].mean(axis=1)
    return _safe_pearsonr(p_vals, o_vals)


# ── Condition-split helpers ───────────────────────────────────────────

def condition_traces(arr: np.ndarray, condition: np.ndarray) -> Dict:
    result = {}
    for val in np.unique(condition):
        mask = condition == val
        result[val] = {"mean": arr[mask].mean(axis=0), "n": int(mask.sum())}
    return result


def median_split_within(arr: np.ndarray, group_var: np.ndarray, split_var: np.ndarray) -> Dict:
    result = {}
    for gval in np.unique(group_var):
        gmask = group_var == gval
        med = np.median(split_var[gmask])
        hi = gmask & (split_var >= med)
        lo = gmask & (split_var < med)
        if lo.sum() == 0:
            lo = gmask & (split_var <= med)
        result[gval] = {
            "hi_mean": arr[hi].mean(axis=0) if hi.sum() > 0 else np.zeros(arr.shape[1]),
            "lo_mean": arr[lo].mean(axis=0) if lo.sum() > 0 else np.zeros(arr.shape[1]),
            "n_hi": int(hi.sum()),
            "n_lo": int(lo.sum()),
        }
    return result


def windowed_mean(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return arr[:, idx].mean(axis=1)


# ── Divergence time (fixed: baseline-subtracted, persistent) ─────────

def robust_divergence_time(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    baseline_idx: np.ndarray,
    threshold_frac: float = 0.2,
    min_consecutive: int = 3,
) -> int:
    """Event-aligned divergence with baseline subtraction and persistence."""
    diff = np.abs(trace_a - trace_b)
    if len(baseline_idx) > 0:
        baseline_diff = diff[baseline_idx].mean()
    else:
        baseline_diff = 0.0

    centered = diff - baseline_diff
    peak = centered.max()
    if peak < 1e-8:
        return -1

    thresh = threshold_frac * peak
    above = centered > thresh
    count = 0
    for t in range(len(above)):
        if above[t]:
            count += 1
            if count >= min_consecutive:
                return t - min_consecutive + 1
        else:
            count = 0
    return -1


# ── Context task: matched-input comparisons ───────────────────────────

def context_selectivity_index(
    q_trace: np.ndarray,
    context: np.ndarray,
    epoch_idx: np.ndarray,
) -> float:
    """How much does B's readout in the given epoch depend on context?
    Returns mean |q_ctx0 - q_ctx1| / (mean |q_ctx0| + mean |q_ctx1| + eps).
    """
    ctx0 = q_trace[context == 0]
    ctx1 = q_trace[context == 1]
    if ctx0.shape[0] == 0 or ctx1.shape[0] == 0:
        return 0.0
    mean0 = np.abs(ctx0[:, epoch_idx]).mean()
    mean1 = np.abs(ctx1[:, epoch_idx]).mean()
    diff = np.abs(ctx0[:, epoch_idx].mean(axis=0) - ctx1[:, epoch_idx].mean(axis=0)).mean()
    return float(diff / (mean0 + mean1 + 1e-8))


def matched_input_context_comparison(
    arr: np.ndarray,
    context: np.ndarray,
    relevant_coh: np.ndarray,
    n_bins: int = 4,
) -> Dict:
    """For matched relevant-coherence bins, compare traces by context."""
    edges = np.quantile(np.abs(relevant_coh), np.linspace(0, 1, n_bins + 1))
    result = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            bin_mask = (np.abs(relevant_coh) >= lo) & (np.abs(relevant_coh) <= hi)
        else:
            bin_mask = (np.abs(relevant_coh) >= lo) & (np.abs(relevant_coh) < hi)
        ctx0 = bin_mask & (context == 0)
        ctx1 = bin_mask & (context == 1)
        if ctx0.sum() < 3 or ctx1.sum() < 3:
            continue
        result[i] = {
            "coh_range": (round(float(lo), 3), round(float(hi), 3)),
            "ctx0_mean": arr[ctx0].mean(axis=0),
            "ctx1_mean": arr[ctx1].mean(axis=0),
            "n_ctx0": int(ctx0.sum()),
            "n_ctx1": int(ctx1.sum()),
        }
    return result


# ── State-setting task: same-probe different-mode ─────────────────────

def mode_conditioned_traces(
    arr: np.ndarray,
    mode: np.ndarray,
    mode_names: Dict[int, str],
) -> Dict:
    result = {}
    for val, name in mode_names.items():
        mask = mode == val
        if mask.sum() == 0:
            continue
        result[name] = {"mean": arr[mask].mean(axis=0), "n": int(mask.sum())}
    return result


# ── Redundant task: difficulty-conditioned usage ──────────────────────

def difficulty_conditioned_effect(
    coupled_q: np.ndarray,
    decoupled_q: np.ndarray,
    difficulty: np.ndarray,
    epoch_idx: np.ndarray,
) -> List[Dict]:
    """Decoupling cost stratified by difficulty."""
    rows = []
    for dval in np.sort(np.unique(difficulty)):
        mask = difficulty == dval
        if mask.sum() < 3:
            continue
        c_mean = np.abs(coupled_q[mask][:, epoch_idx]).mean()
        d_mean = np.abs(decoupled_q[mask][:, epoch_idx]).mean()
        cost = c_mean - d_mean
        rows.append({
            "difficulty": round(float(dval), 3),
            "n": int(mask.sum()),
            "coupled_mean_abs_q": round(float(c_mean), 4),
            "decoupled_mean_abs_q": round(float(d_mean), 4),
            "decoupling_cost": round(float(cost), 4),
            "relative_cost": round(float(cost / (c_mean + 1e-8)), 4),
        })
    return rows
