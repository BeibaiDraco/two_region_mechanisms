from __future__ import annotations

import numpy as np
import torch


def default_readout(rollout: dict) -> torch.Tensor:
    outputs = rollout["outputs"]
    if outputs.shape[-1] == 1:
        return outputs[..., 0]
    return outputs.mean(dim=-1)


def summarize_response(q: torch.Tensor, batch) -> dict:
    q_np = q.detach().cpu().numpy()
    tgt = batch.targets[..., 0].detach().cpu().numpy()
    mask = batch.mask[..., 0].detach().cpu().numpy()
    per_trial_pred = []
    per_trial_tgt = []
    for b in range(q_np.shape[1]):
        active = mask[:, b] > 0
        if active.sum() == 0:
            continue
        per_trial_pred.append(float(q_np[active, b].mean()))
        per_trial_tgt.append(float(tgt[active, b].mean()))
    return {
        "pred_mean": float(np.mean(per_trial_pred)) if per_trial_pred else 0.0,
        "target_mean": float(np.mean(per_trial_tgt)) if per_trial_tgt else 0.0,
        "per_trial_pred": per_trial_pred,
        "per_trial_tgt": per_trial_tgt,
    }
