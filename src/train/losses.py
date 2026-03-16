from __future__ import annotations

import torch


def masked_mse(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((outputs - targets) ** 2) * mask).sum() / denom


def activity_regularization(rollout: dict, coeff: float) -> torch.Tensor:
    if coeff <= 0:
        device = rollout["outputs"].device
        return torch.tensor(0.0, device=device)
    return coeff * (rollout["r_a"].pow(2).mean() + rollout["r_b"].pow(2).mean())


def weight_regularization(model, coeff: float) -> torch.Tensor:
    device = next(model.parameters()).device
    if coeff <= 0:
        return torch.tensor(0.0, device=device)
    penalty = torch.tensor(0.0, device=device)
    for param in model.parameters():
        penalty = penalty + param.pow(2).mean()
    return coeff * penalty


def communication_regularization(model, coeff: float) -> torch.Tensor:
    device = next(model.parameters()).device
    if coeff <= 0:
        return torch.tensor(0.0, device=device)
    return coeff * model.communication_weight_penalty()
