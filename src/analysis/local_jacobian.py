from __future__ import annotations

import torch

from .interventions import build_intervention


def compute_local_jacobian(model, x_a: torch.Tensor, x_b: torch.Tensor, u_t: torch.Tensor, decoupled: bool = True) -> torch.Tensor:
    """
    Jacobian of the one-step B update wrt x_b for a single batch element.
    Returns [Hb, Hb].
    """
    if x_a.dim() != 2 or x_b.dim() != 2 or u_t.dim() != 2:
        raise ValueError("Expected batched tensors with leading batch dimension")
    if x_a.shape[0] != 1 or x_b.shape[0] != 1 or u_t.shape[0] != 1:
        raise ValueError("Pass a single batch element for local Jacobian")

    x_a_ = x_a.detach()
    x_b_ = x_b.detach().clone().requires_grad_(True)
    intervention = {"type": "decouple"} if decoupled else None
    step = model.step(x_a_, x_b_, u_t.detach(), step_idx=0, intervention=intervention, add_noise=False)
    x_b_next = step["x_b"]

    jac_rows = []
    for idx in range(x_b_next.shape[-1]):
        grad = torch.autograd.grad(x_b_next[0, idx], x_b_, retain_graph=True, allow_unused=False)[0]
        jac_rows.append(grad[0])
    return torch.stack(jac_rows, dim=0)
