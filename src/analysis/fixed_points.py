from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def find_slow_point(
    model,
    x_a: torch.Tensor,
    x_b_init: torch.Tensor,
    u_t: torch.Tensor,
    steps: int = 200,
    lr: float = 0.1,
    decoupled: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Lightweight slow-point search for the B dynamics under a fixed input.
    Intended as a convenience diagnostic, not a production fixed-point solver.
    """
    x_b = x_b_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_b], lr=lr)
    intervention = {"type": "decouple"} if decoupled else None

    for _ in range(steps):
        optimizer.zero_grad()
        step = model.step(x_a.detach(), x_b, u_t.detach(), step_idx=0, intervention=intervention, add_noise=False)
        residual = ((step["x_b"] - x_b) ** 2).mean()
        residual.backward()
        optimizer.step()

    with torch.no_grad():
        step = model.step(x_a.detach(), x_b.detach(), u_t.detach(), step_idx=0, intervention=intervention, add_noise=False)
        residual = ((step["x_b"] - x_b.detach()) ** 2).mean()
    return {"x_b": x_b.detach(), "residual": residual.detach()}
