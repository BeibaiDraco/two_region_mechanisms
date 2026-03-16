from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def _as_mask(values, size: int) -> torch.Tensor:
    if values is None:
        return torch.ones(size, dtype=torch.float32)
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() != size:
        raise ValueError(f"Expected mask of length {size}, got {tensor.numel()}")
    return tensor


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.u = nn.Parameter(torch.empty(out_features, rank))
        self.v = nn.Parameter(torch.empty(rank, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        return (x @ self.v.t()) @ self.u.t()

    @property
    def weight(self) -> torch.Tensor:
        return self.u @ self.v


class TwoRegionRNNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size_a: int = 64,
        hidden_size_b: int = 64,
        dt: float = 0.1,
        tau_a: float = 1.0,
        tau_b: float = 1.5,
        reciprocal: bool = False,
        noise_std: float = 0.0,
        input_mask_a=None,
        input_mask_b=None,
        context_input_indices=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size_a = hidden_size_a
        self.hidden_size_b = hidden_size_b
        self.dt = float(dt)
        self.tau_a = float(tau_a)
        self.tau_b = float(tau_b)
        self.alpha_a = self.dt / self.tau_a
        self.alpha_b = self.dt / self.tau_b
        self.reciprocal = reciprocal
        self.noise_std = float(noise_std)
        self.context_input_indices = list(context_input_indices or [])

        self.w_aa = nn.Linear(hidden_size_a, hidden_size_a, bias=False)
        self.w_bb = nn.Linear(hidden_size_b, hidden_size_b, bias=False)
        self.w_in_a = nn.Linear(input_dim, hidden_size_a, bias=False)
        self.w_in_b = nn.Linear(input_dim, hidden_size_b, bias=False)
        self.bias_a = nn.Parameter(torch.zeros(hidden_size_a))
        self.bias_b = nn.Parameter(torch.zeros(hidden_size_b))
        self.w_ab = nn.Linear(hidden_size_b, hidden_size_a, bias=False) if reciprocal else None
        self.w_out = nn.Linear(hidden_size_b, output_dim, bias=True)

        self.register_buffer("input_mask_a", _as_mask(input_mask_a, input_dim))
        self.register_buffer("input_mask_b", _as_mask(input_mask_b, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.w_aa.weight, gain=1.0)
        nn.init.orthogonal_(self.w_bb.weight, gain=1.0)
        nn.init.xavier_uniform_(self.w_in_a.weight)
        nn.init.xavier_uniform_(self.w_in_b.weight)
        if self.w_ab is not None:
            nn.init.xavier_uniform_(self.w_ab.weight)
        nn.init.xavier_uniform_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)
        nn.init.zeros_(self.bias_a)
        nn.init.zeros_(self.bias_b)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def init_state(self, batch_size: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        x_a = torch.zeros(batch_size, self.hidden_size_a, device=device)
        x_b = torch.zeros(batch_size, self.hidden_size_b, device=device)
        return x_a, x_b

    def readout_from_xb(self, x_b: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.activation(x_b))

    def readout_gradient(self, x_b: torch.Tensor, output_index: int = 0) -> torch.Tensor:
        # Scalar gradient of q = output[output_index] wrt x_b for q = W_out tanh(x_b) + b
        weight = self.w_out.weight[output_index].unsqueeze(0)  # [1, Hb]
        sech2 = 1.0 - torch.tanh(x_b) ** 2
        return weight * sech2

    def communication_weight_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.bias_b.device)
        for tensor in self.communication_parameters():
            penalty = penalty + tensor.pow(2).mean()
        return penalty

    def communication_parameters(self):
        raise NotImplementedError

    def compute_source_b(
        self,
        r_a: torch.Tensor,
        r_b: torch.Tensor,
        u_t: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _apply_intervention(
        self,
        source_b: torch.Tensor,
        x_b: torch.Tensor,
        step_idx: int,
        intervention: Optional[Dict],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        info: Dict[str, torch.Tensor] = {}
        if intervention is None:
            return source_b, info

        active_steps = intervention.get("active_steps")
        if active_steps is not None and step_idx not in active_steps:
            return source_b, info

        kind = intervention.get("type", "none")
        if kind == "decouple":
            info["projected"] = torch.zeros_like(source_b)
            return torch.zeros_like(source_b), info

        grad = self.readout_gradient(x_b)
        grad_norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = grad / grad_norm
        aligned = (source_b * direction).sum(dim=-1, keepdim=True) * direction

        if kind == "project_aligned":
            info["projected"] = aligned
            return aligned, info
        if kind == "project_orthogonal":
            orth = source_b - aligned
            info["projected"] = orth
            return orth, info

        return source_b, info

    def step(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        u_t: torch.Tensor,
        step_idx: int,
        intervention: Optional[Dict] = None,
        add_noise: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r_a = self.activation(x_a)
        r_b = self.activation(x_b)

        rec_a = self.w_aa(r_a)
        rec_b = self.w_bb(r_b)
        fb_a = self.w_ab(r_b) if self.w_ab is not None else torch.zeros_like(x_a)
        source_b, source_details = self.compute_source_b(r_a, r_b, u_t)
        source_b_after, intervention_info = self._apply_intervention(source_b, x_b, step_idx, intervention)

        input_a = self.w_in_a(u_t * self.input_mask_a)
        input_b = self.w_in_b(u_t * self.input_mask_b)

        dx_a = -x_a + rec_a + fb_a + input_a + self.bias_a
        dx_b = -x_b + rec_b + source_b_after + input_b + self.bias_b

        if add_noise and self.noise_std > 0.0:
            dx_a = dx_a + self.noise_std * torch.randn_like(dx_a)
            dx_b = dx_b + self.noise_std * torch.randn_like(dx_b)

        x_a_new = x_a + self.alpha_a * dx_a
        x_b_new = x_b + self.alpha_b * dx_b
        r_a_new = self.activation(x_a_new)
        r_b_new = self.activation(x_b_new)
        output = self.w_out(r_b_new)

        return {
            "x_a": x_a_new,
            "x_b": x_b_new,
            "r_a": r_a_new,
            "r_b": r_b_new,
            "output": output,
            "source_b_pre": source_b,
            "source_b_post": source_b_after,
            "rec_b": rec_b,
            **source_details,
            **intervention_info,
        }

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        intervention: Optional[Dict] = None,
        add_noise: bool = False,
    ) -> Dict[str, torch.Tensor]:
        seq_len, batch_size, _ = inputs.shape
        if initial_state is None:
            x_a, x_b = self.init_state(batch_size=batch_size, device=inputs.device)
        else:
            x_a, x_b = initial_state

        xs_a, xs_b, rs_a, rs_b, ys = [], [], [], [], []
        source_pre, source_post = [], []
        extra_keys = {}

        for t in range(seq_len):
            step_out = self.step(x_a, x_b, inputs[t], step_idx=t, intervention=intervention, add_noise=add_noise)
            x_a, x_b = step_out["x_a"], step_out["x_b"]
            xs_a.append(x_a)
            xs_b.append(x_b)
            rs_a.append(step_out["r_a"])
            rs_b.append(step_out["r_b"])
            ys.append(step_out["output"])
            source_pre.append(step_out["source_b_pre"])
            source_post.append(step_out["source_b_post"])
            for key, value in step_out.items():
                if key in {"x_a", "x_b", "r_a", "r_b", "output", "source_b_pre", "source_b_post"}:
                    continue
                extra_keys.setdefault(key, []).append(value)

        out = {
            "x_a": torch.stack(xs_a, dim=0),
            "x_b": torch.stack(xs_b, dim=0),
            "r_a": torch.stack(rs_a, dim=0),
            "r_b": torch.stack(rs_b, dim=0),
            "outputs": torch.stack(ys, dim=0),
            "source_b_pre": torch.stack(source_pre, dim=0),
            "source_b_post": torch.stack(source_post, dim=0),
        }
        for key, values in extra_keys.items():
            out[key] = torch.stack(values, dim=0)
        return out


class TwoRegionAdditiveRNN(TwoRegionRNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_ba = nn.Linear(self.hidden_size_a, self.hidden_size_b, bias=False)
        nn.init.xavier_uniform_(self.w_ba.weight)

    def communication_parameters(self):
        return [self.w_ba.weight]

    def compute_source_b(self, r_a: torch.Tensor, r_b: torch.Tensor, u_t: torch.Tensor):
        source = self.w_ba(r_a)
        return source, {"source_additive": source}


class TwoRegionGatedRNN(TwoRegionRNNBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_ba = nn.Linear(self.hidden_size_a, self.hidden_size_b, bias=False)
        self.w_gate = nn.Linear(self.hidden_size_a, self.hidden_size_b, bias=True)
        self.w_mod = nn.Linear(self.hidden_size_b, self.hidden_size_b, bias=False)
        nn.init.xavier_uniform_(self.w_ba.weight)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.zeros_(self.w_gate.bias)
        nn.init.xavier_uniform_(self.w_mod.weight)

    def communication_parameters(self):
        return [self.w_ba.weight, self.w_gate.weight, self.w_mod.weight]

    def compute_source_b(self, r_a: torch.Tensor, r_b: torch.Tensor, u_t: torch.Tensor):
        additive = self.w_ba(r_a)
        gate = torch.sigmoid(self.w_gate(r_a))
        modulation = gate * self.w_mod(r_b)
        source = additive + modulation
        return source, {
            "source_additive": additive,
            "source_gate": gate,
            "source_modulation": modulation,
        }


class TwoRegionLowRankCommRNN(TwoRegionRNNBase):
    def __init__(self, rank: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.w_ba = LowRankLinear(self.hidden_size_a, self.hidden_size_b, rank=rank)

    def communication_parameters(self):
        return [self.w_ba.u, self.w_ba.v]

    def compute_source_b(self, r_a: torch.Tensor, r_b: torch.Tensor, u_t: torch.Tensor):
        source = self.w_ba(r_a)
        return source, {"source_additive": source}


class TwoRegionReciprocalContextRNN(TwoRegionRNNBase):
    def __init__(self, context_gain: float = 1.0, **kwargs):
        kwargs.setdefault("reciprocal", True)
        kwargs.setdefault("tau_b", 2.0)
        super().__init__(**kwargs)
        self.context_gain = float(context_gain)
        self.w_ba = nn.Linear(self.hidden_size_a, self.hidden_size_b, bias=False)
        self.w_gate = nn.Linear(self.hidden_size_a, self.hidden_size_b, bias=True)
        self.w_mod = nn.Linear(self.hidden_size_b, self.hidden_size_b, bias=False)
        self.w_context_u = nn.Linear(self.input_dim, self.hidden_size_b, bias=False)
        nn.init.xavier_uniform_(self.w_ba.weight)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.zeros_(self.w_gate.bias)
        nn.init.orthogonal_(self.w_mod.weight, gain=1.2)
        nn.init.xavier_uniform_(self.w_context_u.weight)
        nn.init.orthogonal_(self.w_bb.weight, gain=1.15)

    def communication_parameters(self):
        return [self.w_ba.weight, self.w_gate.weight, self.w_mod.weight, self.w_context_u.weight]

    def compute_source_b(self, r_a: torch.Tensor, r_b: torch.Tensor, u_t: torch.Tensor):
        additive = self.w_ba(r_a)
        ctx_mask = torch.zeros_like(u_t)
        if self.context_input_indices:
            ctx_mask[:, self.context_input_indices] = 1.0
        else:
            ctx_mask = torch.ones_like(u_t)
        context_drive = self.w_context_u(u_t * ctx_mask)
        gate = torch.sigmoid(self.w_gate(r_a) + self.context_gain * context_drive)
        modulation = gate * self.w_mod(r_b)
        source = additive + modulation
        return source, {
            "source_additive": additive,
            "source_gate": gate,
            "source_modulation": modulation,
            "source_context_drive": context_drive,
        }
