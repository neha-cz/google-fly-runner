"""Policies share one interface so the PPO loop is agnostic:
    initial_state(n) -> Tensor | None
    forward(obs [B, obs], state) -> (logits [B, A], value [B], new_state)
    mask_state(state, keep [B]) -> state with rows reset where keep == 0
"""
from __future__ import annotations

import torch
from torch import nn, Tensor

from flybrain.pipeline.artifact import Graph
from flybrain.model.agent import FlyAgent
from flybrain.model.substrate import SubstrateConfig


class MLPPolicy(nn.Module):
    """Small MLP baseline: same input (36 floats) and output (5 logits + value) shapes as the fly agent."""
    def __init__(self, obs_size: int = 36, hidden: int = 64, n_actions: int = 5, noop_bias: float = 3.0):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions); self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        # Action prior: the good policy is ~96% no-op with rare, precisely timed actions, and under
        # uniform exploration a correct jump is almost always cancelled by a later random slide
        # (fast-fall). Starting at ~83% no-op makes "act, then hold" sequences common enough to learn from.
        with torch.no_grad():
            self.pi.bias[0] = noop_bias
    recurrent = False

    def initial_state(self, n: int):
        return None

    def forward(self, obs: Tensor, state=None):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1), None

    def mask_state(self, state, keep):
        return None


class FlyPolicy(nn.Module):
    """Connectome-constrained agent behind the same interface; state = substrate membrane vector."""
    recurrent = True

    def __init__(self, graph: Graph, cfg: SubstrateConfig, encoder: str = "state", readout: str = "linear", noop_bias: float = 3.0):
        super().__init__()
        self.agent = FlyAgent(graph, cfg, encoder=encoder, readout=readout)
        self.last_rates: Tensor | None = None
        with torch.no_grad():                       # same action prior as MLPPolicy
            ro = self.agent.readout
            (ro.policy.bias if hasattr(ro, "policy") else ro.bias)[0] = noop_bias

    def initial_state(self, n: int) -> Tensor:
        return self.agent.init_state(n)

    def forward(self, obs: Tensor, state: Tensor):
        logits, value, v, rates = self.agent(obs, state)
        self.last_rates = rates
        return logits, value, v

    def mask_state(self, state: Tensor, keep: Tensor) -> Tensor:
        return state * keep.to(state.dtype).unsqueeze(-1)

    def clamp(self, scale_max: float | None):
        """Stability guard for Stage C (calibration showed divergence past scale ~2 on the per-neuron graph)."""
        if scale_max is not None:
            with torch.no_grad():
                self.agent.substrate.scale.clamp_(-scale_max, scale_max)
