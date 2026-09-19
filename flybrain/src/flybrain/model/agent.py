"""FlyAgent = encoder -> connectome substrate -> DN readout. One recurrent policy/value network."""
from __future__ import annotations

import torch
from torch import nn, Tensor

from flybrain.pipeline.artifact import Graph
from .substrate import ConnectomeSubstrate, SubstrateConfig
from .io import StateEncoder, FrameEncoder, LinearReadout, FixedReadout, MLPReadout


class FlyAgent(nn.Module):
    def __init__(self, graph: Graph, cfg: SubstrateConfig | None = None, encoder: str = "state", readout: str = "linear"):
        super().__init__()
        self.graph = graph
        self.substrate = ConnectomeSubstrate(graph, cfg)
        self.encoder = StateEncoder(graph) if encoder == "state" else FrameEncoder(graph)
        self.readout = {"linear": LinearReadout, "fixed": FixedReadout, "mlp": MLPReadout}[readout](graph)

    def init_state(self, batch: int) -> Tensor:
        return self.substrate.init_state(batch)

    def forward(self, obs: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns (action logits, value, new substrate state, mean rates)."""
        x = self.encoder(obs)
        v, rates = self.substrate(v, x)
        logits, value = self.readout(rates)
        return logits, value, v, rates

    def param_groups(self) -> dict[str, int]:
        return {name: sum(p.numel() for p in m.parameters() if p.requires_grad) for name, m in
                (("encoder", self.encoder), ("substrate", self.substrate), ("readout", self.readout))}
