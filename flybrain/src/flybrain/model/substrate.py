"""Connectome-constrained substrate.

Rate form (flyvis, Lappalainen et al. 2024):
    tau_i dV_i/dt = -V_i + b_i + sum_j W_ij r_j + x_i,     r = relu(V)
    W_ij = sign_j * scale[class(j->i)] * syn_ij / S
where the edge set, sign (from neurotransmitter prediction) and synapse counts syn_ij are FROZEN
to the Phase-2 graph, S is a global normaliser (mean total input synapses per node), and the free
parameters are: per-cell-type log tau and bias, and per synapse-class scale (class = (pre type,
post type)). Signed classes use |scale| so the sign can never flip; unsigned classes (modulatory
NTs) have a free-signed scale. No edge is ever added.

LIF form (flag `dynamics="lif"`): same input current, membrane v with threshold 1, reset to 0,
Heaviside spike with a fast-sigmoid surrogate gradient; the "rate" reported is the spike fraction
over the sub-steps of one decision window.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor

from flybrain.pipeline.artifact import Graph


@dataclass
class SubstrateConfig:
    dt_ms: float = 5.0
    substeps: int = 13          # 13 * 5 ms ≈ one 15 Hz decision window
    tau_init_ms: float = 20.0
    tau_min_ms: float = 5.0
    tau_max_ms: float = 200.0
    scale_init: float = 1.0
    input_gain: float = 1.0     # calibrated; 0 = "silenced" control
    dynamics: str = "rate"      # "rate" | "lif"
    trainable: bool = True      # False = readout-only training (substrate frozen)
    lif_threshold: float = 1.0
    matmul_bf16: bool = True    # cast only the recurrent matmul operands to bf16 (state stays fp32): 5-6x faster on MPS


class SpikeFn(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward (Zenke & Ganguli 2018)."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2


class ConnectomeSubstrate(nn.Module):
    def __init__(self, graph: Graph, cfg: SubstrateConfig | None = None):
        super().__init__()
        self.cfg = cfg or SubstrateConfig()
        self.graph = graph
        nodes, edges = graph.nodes, graph.edges
        self.n = len(nodes)
        types = pd.Categorical(nodes["type"])
        self.type_names: list[str] = list(types.categories)
        self.register_buffer("node_type", torch.as_tensor(types.codes.astype(np.int64).copy()))
        # synapse classes = (pre type, post type)
        pre_t, post_t = types.codes[edges["pre"].to_numpy()], types.codes[edges["post"].to_numpy()]
        cls_key = pre_t.astype(np.int64) * len(self.type_names) + post_t
        cls_codes, cls_uniq = pd.factorize(cls_key)
        self.n_classes = len(cls_uniq)
        self.class_names = [f"{self.type_names[k // len(self.type_names)]}->{self.type_names[k % len(self.type_names)]}" for k in cls_uniq]
        self.register_buffer("edge_class", torch.as_tensor(cls_codes.astype(np.int64)))
        self.register_buffer("edge_pre", torch.as_tensor(edges["pre"].to_numpy(dtype=np.int64).copy()))
        self.register_buffer("edge_post", torch.as_tensor(edges["post"].to_numpy(dtype=np.int64).copy()))
        syn_col = "syn_per_post" if graph.pooled else "syn"
        syn = edges[syn_col].to_numpy(dtype=np.float32)
        in_total = np.bincount(edges["post"].to_numpy(), weights=syn, minlength=self.n)
        S = float(in_total[in_total > 0].mean())
        self.syn_norm = S
        self.register_buffer("edge_syn", torch.as_tensor(syn / S))
        sign = edges["sign"].to_numpy(dtype=np.float32)
        self.register_buffer("edge_sign", torch.as_tensor(sign))
        # a class is "signed" if every edge in it carries a nonzero sign
        cls_signed = np.ones(self.n_classes, dtype=bool)
        np.logical_and.at(cls_signed, cls_codes, sign != 0)
        self.register_buffer("class_signed", torch.as_tensor(cls_signed))
        # free parameters
        self.log_tau = nn.Parameter(torch.full((len(self.type_names),), float(np.log(self.cfg.tau_init_ms))))
        self.bias = nn.Parameter(torch.zeros(len(self.type_names)))
        self.scale = nn.Parameter(torch.full((self.n_classes,), float(self.cfg.scale_init)))
        for p in self.parameters():
            p.requires_grad_(self.cfg.trainable)
        self.input_gain = self.cfg.input_gain

    # ---- parameter views ----------------------------------------------------
    def tau_ms(self) -> Tensor:
        return self.log_tau.exp().clamp(self.cfg.tau_min_ms, self.cfg.tau_max_ms)

    def edge_weights(self) -> Tensor:
        s = self.scale[self.edge_class]
        signed = self.class_signed[self.edge_class]
        eff = torch.where(signed, s.abs() * self.edge_sign, s)
        return eff * self.edge_syn

    def weight_matrix(self) -> Tensor:
        """Dense (post x pre). Rebuilt each call so gradients reach scale."""
        W = self.edge_weights().new_zeros(self.n, self.n)
        return W.index_put((self.edge_post, self.edge_pre), self.edge_weights())

    def init_state(self, batch: int, device=None) -> Tensor:
        return torch.zeros(batch, self.n, device=device or self.log_tau.device)

    # ---- dynamics -----------------------------------------------------------
    def forward(self, v: Tensor, x: Tensor, substeps: int | None = None) -> tuple[Tensor, Tensor]:
        """Integrate one decision window. x: external drive [B, n]. Returns (v, mean rate over window)."""
        T = substeps or self.cfg.substeps
        W = self.weight_matrix()
        half = self.cfg.matmul_bf16 and v.device.type != "cpu"
        Wm = W.to(torch.bfloat16) if half else W
        def recur(r: Tensor) -> Tensor:
            return (r.to(torch.bfloat16) @ Wm.T).float() if half else r @ W.T
        alpha = (self.cfg.dt_ms / self.tau_ms())[self.node_type]          # [n]
        b = self.bias[self.node_type]
        drive = self.input_gain * x
        rate_acc = torch.zeros_like(v)
        if self.cfg.dynamics == "rate":
            for _ in range(T):
                r = torch.relu(v)
                v = v + alpha * (-v + b + recur(r) + drive)
                rate_acc = rate_acc + r
            return v, rate_acc / T
        elif self.cfg.dynamics == "lif":
            for _ in range(T):
                s = SpikeFn.apply(v - self.cfg.lif_threshold)
                v = v * (1 - s)                                              # reset
                v = v + alpha * (-v + b + recur(s) + drive)
                rate_acc = rate_acc + s
            return v, rate_acc / T
        raise ValueError(self.cfg.dynamics)

    def describe(self) -> dict:
        return {"n_nodes": self.n, "n_edges": int(self.edge_pre.numel()), "n_types": len(self.type_names),
                "n_classes": self.n_classes, "n_signed_classes": int(self.class_signed.sum()), "syn_norm": self.syn_norm,
                "free_params": sum(p.numel() for p in self.parameters()), "cfg": asdict(self.cfg)}
