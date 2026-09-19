"""Calibrate the substrate before RL (fly-craftax lesson: PPO gets no gradient from a dead or
saturated network). Sweeps input gain x synapse-scale init on real game states from
data/cache (dumped by `npm run dump-states`) with a random-ish policy and reports, per setting:
active fraction (nodes with mean rate > eps), rate blow-up, and DN activity spread. Picks the
setting whose active fraction is in [lo, hi] with finite, bounded rates and the largest DN
variance, and writes calibration.json next to the graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flybrain.pipeline.artifact import load_graph, Graph
from .substrate import ConnectomeSubstrate, SubstrateConfig
from .io import StateEncoder, dn_indices, STATE_SIZE


def load_states(cache: Path) -> np.ndarray:
    meta = json.loads((cache / "meta.json").read_text())
    return np.fromfile(cache / "states.f32", dtype=np.float32).reshape(meta["steps"], meta["stateSize"])


@torch.no_grad()
def evaluate(graph: Graph, states: np.ndarray, gain: float, scale: float, dynamics: str, device: str,
             batch: int = 64, T: int = 30, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = SubstrateConfig(input_gain=gain, scale_init=scale, dynamics=dynamics, trainable=False)
    sub = ConnectomeSubstrate(graph, cfg).to(device); enc = StateEncoder(graph).to(device)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(states) - T, batch)
    v = sub.init_state(batch, device)
    rates_sum = torch.zeros(batch, sub.n, device=device)
    for t in range(T):
        obs = torch.as_tensor(states[starts + t], device=device)
        v, r = sub(v, enc(obs))
        rates_sum += r
    mean_rate = (rates_sum / T)                                  # [B, n]
    node_rate = mean_rate.mean(0)
    dn = mean_rate[:, torch.as_tensor(dn_indices(graph), device=device)]
    finite = bool(torch.isfinite(mean_rate).all())
    return {"gain": gain, "scale": scale,
            "active_frac": float((node_rate > 1e-3).float().mean()) if finite else 0.0,
            "mean_rate": float(node_rate.mean()) if finite else float("inf"),
            "max_rate": float(node_rate.max()) if finite else float("inf"),
            "dn_mean": float(dn.mean()) if finite else float("inf"),
            "dn_active_frac": float((dn.mean(0) > 1e-3).float().mean()) if finite else 0.0,
            "dn_std_over_batch": float(dn.std(0).mean()) if finite else 0.0, "finite": finite}


def run(graph_path: Path, cache: Path, dynamics: str = "rate", pooled: bool = True, device: str | None = None,
        lo: float = 0.05, hi: float = 0.5, max_rate: float = 50.0, control: str = "none", seed: int = 0) -> dict:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    graph = load_graph(graph_path, pooled=pooled)
    if control in ("shuffled", "random"):                 # controls have their own dynamics -> their own calibration
        from . import controls as _c
        graph = getattr(_c, control)(graph, seed=seed)
    states = load_states(cache)
    results = []
    # LIF needs far larger scales: one presynaptic spike delivers ~alpha*scale*syn/S against a threshold of 1
    scales = [3.0, 10.0, 30.0, 100.0, 300.0] if dynamics == "lif" else [0.3, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for scale in scales:
        for gain in [0.1, 0.3, 1.0, 3.0, 10.0]:
            r = evaluate(graph, states, gain, scale, dynamics, device); results.append(r)
            if not r["finite"]:
                print(f"  scale={scale:<4} gain={gain:<5} DIVERGED"); continue
            print(f"  scale={scale:<4} gain={gain:<5} active={r['active_frac']:.2f} mean={r['mean_rate']:.3g} max={r['max_rate']:.3g} dn_active={r['dn_active_frac']:.2f} dn_std={r['dn_std_over_batch']:.3g}")
    ok = [r for r in results if r["finite"] and lo <= r["active_frac"] <= hi and r["max_rate"] <= max_rate and r["dn_active_frac"] > 0]
    best = max(ok, key=lambda r: r["dn_std_over_batch"]) if ok else None
    out = {"graph": str(graph_path), "pooled": pooled, "dynamics": dynamics, "device": device, "criteria": {"active_frac": [lo, hi], "max_rate": max_rate},
           "results": results, "chosen": best}
    tag = f"{'pool' if pooled else 'neuron'}_{dynamics}" + (f"_{control}" if control in ("shuffled", "random") else "")
    (Path(graph_path) / f"calibration_{tag}.json").write_text(json.dumps(out, indent=2))
    print("chosen:", best)
    return out
