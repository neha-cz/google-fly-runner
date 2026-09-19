"""M3 acceptance benchmark: forward + backward through T decision windows of the real substrate."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from flybrain.pipeline.artifact import load_graph
from flybrain.model.agent import FlyAgent
from flybrain.model.substrate import SubstrateConfig


def run(graph_path: Path, batch: int = 256, T: int = 32, device: str | None = None, reps: int = 3) -> int:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device} batch={batch} T={T} (decision windows of {SubstrateConfig().substeps} sub-steps)")
    rows = []
    for pooled in (True, False):
        for dyn in ("rate", "lif"):
            g = load_graph(graph_path, pooled=pooled)
            agent = FlyAgent(g, SubstrateConfig(dynamics=dyn)).to(device)
            obs = torch.rand(T, batch, 36, device=device) * 2 - 1
            def step():
                v = agent.init_state(batch); loss = 0.0
                for t in range(T):
                    logits, value, v, _ = agent(obs[t], v)
                    loss = loss + logits.logsumexp(-1).mean() + value.mean()
                loss.backward()
                if device == "mps": torch.mps.synchronize()
            step(); t0 = time.perf_counter()
            for _ in range(reps): step()
            dt = (time.perf_counter() - t0) / reps
            rows.append((("pooled" if pooled else "per-neuron"), dyn, agent.substrate.n, dt))
            print(f"  {'pooled' if pooled else 'per-neuron':10s} {dyn:4s} n={agent.substrate.n:5d}  {dt*1000:7.0f} ms/rollout  {T/dt:8.0f} decision-steps/s (batch {batch})  {T*batch/dt:10,.0f} env-decisions/s")
    return 0
