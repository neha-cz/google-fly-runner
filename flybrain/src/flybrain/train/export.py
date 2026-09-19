"""Export a trained run to a JSON bundle the browser runtime (game/src/agents/fly.ts, mlp.ts) can
execute without PyTorch. Includes a self-test block (reference states -> logits) so the JS
implementation is checked against this one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .run import load_run
from .policies import MLPPolicy, FlyPolicy


def _selftest(pol, device: str, n: int = 8, steps: int = 6) -> dict:
    """Run `steps` decisions on fixed pseudo-random states from a zero state; record logits after each."""
    g = torch.Generator().manual_seed(123)
    states = (torch.rand(steps, n, 36, generator=g) * 2 - 1).to(device)
    st = pol.initial_state(n); logits = []
    with torch.no_grad():
        for t in range(steps):
            lg, _, st = pol(states[t], st); logits.append(lg.cpu().numpy().tolist())
    return {"states": states.cpu().numpy().tolist(), "logits": logits}


def export(run_dir: Path, out: Path, device: str = "cpu") -> dict:
    pol, meta = load_run(run_dir, device)
    pol = pol.to("cpu").eval()
    bundle = {"name": run_dir.name, "meta": {k: v for k, v in meta.items() if k not in ("args",)}}
    if isinstance(pol, MLPPolicy):
        L = [m for m in pol.body if isinstance(m, torch.nn.Linear)]
        bundle.update({"kind": "mlp", "layers": [{"w": l.weight.detach().numpy().tolist(), "b": l.bias.detach().numpy().tolist()} for l in L],
                       "pi": {"w": pol.pi.weight.detach().numpy().tolist(), "b": pol.pi.bias.detach().numpy().tolist()}})
    elif isinstance(pol, FlyPolicy):
        sub, enc, ro = pol.agent.substrate, pol.agent.encoder, pol.agent.readout
        g = pol.agent.graph
        w = sub.edge_weights().detach().numpy()
        alpha = (sub.cfg.dt_ms / sub.tau_ms().detach())[sub.node_type].numpy()
        bias = sub.bias.detach()[sub.node_type].numpy()
        names = (g.nodes["pool"] if g.pooled else (g.nodes["type"] + "_" + g.nodes["side"] + "_" + g.nodes["bodyId"].astype(str))).tolist()
        if hasattr(ro, "body"):
            h = ro.body[0]
            readout = {"kind": "mlp", "h": {"w": h.weight.detach().numpy().tolist(), "b": h.bias.detach().numpy().tolist()},
                       "w": ro.policy.weight.detach().numpy().tolist(), "b": ro.policy.bias.detach().numpy().tolist()}
        elif hasattr(ro, "policy"):
            readout = {"kind": "linear", "w": ro.policy.weight.detach().numpy().tolist(), "b": ro.policy.bias.detach().numpy().tolist()}
        else:
            readout = {"kind": "fixed", "M": ro.M.numpy().tolist(), "gain": ro.gain.detach().numpy().tolist(), "b": ro.bias.detach().numpy().tolist()}
        norm = {"mean": ro.norm.mean.numpy().tolist(), "var": ro.norm.var.numpy().tolist()} if hasattr(ro, "norm") else None
        bundle.update({
            "kind": "fly", "dn_norm": norm, "dynamics": sub.cfg.dynamics, "substeps": sub.cfg.substeps, "input_gain": float(sub.input_gain),
            "lif_threshold": sub.cfg.lif_threshold, "n": sub.n,
            "nodes": {"name": names, "type": g.nodes["type"].tolist(), "group": g.nodes["group"].tolist(), "side": g.nodes["side"].tolist(),
                      "alpha": alpha.tolist(), "bias": bias.tolist()},
            "edges": {"pre": sub.edge_pre.numpy().tolist(), "post": sub.edge_post.numpy().tolist(), "w": w.tolist()},
            "encoder": {"idx": enc.idx.numpy().tolist(), "w": enc.lin.weight.detach().numpy().tolist(), "b": enc.lin.bias.detach().numpy().tolist()},
            "readout": {"idx": ro.idx.numpy().tolist(), **readout},
            "graph": {"version": g.manifest.get("version"), "dataset": g.manifest.get("dataset"), "pooled": g.pooled, "control": g.manifest.get("control", "none")},
        })
    else:
        raise TypeError(type(pol))
    bundle["selftest"] = _selftest(pol, "cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle))
    # keep a manifest next to the bundles so the demo UI can list them
    idx_path = out.parent / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    label = run_dir.name if bundle["kind"] == "mlp" else f"{run_dir.name} ({'pooled' if bundle['graph']['pooled'] else 'per-neuron'}, {bundle['graph']['control']})"
    idx = [e for e in idx if e["file"] != out.name] + [{"file": out.name, "name": run_dir.name, "kind": bundle["kind"], "label": label}]
    idx_path.write_text(json.dumps(idx, indent=1))
    print(f"exported {run_dir.name} ({bundle['kind']}) -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return bundle
