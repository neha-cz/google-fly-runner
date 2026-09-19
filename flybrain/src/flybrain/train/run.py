"""Assemble policy + env + optimiser from CLI flags (the training stages of design §6)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from flybrain.pipeline.artifact import load_graph, Graph
from flybrain.model.substrate import SubstrateConfig
from flybrain.model import controls
from flybrain.env.vec import VecFlyRunner
from .policies import MLPPolicy, FlyPolicy
from .ppo import PPO, PPOConfig
from .es import ESConfig, train as es_train


def build_graph(graph_dir: Path, pooled: bool, control: str, seed: int) -> Graph:
    g = load_graph(graph_dir, pooled=pooled)
    if control == "shuffled":
        g = controls.shuffled(g, seed=seed)
    elif control == "random":
        g = controls.random(g, seed=seed)
    return g


def calibrated_config(graph_dir: Path, pooled: bool, dynamics: str, stage: str, control: str) -> SubstrateConfig:
    tag = f"{'pool' if pooled else 'neuron'}_{dynamics}" + (f"_{control}" if control in ("shuffled", "random") else "")
    f = graph_dir / f"calibration_{tag}.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run `flybrain calibrate` for this graph/dynamics/control first")
    gain, scale = 1.0, 1.0
    if f.exists():
        chosen = json.loads(f.read_text()).get("chosen")
        if chosen:
            gain, scale = chosen["gain"], chosen["scale"]
    if control == "silenced":
        gain = 0.0
    return SubstrateConfig(input_gain=gain, scale_init=scale, dynamics=dynamics, trainable=(stage == "full"))


def apply_integration(cfg: SubstrateConfig, args) -> SubstrateConfig:
    """Optional coarser integration (e.g. --dt-ms 10 --substeps 7): 1.6x faster per decision on MPS
    at the cost of a 10 ms minimum time constant."""
    dt, sub = getattr(args, "dt_ms", None), getattr(args, "substeps", None)
    if dt: cfg.dt_ms = dt; cfg.tau_min_ms = max(cfg.tau_min_ms, dt)
    if sub: cfg.substeps = sub
    return cfg


def build_policy(args, device: str):
    if args.agent == "mlp":
        return MLPPolicy(noop_bias=args.noop_bias).to(device), {"agent": "mlp"}
    graph_dir = Path(args.graph_dir); pooled = args.graph == "pooled"
    g = build_graph(graph_dir, pooled, args.control, args.seed)
    cfg = apply_integration(calibrated_config(graph_dir, pooled, args.dynamics, args.stage, args.control), args)
    pol = FlyPolicy(g, cfg, encoder="state", readout=args.readout, noop_bias=args.noop_bias).to(device)
    meta = {"agent": "fly", "graph": args.graph, "control": args.control, "stage": args.stage, "dynamics": args.dynamics,
            "readout": args.readout, "substrate": pol.agent.substrate.describe(), "params": pol.agent.param_groups()}
    return pol, meta


def train(args) -> dict:
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(args.runs) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    pol, meta = build_policy(args, device)
    if getattr(args, "init", None):                        # continue from an earlier run's weights
        ck = torch.load(Path(args.init) / "model.pt", map_location=device)
        pol.load_state_dict(ck["policy"]); meta["init"] = args.init
    meta.update({"algo": args.algo, "device": device, "args": vars(args)})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"run {run_dir}: {json.dumps({k: v for k, v in meta.items() if k not in ('args', 'substrate')})}")
    if args.algo == "ppo":
        cfg = PPOConfig(n_envs=args.envs, rollout=args.rollout, total_steps=args.steps, minutes=args.minutes, seed=args.seed,
                        lr=args.lr, scale_max=args.scale_max, ent_coef=args.ent_coef, epochs=args.epochs, minibatches=args.minibatches)
        env = VecFlyRunner(cfg.n_envs, seed=args.seed)
        return PPO(pol, env, cfg, device, run_dir).train()
    cfg = ESConfig(n_envs=args.envs, generations=args.steps, minutes=args.minutes, seed=args.seed, lr=args.lr)
    env = VecFlyRunner(cfg.n_envs, seed=args.seed)
    return es_train(pol, env, cfg, device, run_dir)


def load_run(run_dir: Path, device: str):
    meta = json.loads((run_dir / "run.json").read_text())
    class A: pass
    a = A(); a.noop_bias = 0.0; a.__dict__.update(meta["args"])   # loaded weights override the init prior anyway
    pol, _ = build_policy(a, device)
    ck = torch.load(run_dir / "model.pt", map_location=device)
    pol.load_state_dict(ck["policy"], strict=False); pol.eval()   # older runs lack the DNNorm buffers
    return pol, meta
