"""Fallback optimiser: OpenAI-style natural evolution strategy (antithetic Gaussian perturbations,
rank-normalised fitness, Adam on the estimated gradient) over the same trainable parameter vector
PPO uses. Gradient-free, so it does not care whether BPTT through the substrate is stable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from flybrain.env.vec import VecFlyRunner


@dataclass
class ESConfig:
    population: int = 32          # antithetic pairs -> 2*population evaluations per generation
    sigma: float = 0.05
    lr: float = 0.02
    n_envs: int = 16              # episodes per candidate (each env = one episode, different seeds)
    max_decisions: int = 900      # 60 s per episode during training (cap at eval time is 180 s)
    generations: int = 1000
    minutes: float | None = None
    seed: int = 0


def _flat(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def _assign(params, vec: torch.Tensor) -> None:
    i = 0
    for p in params:
        n = p.numel(); p.data.copy_(vec[i:i + n].view_as(p)); i += n


@torch.no_grad()
def episode_return(policy, env: VecFlyRunner, seed: int, max_decisions: int, device: str) -> float:
    obs = torch.as_tensor(env.reset(seed), device=device); state = policy.initial_state(env.n)
    ret = np.zeros(env.n); alive = np.ones(env.n, dtype=bool)
    for _ in range(max_decisions):
        logits, _, state = policy(obs, state)
        act = logits.argmax(-1).cpu().numpy()
        nobs, rew, done, info = env.step(act)
        ret += rew * alive; alive &= ~done
        if not alive.any():
            break
        obs = torch.as_tensor(nobs, device=device)
        state = policy.mask_state(state, torch.as_tensor(~done, device=device)) if state is not None else None
    return float(ret.mean())


def train(policy, env: VecFlyRunner, cfg: ESConfig, device: str, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True); log = open(run_dir / "metrics.jsonl", "a")
    params = [p for p in policy.parameters() if p.requires_grad]
    theta = _flat(params); d = theta.numel()
    opt = torch.optim.Adam([torch.nn.Parameter(theta)], lr=cfg.lr); theta_p = opt.param_groups[0]["params"][0]
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    t0 = time.time(); last = {}
    for gen in range(cfg.generations):
        if cfg.minutes is not None and (time.time() - t0) / 60 > cfg.minutes:
            break
        eps = torch.randn(cfg.population, d, generator=g).to(device)
        fit = np.zeros(2 * cfg.population)
        seed = cfg.seed + gen * 100_000
        for i in range(cfg.population):
            for s, k in ((1, 2 * i), (-1, 2 * i + 1)):
                _assign(params, theta_p.data + s * cfg.sigma * eps[i])
                fit[k] = episode_return(policy, env, seed, cfg.max_decisions, device)
        ranks = np.empty_like(fit); ranks[np.argsort(fit)] = np.arange(len(fit)); u = ranks / (len(fit) - 1) - 0.5
        u = torch.as_tensor(u, device=device, dtype=torch.float32)
        grad = -(u[0::2] - u[1::2]).unsqueeze(1).mul(eps).sum(0) / (cfg.population * cfg.sigma)
        opt.zero_grad(); theta_p.grad = grad; opt.step()
        _assign(params, theta_p.data)
        centre = episode_return(policy, env, seed + 1, cfg.max_decisions, device)
        last = {"generation": gen, "minutes": round((time.time() - t0) / 60, 2), "fit_mean": float(fit.mean()), "fit_max": float(fit.max()), "centre_return": centre}
        log.write(json.dumps(last) + "\n"); log.flush()
        print(f"[{last['minutes']:6.1f}m] gen {gen:4d} fit mean {fit.mean():8.1f} max {fit.max():8.1f} centre {centre:8.1f}")
    torch.save({"policy": policy.state_dict(), "cfg": asdict(cfg)}, run_dir / "model.pt")
    return last
