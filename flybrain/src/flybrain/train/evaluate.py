"""Fixed evaluation protocol (design §8): held-out seeds, 180 s cap, greedy actions for learned
policies. Reports distance distribution and survival-to-cap rate; writes eval.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flybrain.env.vec import VecFlyRunner
from flybrain.env.agents import HeuristicAgent, RandomAgent

HELD_OUT_SEED = 10_000


def _scripted(agent_name: str, episodes: int, max_seconds: float) -> list[float]:
    env = VecFlyRunner(episodes, seed=HELD_OUT_SEED, max_seconds=max_seconds)
    obs = env.reset(HELD_OUT_SEED)
    agents = [HeuristicAgent() if agent_name == "heuristic" else RandomAgent(HELD_OUT_SEED + i) for i in range(episodes)]
    dist = np.full(episodes, np.nan); alive = np.ones(episodes, dtype=bool)
    for _ in range(env.max_decisions):
        act = [ag.act(obs[i]) for i, ag in enumerate(agents)]
        obs, _, done, info = env.step(act)
        for j, i in enumerate(info["fin_idx"]):
            if alive[i]:
                dist[i] = info["fin_distance"][j]; alive[i] = False
        if not alive.any():
            break
    return dist.tolist()


@torch.no_grad()
def _policy(policy, episodes: int, max_seconds: float, device: str, greedy: bool = True) -> list[float]:
    env = VecFlyRunner(episodes, seed=HELD_OUT_SEED, max_seconds=max_seconds)
    obs = torch.as_tensor(env.reset(HELD_OUT_SEED), device=device); state = policy.initial_state(episodes)
    dist = np.full(episodes, np.nan); alive = np.ones(episodes, dtype=bool)
    policy.eval()
    for _ in range(env.max_decisions):
        logits, _, state = policy(obs, state)
        act = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        nobs, _, done, info = env.step(act.cpu().numpy())
        for j, i in enumerate(info["fin_idx"]):
            if alive[i]:
                dist[i] = info["fin_distance"][j]; alive[i] = False
        if not alive.any():
            break
        obs = torch.as_tensor(nobs, device=device)
        state = policy.mask_state(state, torch.as_tensor(~done, device=device)) if state is not None else None
    return dist.tolist()


def summarize(name: str, dists: list[float], max_seconds: float) -> dict:
    d = np.array(dists)
    # survived-to-cap = distance reached the cap; cap distance depends on speed ramp, so use the max achievable
    cap_dist = _cap_distance(max_seconds)
    return {"agent": name, "episodes": len(d), "distance_mean": float(d.mean()), "distance_median": float(np.median(d)),
            "distance_p10": float(np.percentile(d, 10)), "distance_p90": float(np.percentile(d, 90)),
            "survive_cap": float((d >= cap_dist - 1.0).mean()), "cap_distance": cap_dist, "max_seconds": max_seconds, "distances": dists}


def _cap_distance(max_seconds: float) -> float:
    from flybrain.env.engine import Runner
    r = Runner(1); r.obstacles.clear(); r.coins.clear()
    for _ in range(int(round(max_seconds / (1 / 60)))):
        r.step(); r.obstacles.clear(); r.coins.clear()
    return r.distance


def run(agent: str, episodes: int, max_seconds: float, device: str, run_dir: Path | None = None, policy=None, out: Path | None = None) -> dict:
    if agent in ("heuristic", "random"):
        dists = _scripted(agent, episodes, max_seconds)
    else:
        dists = _policy(policy, episodes, max_seconds, device)
    s = summarize(agent, dists, max_seconds)
    print(f"{agent:12s} n={s['episodes']}  dist mean {s['distance_mean']:7.1f}  median {s['distance_median']:7.1f}  p10 {s['distance_p10']:7.1f}  p90 {s['distance_p90']:7.1f}  survive-cap {s['survive_cap']:.2f}  (cap {s['cap_distance']:.0f} m)")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(s, indent=1))
    return s
