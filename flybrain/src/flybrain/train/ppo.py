"""PPO with GAE for both stateless (MLP) and recurrent (connectome) policies.

Recurrent policies are trained with truncated BPTT over each rollout of T decision steps: the
substrate state at the start of the rollout is stored (detached) and the sequence is re-run from
it during the update, with the state reset at episode boundaries. Minibatches are over
environments (whole sequences) for recurrent policies and over flat samples for the MLP.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from flybrain.env.vec import VecFlyRunner


@dataclass
class PPOConfig:
    n_envs: int = 256
    rollout: int = 32
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_steps: int = 2_000_000       # env decisions
    minutes: float | None = None       # wall-clock budget, whichever comes first
    seed: int = 0
    scale_max: float | None = None     # FlyPolicy stability clamp
    log_every: int = 1
    norm_rewards: bool = True          # divide rewards by a running std of discounted returns (see RewardNormalizer)


class RewardNormalizer:
    """Scales rewards by the running std of the discounted return (SB3 VecNormalize style).
    Without it the value loss (~100 with a -50 crash penalty) dominates the shared-body gradient
    and global grad-norm clipping starves the policy gradient: the first runs showed
    approx_kl ~1e-4 and clipfrac 0 at every update."""
    def __init__(self, n: int, gamma: float, device):
        self.ret = torch.zeros(n, device=device); self.gamma = gamma
        self.count = 1e-4; self.mean = 0.0; self.var = 1.0

    def __call__(self, rew: Tensor, done: Tensor) -> Tensor:
        self.ret = self.ret * self.gamma * (1 - done) + rew
        b = self.ret; bc = b.numel(); bm = b.mean().item(); bv = b.var(unbiased=False).item()
        delta = bm - self.mean; tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot
        return rew / (self.var ** 0.5 + 1e-8)


class PPO:
    def __init__(self, policy, env: VecFlyRunner, cfg: PPOConfig, device: str, run_dir: Path):
        self.policy, self.env, self.cfg, self.device, self.run_dir = policy, env, cfg, device, run_dir
        self.opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr, eps=1e-5)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(run_dir / "metrics.jsonl", "a")
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    def train(self) -> dict:
        cfg, env, pol, dev = self.cfg, self.env, self.policy, self.device
        N, T = cfg.n_envs, cfg.rollout
        obs = torch.as_tensor(env.reset(cfg.seed), device=dev)
        state = pol.initial_state(N)
        steps = 0; update = 0; t_start = time.time()
        recent_dist, recent_ret, recent_cap = [], [], []
        last = {}
        rnorm = RewardNormalizer(N, cfg.gamma, dev) if cfg.norm_rewards else None
        while steps < cfg.total_steps and (cfg.minutes is None or (time.time() - t_start) / 60 < cfg.minutes):
            # ---- rollout ----
            B_obs = torch.zeros(T, N, obs.shape[1], device=dev); B_act = torch.zeros(T, N, dtype=torch.long, device=dev)
            B_logp = torch.zeros(T, N, device=dev); B_val = torch.zeros(T + 1, N, device=dev)
            B_rew = torch.zeros(T, N, device=dev); B_done = torch.zeros(T, N, device=dev); B_boot = torch.zeros(T, N, device=dev)
            state0 = state.detach().clone() if state is not None else None
            pol.eval()
            with torch.no_grad():
                for t in range(T):
                    logits, value, new_state = pol(obs, state)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                    B_obs[t] = obs; B_act[t] = act; B_logp[t] = dist.log_prob(act); B_val[t] = value
                    nobs, rew, done, info = env.step(act.cpu().numpy())
                    r_t = torch.as_tensor(rew, device=dev, dtype=torch.float32)
                    d = torch.as_tensor(done, device=dev, dtype=torch.float32); B_done[t] = d
                    B_rew[t] = rnorm(r_t, d) if rnorm is not None else r_t
                    if info["final_obs"] is not None and info["truncated"].any():
                        # bootstrap truncated episodes with V(final obs)
                        fo = torch.as_tensor(info["final_obs"], device=dev)
                        _, fval, _ = pol(fo, new_state)
                        B_boot[t] = torch.as_tensor(info["truncated"], device=dev, dtype=torch.float32) * fval
                    if len(info["fin_idx"]):
                        recent_dist += info["fin_distance"].tolist(); recent_ret += info["fin_return"].tolist()
                        recent_cap += info["truncated"][info["fin_idx"]].tolist()
                    obs = torch.as_tensor(nobs, device=dev)
                    state = pol.mask_state(new_state, 1 - d) if new_state is not None else None
                _, B_val[T], _ = pol(obs, state)
            steps += T * N
            # ---- GAE ----
            adv = torch.zeros(T, N, device=dev); gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nonterm = 1 - B_done[t]
                next_v = B_val[t + 1] * nonterm + B_boot[t]
                delta = B_rew[t] + cfg.gamma * next_v - B_val[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + B_val[:T]
            # ---- update ----
            pol.train()
            stats = self._update(B_obs, B_act, B_logp, B_val[:T], adv, ret, B_done, state0)
            update += 1
            if update % cfg.log_every == 0:
                el = time.time() - t_start
                rd = np.array(recent_dist[-500:]) if recent_dist else np.array([np.nan])
                last = {"update": update, "steps": steps, "minutes": round(el / 60, 2), "fps": int(steps / el),
                        "episodes": env.episodes_done, "dist_mean": float(np.nanmean(rd)), "dist_median": float(np.nanmedian(rd)),
                        "ret_mean": float(np.mean(recent_ret[-500:])) if recent_ret else float("nan"),
                        "survive_cap": float(np.mean(recent_cap[-500:])) if recent_cap else float("nan"),
                        "reward_scale": float(rnorm.var ** 0.5) if rnorm is not None else 1.0, **stats}
                self.log.write(json.dumps(last) + "\n"); self.log.flush()
                print(f"[{last['minutes']:6.1f}m] upd {update:4d} steps {steps:>9,} fps {last['fps']:5d} eps {last['episodes']:6d} "
                      f"dist {last['dist_mean']:7.1f} (med {last['dist_median']:6.1f}) cap {last['survive_cap']:.2f} "
                      f"pl {stats['policy_loss']:.3f} vl {stats['value_loss']:.2f} ent {stats['entropy']:.2f}")
        torch.save({"policy": pol.state_dict(), "cfg": asdict(cfg)}, self.run_dir / "model.pt")
        return last

    def _update(self, obs, act, logp_old, val_old, adv, ret, done, state0) -> dict:
        cfg, pol = self.cfg, self.policy
        T, N = act.shape
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        tot = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}; cnt = 0
        for _ in range(cfg.epochs):
            if pol.recurrent:
                perm = torch.randperm(N, device=obs.device)
                for mb in perm.chunk(cfg.minibatches):
                    st = state0[mb]
                    logits_l, val_l = [], []
                    for t in range(T):
                        lg, v, st = pol(obs[t, mb], st)
                        logits_l.append(lg); val_l.append(v)
                        st = pol.mask_state(st, 1 - done[t, mb])
                    logits = torch.stack(logits_l); value = torch.stack(val_l)
                    s = self._loss(logits, value, act[:, mb], logp_old[:, mb], val_old[:, mb], adv_n[:, mb], ret[:, mb])
                    for k in tot: tot[k] += s[k]
                    cnt += 1
            else:
                flat = lambda x: x.reshape(T * N, *x.shape[2:])
                fo, fa, fl, fv, fad, fr = flat(obs), flat(act), flat(logp_old), flat(val_old), flat(adv_n), flat(ret)
                perm = torch.randperm(T * N, device=obs.device)
                for mb in perm.chunk(cfg.minibatches):
                    logits, value, _ = pol(fo[mb], None)
                    s = self._loss(logits, value, fa[mb], fl[mb], fv[mb], fad[mb], fr[mb])
                    for k in tot: tot[k] += s[k]
                    cnt += 1
        return {k: v / max(cnt, 1) for k, v in tot.items()}

    def _loss(self, logits, value, act, logp_old, val_old, adv, ret) -> dict:
        cfg = self.cfg
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act); ratio = (logp - logp_old).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        v_clipped = val_old + (value - val_old).clamp(-cfg.clip, cfg.clip)
        vl = 0.5 * torch.max((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
        ent = dist.entropy().mean()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        self.opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad], cfg.max_grad_norm)
        self.opt.step()
        if hasattr(self.policy, "clamp"):
            self.policy.clamp(cfg.scale_max)
        with torch.no_grad():
            kl = (logp_old - logp).mean().item(); cf = ((ratio - 1).abs() > cfg.clip).float().mean().item()
        return {"policy_loss": pg.item(), "value_loss": vl.item(), "entropy": ent.item(), "approx_kl": kl, "clipfrac": cf}
