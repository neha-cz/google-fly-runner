"""Vectorised FlyRunner environment (N independent Python engines, lock-step, auto-reset) and a
single-instance Gymnasium wrapper.

Decisions happen every `frame_skip` engine ticks (default 4 -> 15 Hz); the chosen action is latched
on the first tick. Reward per decision = metres advanced + 5 * coins - 50 * crash - 0.05 * lane
change. Episodes are truncated at `max_seconds` (180 s, the Fly Dino protocol) so "survived to
cap" is a clean success criterion; truncation is reported separately from crashes so the learner
can bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import Runner, STATE_SIZE, DT, LEFT, RIGHT, NOOP


@dataclass
class RewardConfig:
    per_meter: float = 1.0
    per_coin: float = 5.0
    crash: float = -50.0
    lane_change: float = -0.05


class VecFlyRunner:
    def __init__(self, n: int, seed: int = 0, frame_skip: int = 4, max_seconds: float = 180.0, reward: RewardConfig | None = None):
        self.n, self.frame_skip, self.max_seconds = n, frame_skip, max_seconds
        self.reward = reward or RewardConfig()
        self.max_decisions = int(round(max_seconds / DT / frame_skip))
        self._seed = seed
        self.runners = [Runner(seed + i) for i in range(n)]
        self.next_seed = seed + n
        self.obs = np.zeros((n, STATE_SIZE), dtype=np.float32)
        self.ep_len = np.zeros(n, dtype=np.int64); self.ep_ret = np.zeros(n)
        self.episodes_done = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed; self.next_seed = seed + self.n
            for i, r in enumerate(self.runners):
                r.reset(seed + i)
        for i, r in enumerate(self.runners):
            r.get_state(self.obs[i])
        self.ep_len[:] = 0; self.ep_ret[:] = 0
        return self.obs

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        rw = self.reward
        rewards = np.zeros(self.n); done = np.zeros(self.n, dtype=bool); truncated = np.zeros(self.n, dtype=bool)
        final_obs = None; fin_dist = []; fin_ret = []; fin_len = []; fin_idx = []
        for i, r in enumerate(self.runners):
            a = int(actions[i])
            rew = rw.lane_change if a in (LEFT, RIGHT) else 0.0
            r.act(a)
            crashed = False
            for _ in range(self.frame_skip):
                alive, c, coins, dz = r.step()
                rew += rw.per_meter * dz + rw.per_coin * coins
                if c:
                    crashed = True; rew += rw.crash; break
            self.ep_len[i] += 1; self.ep_ret[i] += rew
            rewards[i] = rew
            if crashed or self.ep_len[i] >= self.max_decisions:
                done[i] = True; truncated[i] = not crashed
                if final_obs is None:
                    final_obs = np.zeros_like(self.obs)
                r.get_state(final_obs[i])
                fin_idx.append(i); fin_dist.append(r.distance); fin_ret.append(self.ep_ret[i]); fin_len.append(self.ep_len[i])
                r.reset(self.next_seed); self.next_seed += 1
                self.ep_len[i] = 0; self.ep_ret[i] = 0; self.episodes_done += 1
            r.get_state(self.obs[i])
        info = {"truncated": truncated, "final_obs": final_obs, "fin_idx": np.array(fin_idx, dtype=np.int64),
                "fin_distance": np.array(fin_dist), "fin_return": np.array(fin_ret), "fin_length": np.array(fin_len)}
        return self.obs, rewards, done, info


try:
    import gymnasium as gym

    class FlyRunnerEnv(gym.Env):
        """Single-instance Gymnasium env over the same engine (for tooling; training uses VecFlyRunner)."""
        metadata = {"render_modes": []}

        def __init__(self, frame_skip: int = 4, max_seconds: float = 180.0):
            self.vec = VecFlyRunner(1, frame_skip=frame_skip, max_seconds=max_seconds)
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (STATE_SIZE,), np.float32)
            self.action_space = gym.spaces.Discrete(5)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.vec.reset(seed if seed is not None else int(self.np_random.integers(1 << 30)))[0].copy(), {}

        def step(self, action):
            obs, rew, done, info = self.vec.step([action])
            terminated = bool(done[0] and not info["truncated"][0]); truncated = bool(info["truncated"][0])
            o = info["final_obs"][0].copy() if done[0] else obs[0].copy()
            return o, float(rew[0]), terminated, truncated, {"distance": float(info["fin_distance"][0]) if done[0] else None}

    gym.register(id="FlyRunner-v0", entry_point="flybrain.env.vec:FlyRunnerEnv")
except ImportError:  # pragma: no cover
    pass
