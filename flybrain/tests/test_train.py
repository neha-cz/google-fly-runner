"""Env semantics, baselines parity, and PPO/ES smoke tests (CPU, tiny)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.env.vec import VecFlyRunner, RewardConfig
from flybrain.env.engine import LEFT, NOOP, JUMP
from flybrain.env.agents import HeuristicAgent, RandomAgent
from flybrain.train.policies import MLPPolicy, FlyPolicy
from flybrain.train.ppo import PPO, PPOConfig
from flybrain.train.es import ESConfig, train as es_train
from flybrain.model.substrate import SubstrateConfig
from flybrain.pipeline.artifact import load_graph

GRAPH = Path(__file__).resolve().parents[1] / "data" / "graph" / "v0"


def test_vec_env_reward_and_truncation():
    env = VecFlyRunner(3, seed=5, max_seconds=2.0)
    obs = env.reset(5)
    assert obs.shape == (3, 36)
    obs, rew, done, info = env.step([NOOP, LEFT, JUMP])
    assert rew[0] > 0 and abs((rew[1] - rew[0]) - (-0.05)) < 1e-6      # lane change cost
    assert not done.any()
    for _ in range(env.max_decisions):
        obs, rew, done, info = env.step([NOOP] * 3)
    assert env.episodes_done >= 3                                          # crashed or truncated at 2 s
    assert info["fin_distance"].size == 0 or np.all(info["fin_distance"] > 0)


def test_crash_penalty_and_autoreset():
    env = VecFlyRunner(1, seed=1)
    env.reset(1)
    total = 0; crashed = False
    for _ in range(env.max_decisions):
        obs, rew, done, info = env.step([NOOP])
        if done[0]:
            crashed = not info["truncated"][0]; total = rew[0]; break
    assert crashed and total < -40                                         # -50 crash dominates
    assert env.runners[0].alive and env.runners[0].distance < 1            # auto-reset happened


def _heuristic_outcomes(frame_skip: int) -> dict:
    env = VecFlyRunner(5, seed=0, frame_skip=frame_skip, max_seconds=60.0)
    env.runners = [env.runners[0].__class__(s) for s in (1, 2, 3, 42, 1234)]     # same seeds as the TS test
    obs = env.reset(); ag = HeuristicAgent(); res = {}
    for _ in range(env.max_decisions):
        obs, rew, done, info = env.step([ag.act(obs[i]) for i in range(5)])
        for j, i in enumerate(info["fin_idx"]):
            res.setdefault(int(i), (bool(info["truncated"][i]), float(info["fin_distance"][j])))
    return res


def test_heuristic_parity_with_ts():
    """At the TS test's cadence (a decision every tick) the heuristic survives 60 s on all five seeds,
    reaching the same distance the TS engine reports for a 60 s run."""
    res = _heuristic_outcomes(frame_skip=1)
    assert all(v[0] for v in res.values()) and len({round(v[1], 3) for v in res.values()}) == 1


def test_heuristic_at_decision_rate():
    """At 15 Hz the scripted policy's reaction thresholds are marginal at high speed; it must still
    last > 30 s on every seed (this is the baseline the learners are compared against)."""
    res = _heuristic_outcomes(frame_skip=4)
    assert all(v[1] > 400 for v in res.values())


def test_random_dies_fast():
    env = VecFlyRunner(8, seed=3, max_seconds=60.0)
    obs = env.reset(3); ags = [RandomAgent(i) for i in range(8)]
    lens = []
    for _ in range(env.max_decisions):
        obs, rew, done, info = env.step([a.act(obs[i]) for i, a in enumerate(ags)])
        lens += info["fin_length"].tolist()
        if len(lens) >= 8: break
    assert np.mean(lens[:8]) * 4 / 60 < 20


def test_gymnasium_env():
    gym = pytest.importorskip("gymnasium")
    env = gym.make("FlyRunner-v0")
    obs, _ = env.reset(seed=1)
    assert obs.shape == (36,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (36,) and isinstance(r, float)


def test_ppo_mlp_smoke(tmp_path):
    pol = MLPPolicy(); env = VecFlyRunner(8, seed=0)
    cfg = PPOConfig(n_envs=8, rollout=8, epochs=1, minibatches=2, total_steps=8 * 8 * 3)
    last = PPO(pol, env, cfg, "cpu", tmp_path).train()
    assert last["update"] == 3 and (tmp_path / "model.pt").exists()
    assert np.isfinite(last["policy_loss"])


@pytest.mark.skipif(not GRAPH.exists(), reason="graph artifact not built")
def test_ppo_fly_recurrent_smoke(tmp_path):
    g = load_graph(GRAPH, pooled=True)
    pol = FlyPolicy(g, SubstrateConfig(input_gain=3.0, scale_init=5.0, trainable=True))
    before = pol.agent.substrate.scale.detach().clone()
    env = VecFlyRunner(4, seed=0)
    cfg = PPOConfig(n_envs=4, rollout=6, epochs=1, minibatches=2, total_steps=4 * 6 * 2, scale_max=6.0)
    last = PPO(pol, env, cfg, "cpu", tmp_path).train()
    assert last["update"] == 2 and np.isfinite(last["policy_loss"])
    assert not torch.equal(before, pol.agent.substrate.scale.detach())          # substrate actually trained
    assert pol.agent.substrate.scale.abs().max() <= 6.0


def test_reward_normalizer_scales_to_unit_return_std():
    from flybrain.train.ppo import RewardNormalizer
    rn = RewardNormalizer(4, 0.99, "cpu")
    for _ in range(200):
        r = torch.randn(4) * 20; d = (torch.rand(4) < 0.05).float()
        out = rn(r, d)
    assert 10 < rn.var ** 0.5 < 300 and out.abs().mean() < 5


def test_action_prior_and_reward_normalisation_wiring(tmp_path):
    pol = MLPPolicy(noop_bias=3.0)
    with torch.no_grad():
        p = torch.softmax(pol(torch.zeros(1, 36))[0], -1)[0]
    assert 0.75 < p[0] < 0.9                                            # ~83% no-op at init
    env = VecFlyRunner(8, seed=0)
    last = PPO(pol, env, PPOConfig(n_envs=8, rollout=8, epochs=1, minibatches=2, total_steps=8 * 8 * 2), "cpu", tmp_path).train()
    assert last["reward_scale"] > 0 and np.isfinite(last["value_loss"])


def test_es_smoke(tmp_path):
    pol = MLPPolicy(); env = VecFlyRunner(2, seed=0)
    last = es_train(pol, env, ESConfig(population=2, n_envs=2, max_decisions=20, generations=2), "cpu", tmp_path)
    assert last["generation"] == 1 and (tmp_path / "model.pt").exists()
