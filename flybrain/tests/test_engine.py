"""The Python engine must reproduce the TypeScript engine's golden trajectories."""
import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.env.engine import Runner, Rng, STATE_SIZE, ACTIONS

GOLDEN = Path(__file__).resolve().parents[2] / "game" / "fixtures" / "golden.json"


def test_rng_matches_ts():
    assert Rng(1).next_u32() == 270369
    r = Rng(42); xs = [r.next_u32() for _ in range(5)]
    assert all(0 <= x < 2 ** 32 for x in xs) and len(set(xs)) == 5


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden fixture missing")
def test_golden_trajectories():
    g = json.loads(GOLDEN.read_text())
    assert g["stateSize"] == STATE_SIZE
    st = np.zeros(STATE_SIZE, dtype=np.float32)
    for ep in g["episodes"]:
        r = Runner(ep["seed"]); ci = 0
        for i, a in enumerate(ep["actions"]):
            r.act(ACTIONS.index(a)); r.step()
            if (i + 1) % g["every"] == 0:
                c = ep["checkpoints"][ci]; ci += 1
                assert r.alive == c["alive"], (ep["seed"], i)
                assert abs(r.distance - c["distance"]) < 1e-5 and abs(r.score - c["score"]) < 1e-5
                assert r.coins_collected == c["coins"] and r.lane == c["lane"]
                assert len(r.obstacles) == c["nObstacles"]
                r.get_state(st)
                np.testing.assert_allclose(st, np.array(c["state"], dtype=np.float32), atol=2e-6, err_msg=f"seed {ep['seed']} step {i+1}")
        assert abs(r.distance - ep["final"]["distance"]) < 1e-5
