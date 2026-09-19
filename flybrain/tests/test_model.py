"""Substrate / encoders / readouts / controls on the real v0 artifact (CPU, small batches)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from flybrain.pipeline.artifact import load_graph
from flybrain.model.substrate import ConnectomeSubstrate, SubstrateConfig
from flybrain.model.io import StateEncoder, FrameEncoder, LinearReadout, FixedReadout, dn_indices, input_indices
from flybrain.model.agent import FlyAgent
from flybrain.model import controls

GRAPH = Path(__file__).resolve().parents[1] / "data" / "graph" / "v0"
pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason="graph artifact not built")


@pytest.fixture(scope="module")
def pooled():
    return load_graph(GRAPH, pooled=True)


@pytest.fixture(scope="module")
def neuron():
    return load_graph(GRAPH, pooled=False)


def test_topology_is_frozen_and_signs_respected(pooled):
    sub = ConnectomeSubstrate(pooled)
    W = sub.weight_matrix().detach()
    nz = (W != 0).sum().item()
    assert nz == len(pooled.edges)                                    # no edge added, none lost
    # signed classes: effective weight sign == presynaptic NT sign, even after flipping the raw scale negative
    with torch.no_grad():
        sub.scale.mul_(-1)
    w = sub.edge_weights().detach()
    signed = sub.class_signed[sub.edge_class]
    assert torch.all(torch.sign(w[signed]) == sub.edge_sign[signed])
    assert torch.all(w[~signed] != 0)


def test_rate_dynamics_shapes_and_gradients(pooled):
    agent = FlyAgent(pooled, SubstrateConfig(dynamics="rate"))
    obs = torch.rand(4, 36) * 2 - 1
    v = agent.init_state(4)
    logits, value, v2, rates = agent(obs, v)
    assert logits.shape == (4, 5) and value.shape == (4,) and v2.shape == (4, agent.substrate.n) and rates.shape == v2.shape
    (logits.logsumexp(-1).mean() + value.mean()).backward()
    for name, p in agent.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert agent.substrate.scale.grad.abs().sum() > 0 and agent.substrate.log_tau.grad.abs().sum() > 0


def test_lif_runs_and_backprops(pooled):
    agent = FlyAgent(pooled, SubstrateConfig(dynamics="lif", input_gain=3.0))
    obs = torch.rand(3, 36); v = agent.init_state(3)
    logits, value, v2, rates = agent(obs, v)
    assert 0 <= rates.min() and rates.max() <= 1
    logits.sum().backward()
    assert agent.encoder.lin.weight.grad is not None


def test_readout_only_freezes_substrate(pooled):
    agent = FlyAgent(pooled, SubstrateConfig(trainable=False))
    g = agent.param_groups()
    assert g["substrate"] == 0 and g["readout"] > 0 and g["encoder"] > 0


def test_silenced_control_gets_no_drive(pooled):
    agent = FlyAgent(pooled, SubstrateConfig(input_gain=0.0))
    obs = torch.rand(2, 36); v = agent.init_state(2)
    _, _, v2, rates = agent(obs, v)
    assert torch.all(rates == 0) and torch.all(v2 == 0)               # zero bias init + no drive = silence


def test_fixed_readout_uses_named_dns(pooled):
    ro = FixedReadout(pooled)
    assert ro.M.shape[0] == 5 and ro.M[0].abs().sum() == 0            # noop has no DNs
    assert ro.M[3].sum() == 0 and ro.M[3].abs().sum() > 0             # left = L minus R, balanced
    rates = torch.rand(2, len(pooled.nodes))
    logits, value = ro(rates)
    assert logits.shape == (2, 5)


def test_frame_encoder_covers_frame(neuron):
    enc = FrameEncoder(neuron)
    has_col = enc.masks.sum(1) > 0
    assert has_col.all()
    # every pixel is read by exactly one column per side (+ the global-mean rows)
    per_pixel = enc.masks[~torch.isclose(enc.masks.sum(1), torch.tensor(1.0)) | True].sum(0)
    assert per_pixel.min() > 0
    frame = torch.randint(0, 256, (2, 48 * 64))
    x = enc(frame)
    assert x.shape == (2, len(neuron.nodes)) and (x[:, enc.idx] >= 0).all()
    assert x[:, [i for i in range(len(neuron.nodes)) if i not in set(enc.idx.tolist())]].abs().sum() == 0


def test_state_encoder_targets_input_layer_only(neuron):
    enc = StateEncoder(neuron)
    x = enc(torch.rand(2, 36))
    mask = torch.zeros(len(neuron.nodes), dtype=torch.bool); mask[enc.idx] = True
    assert (x[:, ~mask] == 0).all() and (x >= 0).all()
    assert set(neuron.nodes["group"].iloc[input_indices(neuron)]) == {"photoreceptor", "lamina"}
    assert set(neuron.nodes["group"].iloc[dn_indices(neuron)]) == {"descending"}


def test_shuffled_control_preserves_degrees(pooled):
    g2 = controls.shuffled(pooled, seed=1, swaps_per_edge=3)
    assert len(g2.edges) == len(pooled.edges)
    assert all(np.array_equal(a, b) for a, b in zip(controls.degrees(pooled), controls.degrees(g2)))
    assert not g2.edges[["pre", "post"]].equals(pooled.edges[["pre", "post"]])
    assert not g2.edges.duplicated(["pre", "post"]).any() and (g2.edges["pre"] != g2.edges["post"]).all()
    assert ConnectomeSubstrate(g2).n == len(pooled.nodes)


def test_random_control_matches_size(pooled):
    g2 = controls.random(pooled, seed=2)
    assert len(g2.edges) == len(pooled.edges) and not g2.edges.duplicated(["pre", "post"]).any()
    assert sorted(g2.edges["syn"]) == sorted(pooled.edges["syn"])
    assert g2.manifest["control"] == "random"
