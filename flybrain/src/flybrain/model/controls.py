"""Control graphs, produced from the real graph by the same loader so nothing else changes.

  shuffled : degree-preserving double-edge swaps (in/out degree of every node kept; synapse counts
             and signs travel with the presynaptic endpoint). Tests whether the *specific* wiring
             matters beyond the degree sequence.
  random   : same node set, same edge count, uniformly random (pre, post) pairs; synapse counts
             permuted. Tests whether anything beyond size/density matters.
  silenced : not a graph transform; set SubstrateConfig.input_gain = 0 (the readout sees only
             spontaneous activity). Tests that the readout depends on circuit activity at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from flybrain.pipeline.artifact import Graph


def _finish(graph: Graph, pre: np.ndarray, post: np.ndarray, syn: np.ndarray, kind: str) -> Graph:
    e = pd.DataFrame({"pre": pre, "post": post, "syn": syn})
    e["sign"] = graph.nodes["sign"].to_numpy()[pre]
    if graph.pooled:
        e["syn_per_post"] = syn / graph.nodes["n_neurons"].to_numpy()[post]
    m = dict(graph.manifest); m["control"] = kind
    return Graph(nodes=graph.nodes, edges=e.reset_index(drop=True), manifest=m, pooled=graph.pooled)


def shuffled(graph: Graph, seed: int = 0, swaps_per_edge: int = 10) -> Graph:
    rng = np.random.default_rng(seed)
    pre = graph.edges["pre"].to_numpy().copy(); post = graph.edges["post"].to_numpy().copy()
    syn = graph.edges["syn"].to_numpy().copy()
    m = len(pre); n = len(graph.nodes)
    existing = set((pre * n + post).tolist())
    done = 0; target = swaps_per_edge * m
    while done < target:
        a, c = rng.integers(0, m, 2)
        pa, pb, pc, pd_ = pre[a], post[a], pre[c], post[c]
        if pa == pc or pb == pd_ or pa == pd_ or pc == pb:
            continue
        k1, k2 = pa * n + pd_, pc * n + pb
        if k1 in existing or k2 in existing:
            continue
        existing.discard(pa * n + pb); existing.discard(pc * n + pd_); existing.add(k1); existing.add(k2)
        post[a], post[c] = pd_, pb
        done += 1
    return _finish(graph, pre, post, syn, "shuffled")


def random(graph: Graph, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    m = len(graph.edges); n = len(graph.nodes)
    keys = set()
    while len(keys) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            keys.add(int(a) * n + int(b))
    keys = np.fromiter(keys, dtype=np.int64, count=m)
    pre, post = keys // n, keys % n
    syn = rng.permutation(graph.edges["syn"].to_numpy())
    return _finish(graph, pre, post, syn, "random")


def degrees(graph: Graph) -> tuple[np.ndarray, np.ndarray]:
    n = len(graph.nodes)
    return (np.bincount(graph.edges["post"].to_numpy(), minlength=n), np.bincount(graph.edges["pre"].to_numpy(), minlength=n))
