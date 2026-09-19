"""Read side of the versioned graph artifact written by `flybrain build-graph`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Graph:
    nodes: pd.DataFrame       # one row per node; index = contiguous node index
    edges: pd.DataFrame       # pre, post (node indices), syn (synapse count), sign (+1/-1/0)
    manifest: dict
    pooled: bool

    @property
    def n(self) -> int:
        return len(self.nodes)

    def adjacency(self, signed: bool = True) -> sp.csr_matrix:
        """Sparse (post x pre) matrix so that `A @ r_pre` gives per-post input."""
        w = self.edges["syn"].to_numpy(dtype=np.float32)
        if signed:
            w = w * np.where(self.edges["sign"].to_numpy() == 0, 1, self.edges["sign"].to_numpy())
        return sp.csr_matrix((w, (self.edges["post"].to_numpy(), self.edges["pre"].to_numpy())), shape=(self.n, self.n))

    def indices(self, group: str | None = None, types: list[str] | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if group: m &= (self.nodes["group"] == group).to_numpy()
        if types: m &= self.nodes["type"].isin(types).to_numpy()
        return np.flatnonzero(m)


def load_graph(path: str | Path, pooled: bool = False) -> Graph:
    p = Path(path)
    prefix = "pool_" if pooled else ""
    nodes = pd.read_parquet(p / f"{prefix}nodes.parquet")
    edges = pd.read_parquet(p / f"{prefix}edges.parquet")
    manifest = json.loads((p / "manifest.json").read_text())
    return Graph(nodes=nodes, edges=edges, manifest=manifest, pooled=pooled)
