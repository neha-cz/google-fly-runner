"""Human-readable summary of a graph artifact."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact import load_graph


def run(path: Path) -> int:
    g = load_graph(path)
    m = g.manifest
    print(f"artifact {path}  dataset={m['dataset']}  built={m['built']}")
    print(f"  neurons={g.n:,}  edges={len(g.edges):,}  synapses={m['summary']['synapses']:,}  pooled: {m['summary']['n_pool_nodes']} nodes / {m['summary']['n_pool_edges']} edges")
    print("  by group:", m["summary"]["neurons_by_group"])
    print("  raw inputs:", {k: v["sha256"][:12] for k, v in m.get("raw_files", {}).items()})
    A = g.adjacency(signed=False)
    indeg = np.asarray(A.getnnz(axis=1)).ravel(); outdeg = np.asarray(A.getnnz(axis=0)).ravel()
    print(f"  in-degree  mean {indeg.mean():.1f}  median {np.median(indeg):.0f}  max {indeg.max()}")
    print(f"  out-degree mean {outdeg.mean():.1f}  median {np.median(outdeg):.0f}  max {outdeg.max()}")
    dense_mb = g.n * g.n * 4 / 1e6
    print(f"  dense fp32 weight matrix would be {dense_mb:,.0f} MB; sparse nnz={A.nnz:,}")
    print("  types (count, mean in-degree):")
    for t, sub in g.nodes.groupby("type"):
        idx = sub.index.to_numpy()
        print(f"    {t:16s} {len(idx):5d}  side={dict(sub['side'].value_counts())}  in={indeg[idx].mean():6.1f}  out={outdeg[idx].mean():6.1f}  nt={sub['nt'].mode().iat[0]}")
    for s in m["steps"]:
        print("  step:", s)
    return 0
