"""Exercise the graph build on a tiny synthetic 'release' with the real file layout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from flybrain.pipeline.build import build, BuildError, hex_distance
from flybrain.pipeline.artifact import load_graph

CONFIG = {
    "version": "test", "dataset": "synthetic",
    "source": {"base_url": "", "files": {"annotations": "ann.feather", "neurotransmitters": "nt.feather", "weights": "w.feather"}},
    "type_patterns": {"photoreceptor": ["^R1-R6$"], "lamina": ["^L1$"], "motion": ["^T4a$"], "descending": ["^DNp01$"]},
    "input_groups": ["photoreceptor"], "output_groups": ["descending"],
    "columnar_groups": ["photoreceptor", "lamina", "motion"],
    "require_status": ["Traced"], "status_exempt_groups": ["photoreceptor"],
    "min_synapses": 5, "prune_unreachable": True, "hex_radius": 1, "max_neurons": 1000,
}


def make_release(raw: Path):
    rows, nt = [], []
    bid = 1
    def add(t, side, status="Traced", hex=(None, None), nt_name="acetylcholine"):
        nonlocal bid
        rows.append(dict(bodyId=bid, type=t, somaSide=side, superclass="x", status=status, assignedOlHex1=hex[0], assignedOlHex2=hex[1]))
        nt.append(dict(body=bid, consensus_nt=nt_name, predicted_nt_confidence=0.9)); bid += 1; return bid - 1
    # right side: 7 columns (centre + ring of 1) + 1 far column; L1 has hex, R1-R6 and T4a don't
    cols = [(10, 10), (11, 10), (9, 10), (10, 11), (10, 9), (11, 9), (9, 11), (20, 20)]
    L1 = [add("L1", "R", hex=c, nt_name="glutamate") for c in cols]
    R16 = [add("R1-R6", None, status=None, nt_name="histamine") for _ in cols]   # no status, no side, no hex
    T4 = [add("T4a", "R", nt_name="acetylcholine") for _ in cols]
    dn = add("DNp01", "R", nt_name="acetylcholine")
    orphan_L1 = add("L1", "R", status="Orphan", hex=(10, 10))           # dropped by status
    dead_end = add("T4a", "R")                                          # gets column but no path to DN
    unrelated = add("Mi15", "R")                                        # not in allowlist
    ann = pd.DataFrame(rows); ntdf = pd.DataFrame(nt)
    w = []
    for i in range(len(cols)):
        w.append((R16[i], L1[i], 50)); w.append((L1[i], T4[i], 30)); w.append((T4[i], dn, 12))
        w.append((L1[i], R16[i], 2))                                    # below threshold
    w.append((L1[0], dead_end, 40)); w.append((unrelated, dn, 99)); w.append((L1[0], orphan_L1, 40))
    wdf = pd.DataFrame(w, columns=["body_pre", "body_post", "weight"])
    feather.write_feather(ann, raw / "ann.feather"); feather.write_feather(ntdf, raw / "nt.feather"); feather.write_feather(wdf, raw / "w.feather")
    return dict(L1=L1, R16=R16, T4=T4, dn=dn, dead_end=dead_end)


def test_hex_distance():
    assert hex_distance(np.array([10, 11, 12]), np.array([10, 10, 8]), 10, 10).tolist() == [0, 1, 2]


def test_build_end_to_end(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); ids = make_release(raw)
    m = build(CONFIG, raw, tmp_path / "graph")
    g = load_graph(tmp_path / "graph" / "test")
    # 7 columns within radius 1 x 3 columnar types + 1 DN; far column, orphan, dead end, unrelated all gone
    assert m["summary"]["n_neurons"] == 7 * 3 + 1
    assert g.nodes["type"].value_counts().to_dict() == {"L1": 7, "R1-R6": 7, "T4a": 7, "DNp01": 1}
    assert len(g.edges) == 7 * 3
    # photoreceptors survived the status filter and got column + side from their L1 partner
    r = g.nodes[g.nodes["type"] == "R1-R6"]
    assert (r["hex_source"] == "partner_round0").all() and (r["side"] == "R").all()
    # T4a got its column from L1 in the same round (direct partner)
    assert (g.nodes[g.nodes["type"] == "T4a"]["hex1"].notna()).all()
    # signs from consensus NT
    e = g.edges.merge(g.nodes[["type"]], left_on="pre", right_index=True)
    assert set(e[e["type"] == "R1-R6"]["sign"]) == {-1} and set(e[e["type"] == "L1"]["sign"]) == {-1} and set(e[e["type"] == "T4a"]["sign"]) == {1}
    # pooled graph: (type, side) nodes with summed synapses
    gp = load_graph(tmp_path / "graph" / "test", pooled=True)
    assert len(gp.nodes) == 4
    t4_dn = gp.edges.merge(gp.nodes[["pool"]], left_on="pre", right_index=True)
    row = t4_dn[t4_dn["pool"] == "T4a_R"].iloc[0]
    assert row["syn"] == 7 * 12 and row["n_pairs"] == 7 and row["syn_per_post"] == 84
    # adjacency: post x pre, signed
    A = g.adjacency()
    dn_idx = g.indices(types=["DNp01"])[0]
    assert A[dn_idx].sum() == 7 * 12
    steps = {s["step"] for s in m["steps"]}
    assert {"resolve_types", "status_filter", "load_edges", "assign_columns", "hex_patch", "prune_unreachable", "signs", "pool"} <= steps


def test_unmatched_pattern_fails(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); make_release(raw)
    cfg = json.loads(json.dumps(CONFIG)); cfg["type_patterns"]["descending"].append("^DNzz$")
    with pytest.raises(BuildError, match="matched nothing"):
        build(cfg, raw, tmp_path / "graph")


def test_size_cap(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir(); make_release(raw)
    cfg = json.loads(json.dumps(CONFIG)); cfg["max_neurons"] = 5
    with pytest.raises(BuildError, match="exceeds max_neurons"):
        build(cfg, raw, tmp_path / "graph")
