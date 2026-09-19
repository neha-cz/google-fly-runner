"""Build the task subgraph artifact from the MaleCNS v1.0 bulk files.

Steps (each recorded in manifest.json with counts):
  1. resolve cell-type regexes -> neuron table (bodyId, type, group, side, hex column, status)
  2. status filter (Traced), with exemptions for input groups whose fragments are still useful
  3. load connectome weights, restrict to selected bodies, threshold on synapse count
  4. assign an optic-lobe column to columnar neurons that lack `assignedOlHex*` (T4/T5,
     photoreceptors) from their strongest synaptic partner that has one
  5. keep only columnar neurons within `hex_radius` of the per-side centre column
  6. prune nodes that are not on any input -> output path
  7. attach sign from consensus neurotransmitter prediction
  8. write per-neuron and (type, side)-pooled graphs + manifest
"""
from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

# Sign convention for the substrate. Histamine (photoreceptor output) is inhibitory in the fly
# lamina via HisCl channels. Modulatory amines are left unsigned (0) -> sign-free learnable scale.
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 0, "serotonin": 0, "octopamine": 0, "tyramine": 0, "unclear": 0, "unknown": 0}


class BuildError(RuntimeError):
    pass


def hex_distance(q: np.ndarray, r: np.ndarray, q0: float, r0: float) -> np.ndarray:
    dq, dr = q - q0, r - r0
    return np.maximum.reduce([np.abs(dq), np.abs(dr), np.abs(dq + dr)])


def resolve_types(all_types: pd.Series, patterns: dict[str, list[str]]) -> tuple[dict[str, str], dict]:
    """Map each concrete type -> group; return also the per-pattern resolution for the manifest."""
    type_to_group: dict[str, str] = {}
    resolution: dict[str, dict] = {}
    uniq = all_types.dropna().unique()
    for group, pats in patterns.items():
        for p in pats:
            rx = re.compile(p)
            matched = sorted(t for t in uniq if rx.match(str(t)))
            resolution[p] = {"group": group, "types": matched}
            for t in matched:
                if t in type_to_group and type_to_group[t] != group:
                    raise BuildError(f"type {t} matched by two groups: {type_to_group[t]} and {group}")
                type_to_group[t] = group
    unmatched = [p for p, r in resolution.items() if not r["types"]]
    if unmatched:
        raise BuildError(f"cell-type patterns matched nothing in this release: {unmatched}")
    return type_to_group, resolution


def _reachable(n: int, edges: pd.DataFrame, sources: np.ndarray, reverse: bool) -> np.ndarray:
    if len(sources) == 0:
        return np.zeros(n, dtype=bool)
    pre, post = edges["pre"].to_numpy(), edges["post"].to_numpy()
    if reverse:
        pre, post = post, pre
    # virtual root (index n) fans out to all sources so one BFS covers them
    rows = np.concatenate([pre, np.full(len(sources), n)]); cols = np.concatenate([post, sources])
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n + 1, n + 1))
    order = breadth_first_order(A, n, directed=True, return_predecessors=False)
    m = np.zeros(n + 1, dtype=bool); m[order] = True
    return m[:n]


def build(config: dict, raw_dir: Path, out_dir: Path, raw_manifest: dict | None = None) -> dict:
    t0 = time.time()
    files = config["source"]["files"]
    steps: list[dict] = []

    def step(name: str, **counts):
        steps.append({"step": name, **counts}); print(f"[{time.time()-t0:6.1f}s] {name}: {counts}")

    # 1. neurons
    ann = feather.read_table(raw_dir / files["annotations"],
                             columns=["bodyId", "type", "somaSide", "superclass", "status", "assignedOlHex1", "assignedOlHex2"]).to_pandas()
    type_to_group, resolution = resolve_types(ann["type"], config["type_patterns"])
    nodes = ann[ann["type"].isin(type_to_group)].copy()
    nodes["group"] = nodes["type"].map(type_to_group)
    nodes["side"] = nodes["somaSide"].fillna("U").astype(str)
    nodes = nodes.rename(columns={"assignedOlHex1": "hex1", "assignedOlHex2": "hex2"})
    step("resolve_types", types=len(type_to_group), neurons=len(nodes))

    # 2. status
    ok_status = set(config.get("require_status", []))
    if ok_status:
        exempt = nodes["group"].isin(config.get("status_exempt_groups", []))
        keep = nodes["status"].isin(ok_status) | exempt
        step("status_filter", dropped=int((~keep).sum()), kept=int(keep.sum()))
        nodes = nodes[keep]
    nodes = nodes.reset_index(drop=True)
    body_to_idx = pd.Series(np.arange(len(nodes)), index=nodes["bodyId"].to_numpy())

    # 3. edges among selected bodies
    wt = feather.read_table(raw_dir / files["weights"], memory_map=True)
    bodies = pa.array(nodes["bodyId"].to_numpy())
    mask = pc.and_(pc.is_in(wt["body_pre"], value_set=bodies), pc.is_in(wt["body_post"], value_set=bodies))
    sub = wt.filter(mask)
    del wt
    edges = sub.to_pandas()
    edges = edges.rename(columns={"weight": "syn"})
    total_before = len(edges)
    edges = edges[edges["syn"] >= config["min_synapses"]].reset_index(drop=True)
    edges["pre"] = body_to_idx[edges["body_pre"].to_numpy()].to_numpy()
    edges["post"] = body_to_idx[edges["body_post"].to_numpy()].to_numpy()
    step("load_edges", pairs_among_selected=total_before, kept_ge_min_syn=len(edges), min_synapses=config["min_synapses"])

    # 4. column assignment for columnar neurons without hex coords
    columnar = nodes["group"].isin(config.get("columnar_groups", [])).to_numpy()
    has_hex = nodes["hex1"].notna().to_numpy()
    nodes["hex_source"] = np.where(has_hex, "annotation", "")
    assigned_total = 0
    for _round in range(3):
        need = columnar & ~nodes["hex1"].notna().to_numpy()
        if not need.any():
            break
        e = edges[(edges["pre"].isin(np.flatnonzero(need)) & edges["post"].isin(np.flatnonzero(nodes["hex1"].notna())))
                  | (edges["post"].isin(np.flatnonzero(need)) & edges["pre"].isin(np.flatnonzero(nodes["hex1"].notna())))]
        if e.empty:
            break
        need_idx = np.where(e["pre"].isin(np.flatnonzero(need)), e["pre"], e["post"]).astype(int)
        partner = np.where(need_idx == e["pre"], e["post"], e["pre"]).astype(int)
        # restrict to same side (or partner side unknown)
        same_side = (nodes["side"].to_numpy()[need_idx] == nodes["side"].to_numpy()[partner]) | (nodes["side"].to_numpy()[need_idx] == "U")
        d = pd.DataFrame({"i": need_idx, "p": partner, "syn": e["syn"].to_numpy()})[same_side]
        best = d.sort_values("syn", ascending=False).drop_duplicates("i")
        nodes.loc[best["i"].to_numpy(), "hex1"] = nodes["hex1"].to_numpy()[best["p"].to_numpy()]
        nodes.loc[best["i"].to_numpy(), "hex2"] = nodes["hex2"].to_numpy()[best["p"].to_numpy()]
        nodes.loc[best["i"].to_numpy(), "hex_source"] = f"partner_round{_round}"
        # a side-less neuron inherits its partner's side
        side_arr = nodes["side"].to_numpy().copy()
        unk = side_arr[best["i"].to_numpy()] == "U"
        side_arr[best["i"].to_numpy()[unk]] = side_arr[best["p"].to_numpy()[unk]]
        nodes["side"] = side_arr
        assigned_total += len(best)
    still_missing = int((columnar & ~nodes["hex1"].notna().to_numpy()).sum())
    step("assign_columns", assigned_from_partners=assigned_total, columnar_without_column=still_missing)

    # 5. hex patch
    R = config.get("hex_radius")
    if R is not None:
        keep = ~columnar
        centres = {}
        for side in ["L", "R"]:
            m = columnar & (nodes["side"] == side).to_numpy() & nodes["hex1"].notna().to_numpy()
            if not m.any():
                continue
            q0, r0 = float(nodes.loc[m, "hex1"].median()), float(nodes.loc[m, "hex2"].median())
            centres[side] = [q0, r0]
            d = hex_distance(nodes["hex1"].to_numpy(dtype=float), nodes["hex2"].to_numpy(dtype=float), q0, r0)
            keep |= m & (d <= R)
        step("hex_patch", radius=R, centres=centres, dropped=int((~keep).sum()), kept=int(keep.sum()))
        nodes, edges, body_to_idx = _subset(nodes, edges, keep)

    # 6. reachability prune
    if config.get("prune_unreachable", True):
        inputs = np.flatnonzero(nodes["group"].isin(config["input_groups"]))
        outputs = np.flatnonzero(nodes["group"].isin(config["output_groups"]))
        fwd = _reachable(len(nodes), edges, inputs, reverse=False)
        bwd = _reachable(len(nodes), edges, outputs, reverse=True)
        keep = fwd & bwd
        dropped_by_group = nodes.loc[~keep, "group"].value_counts().to_dict()
        step("prune_unreachable", inputs=len(inputs), outputs=len(outputs), reach_from_inputs=int(fwd.sum()),
             reach_to_outputs=int(bwd.sum()), kept=int(keep.sum()), dropped_by_group=dropped_by_group)
        if keep.sum() == 0:
            raise BuildError("no node lies on an input->output path; check groups/thresholds")
        nodes, edges, body_to_idx = _subset(nodes, edges, keep)

    if len(nodes) > config["max_neurons"]:
        raise BuildError(f"{len(nodes)} neurons exceeds max_neurons={config['max_neurons']}; lower hex_radius or tighten the allowlist")

    # 7. signs
    nt = feather.read_table(raw_dir / files["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt_confidence"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    nodes["nt"] = nodes["bodyId"].map(nt["consensus_nt"]).fillna("unknown").str.lower()
    nodes["nt_conf"] = nodes["bodyId"].map(nt["predicted_nt_confidence"]).fillna(0.0)
    nodes["sign"] = nodes["nt"].map(NT_SIGN).fillna(0).astype(int)
    edges["sign"] = nodes["sign"].to_numpy()[edges["pre"].to_numpy()]
    step("signs", by_nt=nodes["nt"].value_counts().to_dict(), unsigned_edges=int((edges["sign"] == 0).sum()))

    # 8. pooled graph: node = (type, side)
    nodes["pool"] = nodes["type"] + "_" + nodes["side"]
    pool_nodes = (nodes.groupby("pool").agg(type=("type", "first"), side=("side", "first"), group=("group", "first"),
                                            n_neurons=("bodyId", "size"), sign=("sign", lambda s: int(np.sign(s.sum()))))
                  .reset_index())
    pool_idx = pd.Series(np.arange(len(pool_nodes)), index=pool_nodes["pool"])
    pe = pd.DataFrame({"pre": pool_idx[nodes["pool"].to_numpy()[edges["pre"]]].to_numpy(),
                       "post": pool_idx[nodes["pool"].to_numpy()[edges["post"]]].to_numpy(), "syn": edges["syn"]})
    pool_edges = pe.groupby(["pre", "post"], as_index=False).agg(syn=("syn", "sum"), n_pairs=("syn", "size"))
    pool_edges["syn_per_post"] = pool_edges["syn"] / pool_nodes["n_neurons"].to_numpy()[pool_edges["post"]]
    pool_edges["sign"] = pool_nodes["sign"].to_numpy()[pool_edges["pre"]]
    step("pool", pool_nodes=len(pool_nodes), pool_edges=len(pool_edges))

    # write
    out = out_dir / config["version"]
    out.mkdir(parents=True, exist_ok=True)
    node_cols = ["bodyId", "type", "group", "side", "superclass", "status", "hex1", "hex2", "hex_source", "nt", "nt_conf", "sign", "pool"]
    nodes[node_cols].to_parquet(out / "nodes.parquet", index=False)
    edges[["pre", "post", "body_pre", "body_post", "syn", "sign"]].to_parquet(out / "edges.parquet", index=False)
    pool_nodes.to_parquet(out / "pool_nodes.parquet", index=False)
    pool_edges.to_parquet(out / "pool_edges.parquet", index=False)
    import flybrain
    manifest = {
        "version": config["version"], "dataset": config["dataset"], "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "flybrain_version": flybrain.__version__, "python": platform.python_version(),
        "config": config, "raw_files": raw_manifest or {}, "type_resolution": resolution, "nt_sign_convention": NT_SIGN,
        "steps": steps,
        "summary": {"n_neurons": len(nodes), "n_edges": len(edges), "synapses": int(edges["syn"].sum()),
                    "n_pool_nodes": len(pool_nodes), "n_pool_edges": len(pool_edges),
                    "neurons_by_group": nodes["group"].value_counts().to_dict(),
                    "neurons_by_type": nodes["type"].value_counts().to_dict()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {out}: {len(nodes)} neurons, {len(edges)} edges; pooled {len(pool_nodes)} nodes, {len(pool_edges)} edges")
    return manifest


def _subset(nodes: pd.DataFrame, edges: pd.DataFrame, keep: np.ndarray):
    new_idx = np.full(len(nodes), -1); new_idx[keep] = np.arange(int(keep.sum()))
    nodes = nodes[keep].reset_index(drop=True)
    em = keep[edges["pre"].to_numpy()] & keep[edges["post"].to_numpy()]
    edges = edges[em].reset_index(drop=True)
    edges["pre"] = new_idx[edges["pre"].to_numpy()]; edges["post"] = new_idx[edges["post"].to_numpy()]
    body_to_idx = pd.Series(np.arange(len(nodes)), index=nodes["bodyId"].to_numpy())
    return nodes, edges, body_to_idx
