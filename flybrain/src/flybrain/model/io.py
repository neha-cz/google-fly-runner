"""Input encoders (game -> input-layer drive) and DN readouts (DN activity -> actions)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn, Tensor

from flybrain.pipeline.artifact import Graph

ACTIONS = ["noop", "jump", "slide", "left", "right"]
STATE_SIZE = 36


def input_indices(graph: Graph) -> np.ndarray:
    groups = graph.manifest["config"]["input_groups"]
    return np.flatnonzero(graph.nodes["group"].isin(groups).to_numpy())


def dn_indices(graph: Graph) -> np.ndarray:
    groups = graph.manifest["config"]["output_groups"]
    return np.flatnonzero(graph.nodes["group"].isin(groups).to_numpy())


class StateEncoder(nn.Module):
    """Learned linear map from the 36-float game state onto the input-layer nodes (non-negative drive)."""
    def __init__(self, graph: Graph, state_size: int = STATE_SIZE):
        super().__init__()
        idx = input_indices(graph)
        self.register_buffer("idx", torch.as_tensor(idx))
        self.n = len(graph.nodes)
        self.lin = nn.Linear(state_size, len(idx))
        nn.init.normal_(self.lin.weight, std=0.3); nn.init.constant_(self.lin.bias, 0.2)

    def forward(self, obs: Tensor) -> Tensor:
        drive = torch.relu(self.lin(obs))
        x = obs.new_zeros(obs.shape[0], self.n)
        return x.index_copy(1, self.idx, drive)


class FrameEncoder(nn.Module):
    """Fixed retinotopic sampling: each columnar input node reads the mean luminance of the frame
    region nearest its optic-lobe column (hex axial coords from the graph), as contrast in [-0.5, 0.5],
    times a learnable per-type gain. Non-columnar input nodes (no column) receive the frame mean.
    Requires the per-neuron graph (pooled nodes have no columns)."""
    def __init__(self, graph: Graph, frame_w: int = 64, frame_h: int = 48):
        super().__init__()
        if graph.pooled:
            raise ValueError("FrameEncoder needs the per-neuron graph")
        idx = input_indices(graph)
        nodes = graph.nodes.iloc[idx]
        self.n, self.w, self.h = len(graph.nodes), frame_w, frame_h
        self.register_buffer("idx", torch.as_tensor(idx))
        # project hex axial (q, r) per side to image coords; left eye -> left half, right eye -> right half
        q, r = nodes["hex1"].to_numpy(dtype=float), nodes["hex2"].to_numpy(dtype=float)
        side = nodes["side"].to_numpy()
        has = ~np.isnan(q)
        px = np.full(len(idx), np.nan); py = np.full(len(idx), np.nan)
        for s, x0 in (("L", 0.25), ("R", 0.75)):
            m = has & (side == s)
            if not m.any():
                continue
            cx, cy = q[m] + r[m] / 2, r[m] * np.sqrt(3) / 2
            cx -= cx.mean(); cy -= cy.mean()
            span = max(np.abs(cx).max(), np.abs(cy).max(), 1e-6)
            px[m] = (x0 + 0.22 * cx / span) * frame_w; py[m] = (0.5 + 0.45 * cy / span) * frame_h
        # nearest-column assignment for every pixel within each half; nodes sharing a column share its mask
        yy, xx = np.mgrid[0:frame_h, 0:frame_w]
        masks = np.zeros((len(idx), frame_h * frame_w), dtype=np.float32)
        for s, half in (("L", xx < frame_w / 2), ("R", xx >= frame_w / 2)):
            members = np.flatnonzero(has & (side == s))
            if len(members) == 0:
                continue
            cols, col_of = np.unique(np.stack([px[members], py[members]], 1), axis=0, return_inverse=True)
            d = (xx[..., None] - cols[:, 0]) ** 2 + (yy[..., None] - cols[:, 1]) ** 2
            nearest = d.argmin(-1).ravel()                                  # column id per pixel
            pix = np.flatnonzero(half.ravel())
            col_mask = np.zeros((len(cols), frame_h * frame_w), dtype=np.float32)
            col_mask[nearest[pix], pix] = 1
            masks[members] = col_mask[col_of.ravel()]
        masks[~has] = 1.0 / (frame_h * frame_w)           # columnless inputs: global mean
        masks[has] /= np.maximum(masks[has].sum(1, keepdims=True), 1)
        self.register_buffer("masks", torch.as_tensor(masks))          # [n_in, H*W]
        types = nodes["type"].astype("category")
        self.register_buffer("type_of", torch.as_tensor(types.cat.codes.to_numpy().astype(np.int64)))
        self.gain = nn.Parameter(torch.ones(len(types.cat.categories)))
        self.type_names = list(types.cat.categories)

    def forward(self, frame: Tensor) -> Tensor:
        """frame: [B, H*W] uint8 or float in [0, 255]."""
        f = frame.to(self.masks.dtype).reshape(frame.shape[0], -1) / 255.0
        lum = f @ self.masks.T                                            # [B, n_in]
        drive = torch.relu((lum - 0.5) * self.gain[self.type_of] + 0.5)   # contrast + offset, non-negative
        x = f.new_zeros(f.shape[0], self.n)
        return x.index_copy(1, self.idx, drive)


class DNNorm(nn.Module):
    """Per-DN running standardisation of window-mean rates (no learnable capacity). Calibrated
    substrates can leave DN rates at ~0.03 ± 0.003, which makes any readout see a near-constant
    input; standardising fixes the conditioning without touching the circuit."""
    def __init__(self, n: int, momentum: float = 0.01):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n)); self.register_buffer("var", torch.ones(n)); self.momentum = momentum

    def forward(self, x: Tensor) -> Tensor:
        if self.training and x.shape[0] > 1:
            with torch.no_grad():
                self.mean.lerp_(x.mean(0), self.momentum); self.var.lerp_(x.var(0, unbiased=False), self.momentum)
        return (x - self.mean) / (self.var + 1e-8).sqrt()


class LinearReadout(nn.Module):
    """Trained linear map from (standardised) DN activity -> action logits and a value estimate."""
    def __init__(self, graph: Graph, n_actions: int = len(ACTIONS)):
        super().__init__()
        idx = dn_indices(graph)
        self.register_buffer("idx", torch.as_tensor(idx))
        self.norm = DNNorm(len(idx))
        self.policy = nn.Linear(len(idx), n_actions)
        self.value = nn.Linear(len(idx), 1)
        self.dn_names = [f"{t}_{s}" for t, s in zip(graph.nodes["type"].iloc[idx], graph.nodes["side"].iloc[idx])]

    def forward(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        dn = self.norm(rates[:, self.idx])
        return self.policy(dn), self.value(dn).squeeze(-1)


class MLPReadout(nn.Module):
    """Fly Dino-style small nonlinear head on DN activity (their 16-12-3): DN rates -> hidden -> logits/value."""
    def __init__(self, graph: Graph, hidden: int = 32, n_actions: int = len(ACTIONS)):
        super().__init__()
        idx = dn_indices(graph)
        self.register_buffer("idx", torch.as_tensor(idx))
        self.norm = DNNorm(len(idx))
        self.body = nn.Sequential(nn.Linear(len(idx), hidden), nn.Tanh())
        self.policy = nn.Linear(hidden, n_actions); self.value = nn.Linear(hidden, 1)
        self.dn_names = [f"{t}_{s}" for t, s in zip(graph.nodes["type"].iloc[idx], graph.nodes["side"].iloc[idx])]

    def forward(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        h = self.body(self.norm(rates[:, self.idx]))
        return self.policy(h), self.value(h).squeeze(-1)


class FixedReadout(nn.Module):
    """doomfly-style engineered mapping: each action's logit = gain * (sum of named DN activities),
    with left/right as a lateral difference. Only `gain`/`bias` per action are learnable (2 x 5)."""
    DEFAULT = {"jump": ["DNp01", "DNp02", "DNp04", "DNp06", "DNp11"],     # giant-fibre / takeoff-associated
               "slide": ["DNp09", "DNp03", "DNp35"],
               "left": ["DNa02", "DNa03", "DNb01", "DNp20"], "right": ["DNa02", "DNa03", "DNb01", "DNp20"]}

    def __init__(self, graph: Graph, mapping: dict[str, list[str]] | None = None):
        super().__init__()
        mapping = mapping or self.DEFAULT
        idx = dn_indices(graph); nodes = graph.nodes.iloc[idx]
        M = np.zeros((len(ACTIONS), len(idx)), dtype=np.float32)
        for a, types in mapping.items():
            m = nodes["type"].isin(types).to_numpy()
            if a in ("left", "right"):
                side = nodes["side"].to_numpy()
                sgn = np.where(side == ("L" if a == "left" else "R"), 1.0, -1.0)
                M[ACTIONS.index(a), m] = sgn[m]
            else:
                M[ACTIONS.index(a), m] = 1.0
        self.register_buffer("idx", torch.as_tensor(idx))
        self.register_buffer("M", torch.as_tensor(M))
        self.gain = nn.Parameter(torch.ones(len(ACTIONS)))
        self.bias = nn.Parameter(torch.zeros(len(ACTIONS)))
        self.value = nn.Linear(len(idx), 1)

    def forward(self, rates: Tensor) -> tuple[Tensor, Tensor]:
        dn = rates[:, self.idx]
        return (dn @ self.M.T) * self.gain + self.bias, self.value(dn).squeeze(-1)
