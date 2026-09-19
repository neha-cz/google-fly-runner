"""Benchmark the actual training workload on CPU, MPS and MLX.

Workload: a batch of B independent rate networks with N nodes and ~10 incoming edges per node
(matching the pruned-connectome density target), integrated for T decision steps with the
flyvis-style update  v <- v + dt/tau * (-v + b + W r),  r = relu(v),  then a scalar loss and a
backward pass through all T steps (truncated BPTT). Dense W is used for N <= 5000 (the design
target); sparse for larger N to show the cliff. Numbers go into docs/bench-backend.md.
"""
from __future__ import annotations

import time
import warnings

import numpy as np


def _make_graph(n: int, k: int, rng: np.random.Generator):
    rows = np.repeat(np.arange(n), k)
    cols = rng.integers(0, n, size=n * k)
    sign = np.where(rng.random(n * k) < 0.7, 1.0, -1.0)
    w = (rng.random(n * k) * 0.2 / k * sign).astype(np.float32)
    return rows, cols, w


def _bench_torch(device: str, n: int, batch: int, steps: int, rows, cols, wvals, reps=3) -> float | None:
    import torch

    dense = n <= 5000
    if dense:
        W = torch.zeros(n, n, dtype=torch.float32)
        W[rows, cols] = torch.from_numpy(wvals)
        W = W.to(device)
    else:
        if device == "mps":
            return None  # sparse matmul on MPS is not reliably supported; documented cliff
        W = torch.sparse_coo_tensor(np.stack([rows, cols]), wvals, (n, n)).coalesce().to(device)
    scale = torch.ones(n, device=device, requires_grad=True)
    tau_inv = torch.full((n,), 0.1, device=device, requires_grad=True)
    b = torch.zeros(n, device=device, requires_grad=True)
    x = torch.randn(batch, n, device=device) * 0.1

    def run():
        v = torch.zeros(batch, n, device=device)
        for _ in range(steps):
            r = torch.relu(v)
            inp = (r * scale) @ W.T if dense else torch.sparse.mm(W, (r * scale).T).T
            v = v + tau_inv * (-v + b + inp + x)
        loss = torch.relu(v).mean()
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()

    run()  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    return (time.perf_counter() - t0) / reps


def _bench_mlx(n: int, batch: int, steps: int, rows, cols, wvals, reps=3) -> float | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    if n > 5000:
        return None  # no sparse matmul in MLX yet; same cliff as MPS
    Wn = np.zeros((n, n), dtype=np.float32)
    Wn[rows, cols] = wvals
    W = mx.array(Wn)
    x = mx.random.normal((batch, n)) * 0.1
    params = {"scale": mx.ones((n,)), "tau_inv": mx.full((n,), 0.1), "b": mx.zeros((n,))}

    def loss_fn(p):
        v = mx.zeros((batch, n))
        for _ in range(steps):
            r = mx.maximum(v, 0)
            v = v + p["tau_inv"] * (-v + p["b"] + (r * p["scale"]) @ W.T + x)
        return mx.maximum(v, 0).mean()

    grad_fn = mx.grad(loss_fn)

    def run():
        g = grad_fn(params)
        mx.eval(g)

    run()
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    return (time.perf_counter() - t0) / reps


def run(sizes: list[int], batch: int, steps: int, out: str | None) -> int:
    import torch

    rng = np.random.default_rng(0)
    mps_ok = torch.backends.mps.is_available()
    if not mps_ok:
        warnings.warn("torch MPS backend not available on this machine/build; MPS column will be empty.")
    rows_out = []
    print(f"batch={batch} steps={steps} torch={torch.__version__} mps={mps_ok}")
    for n in sizes:
        rows, cols, w = _make_graph(n, 10, rng)
        cpu = _bench_torch("cpu", n, batch, steps, rows, cols, w)
        mps = _bench_torch("mps", n, batch, steps, rows, cols, w) if mps_ok else None
        mlx = _bench_mlx(n, batch, steps, rows, cols, w)
        fmt = lambda t: "n/a" if t is None else f"{t*1000:.0f} ms ({batch*steps/t:,.0f} step·env/s)"
        line = f"| {n:>6} | {fmt(cpu)} | {fmt(mps)} | {fmt(mlx)} |"
        print(line)
        rows_out.append(line)
    table = "\n".join(
        ["| N nodes | torch CPU | torch MPS | MLX |", "|---|---|---|---|", *rows_out]
    )
    if out:
        with open(out, "w") as f:
            f.write(f"# Backend benchmark\n\nbatch={batch}, BPTT steps={steps}, ~10 in-edges/node, "
                    f"forward+backward per call, mean of 3 (torch {torch.__version__}).\n\n{table}\n")
        print("wrote", out)
    return 0
