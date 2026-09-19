"""Command-line entry point: `flybrain <subcommand>`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flybrain", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench-backend", help="Benchmark CPU vs MPS vs MLX on the recurrent rate update.")
    b.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    b.add_argument("--batch", type=int, default=256)
    b.add_argument("--steps", type=int, default=32, help="BPTT length (decision steps)")
    b.add_argument("--out", default=None, help="Write a markdown table to this path")

    d = sub.add_parser("download", help="Fetch the MaleCNS v1.0 bulk files (public, ~1.2 GB) and record checksums.")
    d.add_argument("--config", default="configs/subgraph.v0.json")
    d.add_argument("--raw-dir", default="data/raw")

    g = sub.add_parser("build-graph", help="Build the versioned task-subgraph artifact from the bulk files.")
    g.add_argument("--config", default="configs/subgraph.v0.json")
    g.add_argument("--raw-dir", default="data/raw")
    g.add_argument("--out", default="data/graph")

    r = sub.add_parser("graph-report", help="Summarise a built graph artifact.")
    r.add_argument("path", nargs="?", default="data/graph/v0")

    c = sub.add_parser("calibrate", help="Sweep input gain x synapse scale on dumped game states; write calibration.json.")
    c.add_argument("--graph", default="data/graph/v0")
    c.add_argument("--cache", default="data/cache")
    c.add_argument("--dynamics", default="rate", choices=["rate", "lif"])
    c.add_argument("--per-neuron", action="store_true", help="use the per-neuron graph instead of the pooled one")
    c.add_argument("--device", default=None)
    c.add_argument("--control", choices=["none", "shuffled", "random"], default="none", help="calibrate a control graph instead")
    c.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("bench-substrate", help="M3 benchmark: forward+backward through the real substrate.")
    s.add_argument("--graph", default="data/graph/v0")
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--steps", type=int, default=32)
    s.add_argument("--device", default=None)

    t = sub.add_parser("train", help="Train an agent (design §6 stages).")
    t.add_argument("--name", required=True)
    t.add_argument("--agent", choices=["mlp", "fly"], default="mlp")
    t.add_argument("--algo", choices=["ppo", "es"], default="ppo")
    t.add_argument("--graph", choices=["pooled", "neuron"], default="pooled")
    t.add_argument("--graph-dir", default="data/graph/v0")
    t.add_argument("--stage", choices=["readout", "full"], default="readout", help="readout: substrate frozen; full: substrate free params trained")
    t.add_argument("--control", choices=["none", "shuffled", "random", "silenced"], default="none")
    t.add_argument("--dynamics", choices=["rate", "lif"], default="rate")
    t.add_argument("--readout", choices=["linear", "fixed", "mlp"], default="linear")
    t.add_argument("--envs", type=int, default=256)
    t.add_argument("--rollout", type=int, default=32)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatches", type=int, default=4, help="per-neuron graph: use --epochs 1 --minibatches 1 (each recurrent minibatch re-runs the whole rollout)")
    t.add_argument("--steps", type=int, default=2_000_000, help="PPO: env decisions; ES: generations")
    t.add_argument("--minutes", type=float, default=None)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--ent-coef", type=float, default=0.01)
    t.add_argument("--scale-max", type=float, default=None, help="clamp substrate synapse scales (Stage C stability)")
    t.add_argument("--noop-bias", type=float, default=3.0, help="initial no-op logit bias (action prior; 3.0 ≈ 83%% no-op)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.add_argument("--runs", default="runs")
    t.add_argument("--dt-ms", type=float, default=None, help="integration step (default 5 ms)")
    t.add_argument("--substeps", type=int, default=None, help="sub-steps per decision (default 13)")
    t.add_argument("--init", default=None, help="runs/<name> to initialise weights from (continue training)")

    e = sub.add_parser("eval", help="Evaluate a run or a scripted baseline on held-out seeds.")
    e.add_argument("--run", default=None, help="runs/<name>")
    e.add_argument("--agent", default=None, help="heuristic | random (if no --run)")
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--max-seconds", type=float, default=180.0)
    e.add_argument("--device", default=None)
    e.add_argument("--out", default=None)

    x = sub.add_parser("export", help="Export a run to a JSON bundle for the browser demo (game/public/agents).")
    x.add_argument("--run", required=True)
    x.add_argument("--out", default=None, help="default: ../game/public/agents/<run>.json")

    di = sub.add_parser("distill", help="Imitate a trained run (teacher) with a fly agent whose substrate stays frozen; then PPO from it with --init.")
    di.add_argument("--name", required=True)
    di.add_argument("--teacher", required=True, help="runs/<mlp run>")
    di.add_argument("--graph", choices=["pooled", "neuron"], default="pooled")
    di.add_argument("--graph-dir", default="data/graph/v0")
    di.add_argument("--readout", choices=["linear", "mlp"], default="linear")
    di.add_argument("--control", choices=["none", "shuffled", "random", "silenced"], default="none")
    di.add_argument("--dynamics", choices=["rate", "lif"], default="rate")
    di.add_argument("--stage", choices=["readout", "full"], default="readout")
    di.add_argument("--envs", type=int, default=256)
    di.add_argument("--decisions", type=int, default=100_000, help="teacher decisions to collect")
    di.add_argument("--epochs", type=int, default=10)
    di.add_argument("--lr", type=float, default=1e-3)
    di.add_argument("--seed", type=int, default=0)
    di.add_argument("--noop-bias", type=float, default=0.0)
    di.add_argument("--device", default=None)
    di.add_argument("--runs", default="runs")
    di.add_argument("--dt-ms", type=float, default=None, help="integration step (default 5 ms)")
    di.add_argument("--substeps", type=int, default=None, help="sub-steps per decision (default 13)")
    di.add_argument("--dagger", type=int, default=0, help="number of DAgger iterations after the initial behaviour-cloning fit")
    di.add_argument("--dagger-decisions", type=int, default=150_000, help="student-visited states added per iteration")
    di.add_argument("--dagger-epochs", type=int, default=6)
    di.add_argument("--beta", type=float, default=0.3, help="prob. an env is driven by the teacher during DAgger collection")
    di.add_argument("--max-decisions", type=int, default=900_000, help="dataset cap (oldest dropped)")
    di.add_argument("--minutes", type=float, default=None, help="wall-clock budget; best-so-far checkpoint is kept")
    di.add_argument("--init", default=None, help="runs/<name> to initialise the student from")

    args = parser.parse_args(argv)
    if args.cmd == "distill":
        from flybrain.train.distill import distill
        distill(args); return 0
    if args.cmd == "export":
        from flybrain.train.export import export
        export(Path(args.run), Path(args.out) if args.out else Path("../game/public/agents") / f"{Path(args.run).name}.json")
        return 0
    if args.cmd == "train":
        from flybrain.train.run import train
        train(args); return 0
    if args.cmd == "eval":
        from flybrain.train.evaluate import run
        from flybrain.train.run import load_run
        import torch
        device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.run:
            pol, meta = load_run(Path(args.run), device)
            run(Path(args.run).name, args.episodes, args.max_seconds, device, policy=pol, out=Path(args.out) if args.out else Path(args.run) / "eval.json")
        else:
            run(args.agent, args.episodes, args.max_seconds, device, out=Path(args.out) if args.out else None)
        return 0
    if args.cmd == "calibrate":
        from flybrain.model.calibrate import run
        run(Path(args.graph), Path(args.cache), dynamics=args.dynamics, pooled=not args.per_neuron, device=args.device, control=args.control, seed=args.seed)
        return 0
    if args.cmd == "bench-substrate":
        from flybrain.bench.substrate import run
        return run(Path(args.graph), batch=args.batch, T=args.steps, device=args.device)
    if args.cmd == "bench-backend":
        from flybrain.bench.backend import run
        return run(sizes=args.sizes, batch=args.batch, steps=args.steps, out=args.out)
    if args.cmd == "download":
        from flybrain.pipeline.download import run
        run(json.loads(Path(args.config).read_text()), Path(args.raw_dir))
        return 0
    if args.cmd == "build-graph":
        from flybrain.pipeline.build import build
        from flybrain.pipeline.download import run as dl
        cfg = json.loads(Path(args.config).read_text())
        raw = dl(cfg, Path(args.raw_dir))
        build(cfg, Path(args.raw_dir), Path(args.out), raw_manifest=raw)
        return 0
    if args.cmd == "graph-report":
        from flybrain.pipeline.report import run
        return run(Path(args.path))
    return 2


if __name__ == "__main__":
    sys.exit(main())
