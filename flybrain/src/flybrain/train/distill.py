"""Distillation: train a fly agent's encoder (+ readout, + optionally substrate) to imitate a
teacher policy's greedy actions along the teacher's own trajectories, with the substrate state
carried sequentially (truncated BPTT over 32-step chunks). Class-weighted cross-entropy because the
teacher is mostly no-op. Answers: can the frozen circuit carry a policy that works at all?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flybrain.env.vec import VecFlyRunner
from .run import build_policy, load_run
from .evaluate import run as evaluate


@torch.no_grad()
def collect(teacher, n_envs: int, decisions: int, device: str, seed: int, student=None, beta: float = 1.0):
    """Roll out a mixture policy (teacher with prob beta, else the student) and label every state
    with the teacher's greedy action. beta=1 is plain behaviour cloning; beta<1 is DAgger data."""
    env = VecFlyRunner(n_envs, seed=seed); obs = torch.as_tensor(env.reset(seed), device=device)
    st_t = teacher.initial_state(n_envs); st_s = student.initial_state(n_envs) if student is not None else None
    T = decisions // n_envs
    O = torch.zeros(T, n_envs, 36, device=device); A = torch.zeros(T, n_envs, dtype=torch.long, device=device); D = torch.zeros(T, n_envs, device=device)
    use_teacher = torch.rand(n_envs, device=device) < beta          # per-env, fixed for the rollout
    dist = []
    for t in range(T):
        lg_t, _, st_t = teacher(obs, st_t); a_t = lg_t.argmax(-1)
        a = a_t
        if student is not None:
            lg_s, _, st_s = student(obs, st_s); a = torch.where(use_teacher, a_t, lg_s.argmax(-1))
        O[t] = obs; A[t] = a_t
        nobs, r, d, info = env.step(a.cpu().numpy()); D[t] = torch.as_tensor(d, device=device, dtype=torch.float32)
        dist += info["fin_distance"].tolist()
        obs = torch.as_tensor(nobs, device=device)
        st_t = teacher.mask_state(st_t, 1 - D[t]) if st_t is not None else None
        if st_s is not None: st_s = student.mask_state(st_s, 1 - D[t])
    return O, A, D, (float(np.mean(dist)) if dist else float("nan"))


def distill(args) -> dict:
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path(args.runs) / args.name; run_dir.mkdir(parents=True, exist_ok=True)
    teacher, tmeta = load_run(Path(args.teacher), device)
    args.agent = "fly"
    student, meta = build_policy(args, device)
    if getattr(args, "init", None):
        student.load_state_dict(torch.load(Path(args.init) / "model.pt", map_location=device)["policy"], strict=False); meta["init"] = args.init
    meta.update({"algo": "distill", "teacher": args.teacher, "device": device, "args": vars(args)})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1, default=str))
    t0 = time.time()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    log = open(run_dir / "metrics.jsonl", "a")
    chunk = 32
    O, A, D, _ = collect(teacher, args.envs, args.decisions, device, args.seed)
    print(f"collected {A.numel():,} teacher decisions in {time.time()-t0:.0f}s")
    best = -1.0
    n_iters = 1 + args.dagger
    for it in range(n_iters):
        if it > 0:                                                    # DAgger: add student-visited states, teacher labels
            O2, A2, D2, d_mix = collect(teacher, args.envs, args.dagger_decisions, device, args.seed + 1000 * it, student=student, beta=args.beta)
            O = torch.cat([O, O2]); A = torch.cat([A, A2]); D = torch.cat([D, D2])
            cap = args.max_decisions // args.envs
            if O.shape[0] > cap: O, A, D = O[-cap:], A[-cap:], D[-cap:]
            print(f"[{(time.time()-t0)/60:5.1f}m] dagger iter {it}: +{A2.numel():,} states from a beta={args.beta} mixture (mixture mean distance {d_mix:.0f} m); dataset {A.numel():,}")
        freq = torch.bincount(A.flatten(), minlength=5).float() / A.numel()
        wts = (1 / (freq + 1e-3)).clamp(max=50); wts = wts / wts.mean()
        T, N = A.shape
        for ep in range(args.epochs if it == 0 else args.dagger_epochs):
            st = student.initial_state(N); tot = 0.0; correct = 0; n = 0; correct_nn = 0; n_nn = 0
            for t0_ in range(0, T - chunk + 1, chunk):
                st = st.detach(); logits_l = []
                for t in range(t0_, t0_ + chunk):
                    lg, _, st = student(O[t], st); logits_l.append(lg); st = student.mask_state(st, 1 - D[t])
                logits = torch.stack(logits_l); a = A[t0_:t0_ + chunk]
                loss = F.cross_entropy(logits.reshape(-1, 5), a.reshape(-1), weight=wts)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                if hasattr(student, "clamp"): student.clamp(8.0)
                with torch.no_grad():
                    pred = logits.argmax(-1); tot += loss.item(); correct += (pred == a).sum().item(); n += a.numel()
                    nn_ = a != 0; correct_nn += (pred[nn_] == a[nn_]).sum().item(); n_nn += nn_.sum().item()
            rec = {"iter": it, "epoch": ep, "minutes": round((time.time() - t0) / 60, 2), "loss": tot / max(1, T // chunk), "acc": correct / n, "acc_non_noop": correct_nn / max(1, n_nn), "dataset": int(A.numel())}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{rec['minutes']:5.1f}m] iter {it} epoch {ep:3d} loss {rec['loss']:.3f} acc {rec['acc']:.3f} non-noop acc {rec['acc_non_noop']:.3f}")
            if args.minutes is not None and (time.time() - t0) / 60 > args.minutes: break
        s_eval = evaluate(f"{args.name}/iter{it}", 100, 180.0, device, policy=student)
        log.write(json.dumps({"iter": it, "eval_distance_mean": s_eval["distance_mean"], "eval_survive_cap": s_eval["survive_cap"]}) + "\n"); log.flush()
        if s_eval["distance_mean"] > best:
            best = s_eval["distance_mean"]; torch.save({"policy": student.state_dict(), "cfg": {}}, run_dir / "model.pt")
        if args.minutes is not None and (time.time() - t0) / 60 > args.minutes: break
    student.load_state_dict(torch.load(run_dir / "model.pt", map_location=device)["policy"])
    s = evaluate(args.name, 200, 180.0, device, policy=student, out=run_dir / "eval.json")
    return s
