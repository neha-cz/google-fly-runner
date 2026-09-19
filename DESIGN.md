# FlyRunner — Design Doc

_Status: draft for review, 2026-09-14. Sections marked **DECISION** are the ones I want sign-off on
before Phase 2+ implementation. Phase 0/1 (scaffold + game) are built in parallel with this doc
because they don't depend on any of the open decisions._

## 0. One-paragraph framing

We fix a neural network's **topology** to a task-relevant subgraph of the MaleCNS v1.0 connectome
(nodes = neurons, edges = synaptic connections with synapse counts and predicted signs), leave a
small set of biologically-unmeasured parameters free (per-synapse-class weight scale, per-cell-type
time constant and bias, plus a small readout), and optimize those with RL until the network plays a
3-lane endless runner. This is a legitimate "connectome-constrained controller" experiment in the
flyvis tradition. It is **not** a simulation of a fly's mind, and nothing about its game score says
anything about real fly cognition. The README carries this disclaimer verbatim.

## 1. Facts verified today (2026-09-14)

| Item | Verified value | Source |
|---|---|---|
| MaleCNS release | **v1.0** (released 2026-06-08), license **CC BY** | male-cns.janelia.org |
| Credits required | FlyEM (HHMI Janelia), Univ. of Cambridge Dept. of Zoology, MRC LMB, Google Research | male-cns.janelia.org |
| neuPrint access | `Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=…)`; token from a neuPrint account, read from `NEUPRINT_APPLICATION_CREDENTIALS` | male-cns.janelia.org/download |
| Bulk downloads | Feather: body annotations (13 MB), NT predictions (42 MB), connectome weights (1.1 GB), synaptic partners (6.8 GB) | male-cns.janelia.org/download |
| Full-graph scale | doomfly reports 166,700 neurons / 25,582,938 directed connections retained from v1.0; MaleCNS DN paper reports **1,314 descending neurons**. Exact official totals are re-checked by the pipeline's manifest step, not hardcoded. | doomfly README; bioRxiv 2025.09.04.674108 |
| flyvis | v1.2.0, MIT, PyTorch, **`requires-python >=3.9,<3.13`** | PyPI |
| Toolchain resolution | On Python **3.12**: torch 2.14.0, mlx 0.32.2, flyvis 1.2.0, neuprint-python 0.6.3, gymnasium 1.3.0 all resolve together. On the system Python 3.14 flyvis does **not** install. | `uv pip install --dry-run` |
| Host | Apple M5, 32 GB unified, macOS 25.5 (Darwin), Node 24, uv present | local |

Consequence: `flybrain/` pins `requires-python = ">=3.12,<3.13"` and uv manages the interpreter.

## 2. Prior art and what we take from each

- **doomfly** (MaleCNS → ViZDoom, full 166k-neuron graph, fixed DN→button map, dopamine-gated
  KC→MBON plasticity). Reports honestly that its v6 candidate **failed** its visual, conditioning
  and survival gates, and that "changing weights and longer individual rounds do not establish
  learning". Lessons: (a) the full graph + local plasticity on one pathway is too weak a learning
  signal; (b) their validation-gate structure (visual gate, conditioning gate, survival gate) is
  worth copying as our evaluation discipline.
- **Fly Dino / flyjump** (80 MaleCNS cells, 1,296 edges, signed leaky-tanh rate dynamics, 16
  DN activities → 16-12-3 readout, only the 243 readout params trained by CEM). Result: 99/100
  held-out courses; silencing the circuit → 0/100. This is the existence proof that "small circuit
  + trained readout" works on a runner-like game. Their own caveat, which we adopt: it shows
  *learned dependence on circuit activity*, not *superiority of biological topology*. We go one
  step further than they did by (i) also training the connectome's free parameters, not just the
  readout, and (ii) running a topology-matched shuffled-graph control so we can actually test the
  topology question.
- **fly-craftax** (connectome + linear DN readout trained with PPO in JAX; calibrates lamina bias
  and synapse weight scale first). Lesson: calibrate the substrate to be *active but not saturated*
  before RL, or PPO gets no gradient. We add an explicit calibration milestone.
- **flyvis** (Lappalainen et al., Nature 2024): FIB-25-derived optic-lobe model, 64 cell types,
  hexagonal lattice of columns, dynamics `tau_i dV_i/dt = -V_i + b_i + Σ_j W_ij r_j`,
  `r = ReLU(V)`, `W_ij = sign(type_j) · scale(type_j→type_i, offset) · syncount_ij`. Free:
  per-type `tau`, per-type bias `b`, per-synapse-class `scale`. Fixed: connectivity and sign.
  We reuse this parametrization **exactly** for our substrate, and reuse flyvis's pretrained
  optic-lobe network as an optional visual front end (see §6.3).

## 3. The game (Phase 1) — built now

**Name:** FlyRunner. **Theme:** temple-ruin path through jungle, explorer character, chase camera
(the classic endless-runner look, per the 2026-09-14 review). three.js with procedural geometry and
canvas-painted textures; no external assets, names or trademarks. A 2D canvas fallback is kept.

Note on the reference repo: `priyanksharma7/temple-run` does not embed Temple Run — its iframe
points at a third-party clone ("Tomb Runner") on gamedistribution.com, cross-origin, driven only
by OS-level `pyautogui` keypresses with nothing read back. Embedding that would remove `getState`,
headless `step`, `reset(seed)` and vectorisation, i.e. the whole training path, so we restyled our
own engine instead.

**Mechanics.** 3 lanes; obstacles are `LOW` (jump over: fallen log), `HIGH` (slide under: spiked
stone beam), `FULL` (change lane: carved stone block); pickups are gold coins. Speed ramps with
distance. Actions: `jump`, `slide`, `moveLeft`, `moveRight`, `noop`. Jump and slide are
fixed-duration latched states; lane changes tween over 0.2 s with a lean ("turn") animation.
Track turns (T-junctions) are not modelled; the path is straight.

**Engine is the important deliverable**, not the art:

- Pure TypeScript, zero DOM dependency, fixed timestep `dt = 1/60`, seeded xorshift PRNG
  (uint32 arithmetic — trivially reproducible in NumPy).
- `RunnerAPI`: `jump() slide() moveLeft() moveRight() step(dt) getState() getFrame() reset(seed)`.
- `getState()` is a fixed-length `Float32Array` (layout in `game/src/engine/state.ts` and README):
  lane one-hot, lateral offset, vertical phase, slide phase, speed, and for each lane the next 2
  obstacles as (type one-hot, normalized distance) plus the nearest pickup distance.
- `getFrame()` is a **software rasterizer** (`Uint8Array`, default 64×48 grayscale, pseudo-3D
  projection of the same lanes/obstacles) that runs identically in Node and the browser — this is
  the retinal input for the visual-front-end variant. The pretty canvas renderer is for humans.
- `VecRunner` runs N engines in one loop for a headless throughput benchmark.
- Human keyboard mode, a random agent, and a scripted heuristic agent (proves the game is solvable
  through the API and is itself a baseline).

**DECISION: two engines, one spec** (accepted). RL training needs thousands of env steps/s inside
the Python process; JS↔Python IPC per tick would dominate. So Phase 4 ports the engine to a
vectorized NumPy implementation. The TypeScript engine is the source of truth; a committed
**golden-trajectory fixture** (seed → sequence of states/rewards under a fixed action script) is
checked by both test suites, so the two engines can't silently drift. The engine is deliberately
kept small (a few hundred lines) to make this port cheap.

## 4. Subgraph selection (Phase 2) — built 2026-09-14

**DECISION: start type-pooled, then per-neuron for the same cell-type list** (accepted).

**Data source.** The MaleCNS v1.0 flat-connectome bulk files on the public GCS bucket
(`body-annotations`, `body-neurotransmitters`, `connectome-weights`; ~1.2 GB, CC BY 4.0), not
neuPrint. Same data, but versioned URLs + local sha256 make a build reproducible without a per-user
token. `flybrain download` records size/sha256 in `data/raw/manifest.json`; `flybrain build-graph`
copies that into the artifact manifest. neuPrint remains useful for ad-hoc queries only.

**Selection is by cell-type membership, then structural pruning; never by hand-picked body IDs.**
Every step and its counts are in `data/graph/v0/manifest.json`.

1. **Cell-type allowlist** (`flybrain/configs/subgraph.v0.json`, regexes over the release's `type`
   column; an unmatched pattern fails the build). Resolved names differ from hemibrain/FlyWire
   conventions in places: photoreceptors are `R1-R6`, `R7{d,p,y}`, `R8{d,p,y}` (`_unclear` variants
   excluded); PEN is `PEN_a(PEN1)`/`PEN_b(PEN2)`; vDelta types have `_a/_b` splits.
   - Early visual: R1-R6, R7, R8, L1–L5, C2, C3, Mi1, Mi4, Mi9, Tm1–4, Tm9, T4a–d, T5a–d.
   - Looming: LC4, LC6, LPLC1, LPLC2, LC11. Optic flow: HSE/HSN/HSS, VS.
   - **Anterior visual pathway: MeTu, TuBu, ring neurons ER1–6/ExR/EL** — added after the first
     build, see below.
   - Central complex: EPG, PEN, Δ7, hΔA–M, vΔA–M, PFL2, PFL3.
   - Descending: DNp01 (giant fibre), DNp02/03/04/06/11/35/103 (looming-targeted), DNp09,
     DNp15/17/20 (HS/VS-targeted), DNa01/02/03, DNb01 (PFL3-targeted), MDN.
2. **Status:** `Traced` only, except photoreceptors (many useful fragments lack a status).
3. **Edge threshold:** synapse count ≥ 5.
4. **Column assignment:** only 15 optic-lobe types carry `assignedOlHex*` coordinates; T4/T5,
   photoreceptors and MeTu do not. Each columnar neuron without one inherits the column (and, if
   missing, the side) of its strongest same-side synaptic partner that has one (≤ 3 rounds;
   `hex_source` records which). 25,660 assigned this way; 628 columnar cells stayed unassigned and
   were dropped.
5. **Hex patch:** columnar cells within hex radius 3 of the per-side median column (centre (19, 21)
   both sides; 37 columns). Non-columnar types are kept whole.
6. **Reachability prune:** keep nodes that are both reachable from an input node and can reach an
   output node (BFS both ways on the thresholded subgraph). Input layer = photoreceptors **and
   lamina**, output layer = descending neurons.
7. **Sign** from `consensus_nt`: ACh +1; GABA, Glu, histamine −1; dopamine/serotonin/octopamine 0
   (unsigned → sign-free learnable scale). Convention is in the manifest.
8. **Cap:** 5,000 neurons (hard error).

**What the first build taught us** (all from the release's own connectivity, not guessed):
- With only the original list, the prune removed 785/800 central-complex cells, all HS/VS and 7/20
  DNs: the visual → CX route runs through MeTu → TuBu → ring neurons, HS/VS project to DNp20/15/17,
  looming LC/LPLC cells also target DNp03/35/103, and PFL3's strongest DN targets are DNa03 and
  DNb01. Adding those reconnected everything (only 8 CX cells now pruned).
- R1-R6 are annotated in only 299 (L) / 521 (R) of ~880 columns per eye in v1.0 and almost none near
  the medulla centre, so a central patch holds few of them (22). Hence lamina cells are part of the
  input layer — the same place flyvis's retina model injects.

**Result (`data/graph/v0`, committed, 716 KB):**

| | per-neuron | (type, side)-pooled |
|---|---|---|
| nodes | 3,678 | 261 |
| edges | 69,127 (1.22 M synapses) | 3,443 |
| dense fp32 weights | 54 MB | < 1 MB |

Per-type counts, sides, degrees and NT are printed by `flybrain graph-report`. Build takes 12 s
with 3.7 GB peak RSS (the memory-mapped weights file); nothing dense is ever built over the full
graph.

**Memory note.** A dense weight matrix over the full 166k-neuron graph would be ~111 GB — never do
that. The full 152 M-row weights table is only ever touched memory-mapped and filtered with pyarrow.

## 5. Model (Phase 3) — built 2026-09-14

Code: `flybrain/src/flybrain/model/`. Tests: `tests/test_model.py` (run against the real v0 artifact).

- **Dynamics** (`substrate.py`): flyvis form, `τ_i dV_i/dt = −V_i + b_i + Σ_j W_ij r_j + x_i`,
  `r = relu(V)`, Euler at 5 ms, 13 sub-steps per 15 Hz decision window. The game runs at 60 Hz and
  the agent acts every 4 ticks; the network integrates continuously across them and the readout
  sees the window-mean DN rate.
- **Frozen:** edge set, synapse counts, and sign (from `consensus_nt`). **Free:** per-type
  `log τ` (clamped 5–200 ms), per-type bias, per-synapse-class scale (class = (pre type, post
  type)). Signed classes use `|scale|·sign` so a sign can never flip; the 3,206 edges from
  modulatory types (octopamine EL, dopamine ExR2, serotonin ExR3) get a free-signed scale.
  `W_ij = sign·scale[class]·syn_ij / S`, `S` = mean total input synapses per node. Pooled graph
  uses synapses-per-postsynaptic-neuron as `syn_ij`. Free-parameter counts: pooled 261-node graph
  → 118 types × 2 + 1,041 classes; per-neuron → same types, classes from the 69k edges.
  `weight_matrix()` is rebuilt each window by `index_put` so gradients reach `scale`.
- **Spiking** (`dynamics="lif"`): threshold 1, reset to 0, Heaviside with fast-sigmoid surrogate;
  the reported rate is the spike fraction over the window. Same interface, same tests.
- **Input encoders** (`io.py`): `StateEncoder` — learned linear map from the 36-float state onto the
  input layer (photoreceptors + lamina), rectified drive. `FrameEncoder` — fixed retinotopic sampling:
  each columnar input node reads the mean luminance of the frame region nearest its optic-lobe
  column (hex axial coords → left/right image halves), as contrast times a per-type learnable gain;
  nodes in the same column share the mask. Per-neuron graph only.
- **Readouts:** `LinearReadout` (DN window-mean rates → 5 action logits + value; primary) and
  `FixedReadout` (doomfly-style engineered map: jump ← DNp01/02/04/06/11, slide ← DNp09/03/35,
  left/right ← lateral difference of DNa02/DNa03/DNb01/DNp20; only per-action gain/bias train).
- **Controls** (`controls.py`), produced from the real graph by the same loader: `shuffled`
  (degree-preserving double-edge swaps; in/out degree of every node exactly kept, tested),
  `random` (same n, m; uniform pairs; synapse counts permuted), `silenced` (`input_gain = 0`),
  readout-only (`trainable = False`; substrate parameters get no gradient, tested).

**M3 benchmark** (`flybrain bench-substrate`, MPS, batch 256, 32 windows × 13 sub-steps, forward +
backward, torch 2.14):

| graph | dynamics | n | ms / rollout | env-decisions/s |
|---|---|---|---|---|
| pooled | rate | 261 | 160 | 51,130 |
| pooled | LIF | 261 | 242 | 33,897 |
| per-neuron | rate | 3,678 | 5,281 | 1,551 |
| per-neuron | LIF | 3,678 | 5,662 | 1,447 |

The per-neuron figure came from 655/s: MPS autograd through 416 recurrent sub-steps was 3× slower
than the FLOPs justify, and casting only the recurrent matmul operands to bf16 (state stays fp32)
gave 5.7× on the isolated window and 2.4× end to end (`matmul_bf16`, default on GPU; pre-transposing
W actually hurt). Pooled is comfortably fast for PPO; per-neuron is usable for the readout-only
stage and for the final experiment at smaller batch, and two further levers are known: fewer
sub-steps (7 × 10 ms ≈ 1.5× more) and a smaller batch with more parallel time.

**Calibration** (`flybrain calibrate`): sweeps input gain × scale init on 20k real game states
dumped from the TypeScript engine (`npm run dump-states`, mixed random/heuristic play), 64 windows
of 30 decisions, and picks the setting with active fraction in [5%, 50%], bounded finite rates and
the largest DN variance across the batch. Results in `data/graph/v0/calibration_*.json`.

### 5.1 Calibration results (2026-09-14)

| graph | dynamics | chosen (gain, scale) | active | DNs responsive | DN batch-std | stable region |
|---|---|---|---|---|---|---|
| pooled | rate | (3, 5) | 37% | 81% (13/16 pool nodes) | 2.4 | diverges at scale 10 |
| per-neuron | rate | (0.1, 1.5) | 50% | 62% (24/39 cells) | 0.003 | grows without bound from scale 2, diverges at 3 |
| pooled | LIF | (3, 100) | 36% | 81% | 0.18 | scale ≥ 30 needed; saturates (rates → 1) by scale 300 |

LIF needed a ~20× larger synapse scale than the rate model because a single presynaptic spike
delivers only `α·scale·syn/S` against a threshold of 1; its stable, responsive region is
scale 30–300, and unlike the rate model it saturates rather than diverging.

Three things this pinned down:

- **The per-neuron graph sits close to its recurrent stability boundary.** With synapse-count
  weights normalised to a mean total input of `scale`, the effective spectral radius crosses 1
  between scale 1.5 and 2. A ReLU rate network has no ceiling, so Stage C (training `scale`) needs
  either a clamp on the per-class scale, a spectral/activity penalty, or a saturating nonlinearity
  — this is now a listed Phase 4 risk, not a surprise.
- **Signal does reach the DNs from the input layer in both graphs** (81% / 62% of DN nodes respond
  to the state-driven input) but in the per-neuron graph the DN modulation is small (batch-std
  0.003 on a mean of 0.026) at the largest stable setting: most of the drive is absorbed before the
  DNs. That is the doomfly failure mode in miniature, and exactly why the readout-only stage (M4b)
  exists as a gate before unfreezing the substrate.
- The "pick the largest DN variance" rule mostly picks the largest gain inside the stable region
  because the rate model is linear in the input; the chosen point is a starting point for PPO, not
  a tuned optimum. The encoder and readout are trained anyway.

## 6. RL algorithm (Phase 4) — built 2026-09-14

**DECISION: PPO throughout, with a staged unfreeze; ES as the fallback** (accepted, both built).

Code: `flybrain/src/flybrain/env/` (engine port, vectorised env, scripted agents) and
`flybrain/src/flybrain/train/` (policies, PPO, ES, evaluation, run assembly). Tests:
`tests/test_engine.py` (golden parity), `tests/test_train.py`.

- **Engine port** (`env/engine.py`): statement-for-statement Python port of the TypeScript engine,
  same xorshift32 stream and PRNG call order. `tests/test_engine.py` replays the three golden
  episodes (30 s each, checkpoints every second) and matches distance/score/coins/lane exactly and
  the 36-float state to 2e-6. 132k ticks/s per core with `get_state`, i.e. ~33k decisions/s —
  fast enough that a JS↔Python bridge was never needed.
- **Env** (`env/vec.py`): `VecFlyRunner(N)` steps N engines in lock-step with auto-reset; decisions
  every 4 ticks (15 Hz), action latched on the first tick. Reward per decision = metres + 5·coins
  − 50·crash − 0.05·lane change. 180 s cap (2,700 decisions); truncation is reported separately
  from crashes and PPO bootstraps it with V(final obs). A single-instance `FlyRunner-v0`
  Gymnasium env wraps the same engine.
- **Policies** (`train/policies.py`) share one interface (`initial_state`, `forward(obs, state)`,
  `mask_state`): `MLPPolicy` (2×64 tanh, same 36-in / 5-out shapes as the fly) and `FlyPolicy`
  (encoder → substrate → readout; state = membrane vector; optional clamp on synapse scales).
- **PPO** (`train/ppo.py`): GAE(γ 0.99, λ 0.95), clip 0.2, 4 epochs × 4 minibatches, Adam 3e-4,
  entropy 0.01, grad-clip 0.5. Recurrent policies get truncated BPTT over each 32-step rollout:
  the substrate state at rollout start is stored detached, sequences are re-run from it during the
  update with state reset at episode boundaries, and minibatches are whole environments.
- **ES fallback** (`train/es.py`): OpenAI-style NES (antithetic Gaussian perturbations,
  rank-normalised fitness, Adam) over the same trainable vector, greedy episodes of 60 s.
- **Stages** (`flybrain train --stage`): A `--agent mlp`; B `--agent fly --stage readout`
  (substrate frozen at the calibrated point, encoder + readout train); C `--stage full
  --scale-max 8` (substrate free parameters train under the stability clamp). Controls:
  `--control shuffled|random|silenced`, same code path. Calibration (§5.1) is read automatically
  from `data/graph/v0/calibration_*.json`.
- **Evaluation** (`flybrain eval`): 200 held-out seeds (base 10,000), greedy actions, 180 s cap;
  distance mean/median/p10/p90 and survival-to-cap; JSON next to the run.

### 6.1 First results (2026-09-14, single M5, runs share the CPU)

_Baselines, 200 held-out seeds, 180 s cap (cap distance 3,409 m):_

| agent | distance mean | median | p10 | p90 | survive cap |
|---|---|---|---|---|---|
| random | 36 | 28 | 18 | 69 | 0.00 |
| scripted heuristic @15 Hz | 1,209 | 952 | 666 | 2,060 | 0.01 |

The scripted heuristic reaches 778 m in 60 s at 60 Hz on every seed (TS parity) but at the 15 Hz
decision rate its reaction thresholds become marginal above ~19 m/s — which is why it is only a
baseline.

**First 15-minute runs (before the action prior), 200 held-out seeds, greedy:** MLP-PPO 124 m
(median 103; 10.6 M decisions), readout-only fly 49 m (1.7 M), full fly 34 m (1.3 M). All far
below the heuristic, and the MLP was still nearly uniform over actions (entropy 1.25 of 1.61,
`approx_kl ≈ 1e-4`, `clipfrac = 0` on every update); the readout-only fly collapsed to
"always slide" — which beats random only because sliding clears `HIGH` obstacles.

**Diagnosis** (`runs/` probes, 2026-09-14). Three hypotheses were tested and two rejected:
1. *Starved policy gradient from the unnormalised value loss* — rejected: reward normalisation
   (kept, as it makes the value scale sane) did not change the KL, and Adam is invariant to the
   gradient scaling that grad-clipping introduces.
2. *Learning rate* — rejected: 3× the LR with/without the entropy bonus learned no faster.
3. *Exploration interference* — confirmed. The good policy is **95.8 % no-op** with rare,
   precisely timed actions (that is the heuristic's action distribution; a naive behaviour clone
   just learns "always no-op", 96 % accuracy, 40 m). Under uniform exploration a correct jump is
   almost always undone: `slide` while airborne fast-falls, and over the ~8 decisions a jump lasts
   the chance of a random slide is 1 − 0.8⁸ ≈ 83 %; lane changes likewise cancel each other.
   Measured under a uniform policy, the return-to-go for "jump with a log 1–4 m ahead" (−18) barely
   beats no-op (−24) and ties with dodging sideways — the credit PPO needs is destroyed by the
   agent's own next actions.

**Fix (training side, game rules untouched):** an action prior — the no-op logit is initialised at
+3 (≈ 83 % no-op at the start, matching the heuristic's statistics) in both the MLP head and the
DN readouts (`--noop-bias`), so "act, then hold" sequences are common enough to learn from.
Reward normalisation stays on. Results with the prior:

_Six parallel 30-minute runs (2 CPU threads each, so per-run throughput is 2–4× below the solo
benchmarks), then 200 held-out seeds, greedy actions, 180 s cap (cap distance 3,409 m):_

| agent | training decisions | distance mean | median | p10 | p90 | survive cap |
|---|---|---|---|---|---|---|
| random | – | 36 | 28 | 18 | 69 | 0.00 |
| scripted heuristic @15 Hz | – | 1,209 | 952 | 666 | 2,060 | 0.01 |
| **MLP-PPO (2×64)** | 9.5 M | **955** | 655 | 189 | 2,309 | 0.04 |
| connectome (pooled, 261 nodes), readout-only | 2.2 M | 49 | 40 | 18 | 91 | 0.00 |
| connectome (pooled), substrate trained, scale ≤ 8 | 1.8 M | 55 | 48 | 26 | 90 | 0.00 |
| control: shuffled topology, readout-only | 2.4 M | 37 | 29 | 18 | 65 | 0.00 |
| control: random graph, readout-only | 2.4 M | 40 | 36 | 18 | 71 | 0.00 |
| control: silenced input, readout-only | 2.2 M | 37 | 29 | 18 | 65 | 0.00 |

Sample-matched: the MLP's training-time mean distance at 2.2 M decisions (what the fly runs got)
was 214 m.

_Follow-ups (45 min each, two processes only):_

| agent | training decisions | distance mean | median | p10 | p90 | survive cap |
|---|---|---|---|---|---|---|
| **MLP-PPO continued from the 30-min checkpoint (75 min total)** | 16 M | **1,954** | 2,579 | 159 | 3,409 | **0.47** |
| connectome, per-neuron graph (3,678 cells), readout-only, `--epochs 1 --minibatches 1` | 1.0 M | 37 | 29 | 18 | 65 | 0.00 |

### 6.2 Distillation: can the frozen circuit carry a working policy at all?

Motivated by the null result, `flybrain distill` trains the fly agent's encoder (+ readout, +
optionally the substrate's free parameters) to imitate the 75-minute MLP's greedy actions along the
MLP's own trajectories (class-weighted cross-entropy; truncated BPTT over 32-step chunks; the
substrate state carried sequentially). If imitation fails, the circuit lacks capacity for the
policy; if it succeeds, PPO's difficulty was optimisation. Two fixes came out of it: a per-DN
running standardisation before the readout (`DNNorm`; at the calibrated per-neuron point DN rates
are 0.026 ± 0.003, so every readout was seeing a near-constant input and the first attempt sat at
loss = ln 5), and a Fly Dino-style nonlinear readout (`--readout mlp`, 39 → 32 → 5).

| student | agreement with MLP | non-no-op agreement | distance (100 seeds) |
|---|---|---|---|
| pooled, linear readout | 59 % | 45 % | 38 m |
| pooled, nonlinear readout | 61 % | 45 % | 46 m |
| per-neuron, nonlinear, 8 epochs / 100k | 74 % | 70 % | 45 m |
| per-neuron, nonlinear, 40 epochs / 300k, substrate frozen | 77 % | 76 % | 77 m |
| **per-neuron, nonlinear, 30 epochs / 300k, substrate trainable** | 78 % | 75 % | **111 m** |

So: the pooled graph is a hard capacity ceiling (~60 % agreement whatever the readout); the
per-neuron graph can carry substantially more of the policy (78 % and still climbing slowly at the
end), and letting the connectome's free parameters move helps (111 vs 77 m). But 78 % agreement at
15 Hz compounds into a ~10 s run — roughly 3× random and 1/18 of the teacher. The next step is
PPO from the distilled initialisation (`--init runs/d_neuron_full`), which is the run reported in
§6.3 below.

### 6.3 Speed levers for the per-neuron substrate (measured 2026-09-14)

Per 13-sub-step window, forward + backward, batch 256, 3,678 nodes / 69k edges:

| formulation | MPS | CPU |
|---|---|---|
| dense bf16 matmul (current) | **149 ms** | 3,156 ms |
| edge-wise `index_add_` gather/scatter | 1,232 ms | 3,419 ms |
| `torch.sparse.mm` (COO) | 279 ms | 2,233 ms |
| dense, fused update (`addcmul`, precomputed constants) | 201 ms | – |
| dense, **7 sub-steps of 10 ms** | **~110 ms** | – |

At this size the dense bf16 path wins on MPS by 2–8×; sparse only wins on CPU, which is 20× slower
overall, so the edge-list route is closed on this hardware. Fusing elementwise ops gains nothing
(the cost is MPS kernel-launch/autograd overhead, ~11 ms per sub-step against ~1 ms of matmul).
The only real lever is a coarser integration step: `--dt-ms 10 --substeps 7` is 1.6× faster and
leaves the calibrated per-neuron operating point statistically identical (active 50 %, DN mean
0.026, batch-std 0.003); the minimum time constant becomes 10 ms. Used for the long runs.

**PPO from the distilled initialisation** (`v4_fly_neuron_ppo`: substrate trainable, `scale ≤ 8`,
lr 1e-4, one full-batch pass per rollout, 45 min / 3.4 M decisions): **65 m** on 200 held-out
seeds — *worse* than the 111 m it started from. On-policy fine-tuning of a policy that is right 78 %
of the time drifts it before it improves it; with a −50 crash the early exploration noise dominates.
DAgger (§6.4) is the right tool for this regime, PPO is not.

### 6.4 DAgger (2026-09-14): the first connectome agent that actually reacts

`flybrain distill --dagger N`: after the initial fit, each iteration rolls out a mixture policy
(teacher with probability β = 0.3 per env, else the student), labels every visited state with the
teacher's action, appends 150k decisions to the aggregate (capped at 900k, oldest dropped), trains
5 epochs, evaluates on 100 held-out seeds and keeps the best checkpoint. Per-neuron graph, nonlinear
readout, substrate free parameters trainable (`scale ≤ 8`), initialised from `d_neuron_full`,
10 ms integration, 4-hour budget (`runs/v5_fly_neuron_dagger`).

| iteration | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 (budget) |
|---|---|---|---|---|---|---|---|---|---|
| held-out distance (m) | 87 | 86 | 107 | 138 | 142 | 159 | 143 | 201 | **232** |
| agreement with teacher | 78 % | | | | | | | | 87 % |

**v5 final, 200 held-out seeds: 232 m mean, median 186, p10 38, p90 478.** Continued for another
4 hours from that checkpoint (`v6_fly_neuron_dagger`, lr 3e-4):

| iteration | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 (budget) |
|---|---|---|---|---|---|---|---|---|
| held-out distance (m) | 144 | 214 | 234 | 208 | 239 | 254 | 321 | **337** |

**v6 final, 200 held-out seeds: 317 m mean, median 218, p10 65, p90 739, 0 % to cap** — 2.9× the
best plain-distillation agent, 8.8× random, 1/6 of the teacher, and still improving at the end of
8 hours total. The compounding-error diagnosis was right: the same circuit that PPO could
not train from scratch, and that plain imitation left at 111 m, follows the teacher through
~15 s of play once it is trained on the states it actually visits.

What this does and does not show: the MaleCNS-constrained substrate, with only per-class
synapse scales, per-type time constants/biases, a linear input map and a 39→32→5 readout free,
can carry a nontrivial obstacle-avoidance policy. It does not show that the *specific* wiring
matters — the shuffled/random controls have not been run through DAgger yet, and that comparison
is the one that would answer the topology question.

### 6.5 The topology test (2026-09-15): does the specific wiring matter? — No.

Same DAgger recipe as §6.4, from scratch (no distilled initialisation), on the real per-neuron
graph and its two topology controls, run concurrently so they shared the machine equally (6-hour
budget each, which under 3-way contention came to 6 iterations apiece). Each control graph was
calibrated by the same sweep (`flybrain calibrate --per-neuron --control …`). One seed each.

| graph | iteration evals (100 seeds) | **final (200 seeds)** | median |
|---|---|---|---|
| real MaleCNS wiring (`v7_real_dagger`) | 53, 51, 59, 99, 113, 180 | **188 m** | 164 |
| degree-preserving shuffled (`v7_shuffled_dagger`) | 41, 45, 103, 183, 364, 403 | **423 m** | 306 |
| random graph, same n and m (`v7_random_dagger`) | 53, 76, 162, 253, 372, 451 | **465 m** | 319 |

The controls beat the real connectome by 2.2–2.5×, and their curves were steeper from iteration 2
on. With the sign convention, synapse counts and cell-type structure all kept, the real wiring is a
*harder* substrate to read a policy out of than a matched random one — plausibly because a random
graph of this density is a well-conditioned reservoir while the real optic-lobe → central-complex
→ DN pathway is deep, sparse in effective paths, and dominated by specific inhibitory motifs that
the calibration has to work around (its calibrated DN response was ~30× weaker than the random
graph's is at its own calibrated point).

**Headline for the project, stated plainly:** a MaleCNS-constrained substrate with a small set of
free biophysical parameters *can* be trained (by imitation + DAgger) to play an endless runner
(317 m after 8 h, §6.4), but under matched conditions the specific connectome topology confers no
advantage — the opposite. This agrees with Fly Dino's own caveat ("learned dependence on circuit
activity, not superiority of biological topology") and extends it with the controls doomfly and
Fly Dino did not run. Caveats: one seed per condition; ~6 h of a laptop per run; type-pooled
synapse scales rather than per-synapse weights; the `state` encoder rather than a retinotopic
input. A per-synapse-weight variant or the `frame` encoder could change the ordering, and are
the obvious follow-ups; nothing here says anything about what the real circuit computes in a fly.

The MLP now clearly beats the scripted heuristic (mean 1,954 vs 1,209; 47 % of episodes reach the
180 s cap; median 2,579 m) and is still improving — M4a's 90 % bar looks reachable with a few more
hours, not a different method. The per-neuron connectome agent is a pure no-op policy after 1 M
decisions (identical numbers to the silenced control). That is under-trained rather than
conclusive — the MLP was at ~130 m at 1.2 M — but it is the honest state: at ~800 decisions/s
(PPO with one full-batch pass per rollout; four epochs of minibatches cost 512 dense 3,678² passes
per update and ran at ~100/s) the per-neuron graph needs hours per run.

**Reading it honestly.**
- **M4a (MLP ≥ 90 % survival to cap): not reached in 30 minutes**, but the MLP is now in the
  heuristic's range (mean 955 vs 1,209; p90 2,309 vs 2,060) and its curve had not flattened. Its
  greedy policy never jumps: 53 % slide, 30 % right, 0 % jump. Sliding is free and re-triggerable
  in this game, so "slide continuously, dodge sideways" is a legitimate strategy that clears
  `HIGH` obstacles for free and turns `LOW` ones into lane changes. That is a property of the game
  worth knowing, not a bug.
- **M4b (connectome readout-only beats random and the heuristic; silenced does not): failed on
  the pooled graph.** The pooled agent learned "always slide" (92–97 % slide), which beats random
  (49–55 m vs 36) only because slide is free; the silenced and shuffled controls collapsed to
  pure no-op (identical trajectories, 37 m), the random-graph control found the same slide trick
  partially (40 m). So the pooled connectome agent is separable from a silenced circuit only by
  a trick that needs no perception, and is ~4× behind a sample-matched MLP. **This is doomfly's
  negative result reproduced under controls**, at this scale.
- Why, mechanistically: the pooled graph gives the learner a 36 → 16 (photoreceptor + lamina
  pools) → fixed substrate → 16 (DN pools) → 5 bottleneck. The decisive game rules are
  conjunctions (own lane × obstacle type × distance window); a fixed 16-wide ReLU layer between
  two learned linear maps has almost no capacity to represent them, and the substrate parameters
  (Stage C) only rescale it. The per-neuron graph (645 input cells, 39 DNs, 3,678 units) removes
  that bottleneck, at ~5× the cost per decision — it is the next experiment, running as of this
  write-up (`v3_fly_neuron_ro`), alongside the MLP continued from its checkpoint (`v3_mlp_cont`).
- Training a substrate that sits near its stability boundary worked mechanically (`scale ≤ 8`
  clamp, no divergence, 55 m vs 49 m readout-only) but did not change the picture.

## 6.5 Integration & live demo (Phase 5) — built 2026-09-14

`flybrain export` serialises a trained run to JSON: MLP layers, or for the fly agent the effective
signed edge weights (`sign·|scale|·syn/S`, evaluated once), per-node `α = dt/τ` and bias, encoder,
readout, node names/types/groups, and a self-test block of 8 × 6 reference states with the PyTorch
logits. `game/src/agents/neural.ts` runs either kind in the browser at the same 15 Hz decision
cadence with greedy actions; `npm test` replays the self-tests (max |Δlogit| < 1e-3 on both current
bundles). The demo UI lists bundles from `public/agents/index.json` in the Driver menu, and for
connectome drivers draws a live panel: mean window rate per cell-type group in anatomical order
(R → lamina → medulla → T4/T5 → LC/LPLC, HS/VS → MeTu → TuBu → ER → CX → DN), each DN pool the
readout sees, and the action logits. Human, heuristic, random, MLP and connectome drivers are all
selectable in the same UI, as the brief asked.

## 7. Acceleration path: MLX vs PyTorch/MPS

**DECISION: PyTorch, `mps` backend, dense weight matrices at ≤ 5k nodes** (accepted).

Reasons: flyvis is PyTorch and reusing its pretrained optic lobe is a core goal; PPO tooling
(torch-native or hand-rolled) is mature; sparse ops on MPS are weak but we don't need them at
this graph size. MLX is genuinely attractive on unified memory and is the better choice if we ever
go past ~20k nodes and need sparse recurrent updates with lazy evaluation.

`flybrain bench-backend` runs the actual workload (batched recurrent rate update on an N-node
graph, ~10 in-edges/node, forward + backward through T = 32 steps, batch 256) on CPU, MPS and MLX.
Result on this M5 (torch 2.14.0, mlx 0.32.2; full table in `docs/bench-backend.md`):

| N nodes | torch CPU | torch MPS | MLX | MLX ÷ MPS |
|---|---|---|---|---|
| 1,000 | 105 ms | 27 ms | 17 ms | 1.6× |
| 5,000 | 1,821 ms | 1,164 ms | 626 ms | 1.9× |
| 20,000 (sparse) | 12,987 ms | n/a (no sparse matmul on MPS) | n/a (no sparse in MLX) | — |

Verdict: MLX is faster but under the 2× bar I set, and flyvis reuse tips it — **PyTorch/MPS
stands**. Two things the numbers force into the plan, though:

1. **Size the trainable graph at ~1–2k nodes, with 5k as a hard cap**, not "up to 5k". Dense
   cost is O(N²) per step: at 5k nodes and ~13 neural sub-steps per decision, MPS gives ≈ 540
   env-decisions/s at batch 256 — usable but slow (a 256-episode batch at the 180 s cap ≈ 20 min).
   At 1k nodes it is ≈ 23k env-decisions/s. Type-pooled (§4) lands in the fast regime by
   construction; per-neuron needs the optic-lobe column subsampling to stay under ~2k.
2. **The sparse cliff is real on both accelerators.** Anything past ~5k nodes means CPU sparse
   (~630 step·env/s at 20k) or a custom gather/scatter kernel. We don't go there in this project.

## 8. Evaluation (Phase 6)

Fixed protocol, all agents, 200 held-out seeds, 180 s cap, report median/mean distance,
survival-to-cap rate, and a distance histogram:
random · scripted heuristic · human (keyboard, ≥ 20 runs) · MLP-PPO · connectome readout-only ·
connectome full · shuffled-topology control · random-graph control · silenced control.
The write-up compares directly against doomfly's reported negative result and Fly Dino's positive
one, and says which of our controls did or didn't separate.

## 9. Milestones

| # | Milestone | Acceptance criterion |
|---|---|---|
| M0 | Scaffold | both packages install from a clean checkout; README covers setup + attribution |
| M1 | Runner game + API | playable in browser at 60 fps; `npm test` passes (determinism, collisions, state layout, heuristic bot survives 60 s); headless bench ≥ 200k single-env steps/s in Node |
| M2 | Connectome artifact | **done** — `flybrain build-graph` produces versioned parquet + manifest; `graph-report` prints the type table; 3.7 GB peak RSS |
| M3 | Substrate | **done** — pooled 51k / per-neuron 1.5k env-decisions/s at batch 256 on MPS (fwd+bwd); calibration sweep in place; shuffled/random/silenced/readout-only controls tested |
| M4a | MLP-PPO baseline | **partial** — 1,954 m mean / 47 % to cap after 75 min (heuristic 1,209 m, 1 %); still improving toward the 90 % bar |
| M4b | Readout-only connectome | **failed on the pooled graph** (49 m; controls 37–40 m; MLP 214 m sample-matched); per-neuron graph under-trained at 1 M decisions (37 m, pure no-op) |
| M4c | Full connectome (or CMA-ES fallback) | **DAgger + substrate trainable: 317 m** (§6.4); **topology controls beat the real graph 2.2–2.5×** under the same protocol (§6.5); ES replication not run |
| M5 | Live demo | **done** — mode selector (human / heuristic / random / MLP / fly) + live activity panel (11 cell-type groups, 16 DN pools, action logits) |
| M6 | Write-up | **done** — README Results / What worked / What didn't / Versus doomfly / Limitations; this doc |

## 10. Risks

- **Python 3.14 on host:** flyvis blocks it. Mitigated: uv-managed 3.12.
- **Cell-type names in MaleCNS differ from hemibrain/FlyWire conventions:** resolve via Cell Type
  Explorer at M2, record in manifest; never guess in code.
- **Recurrent rate-net gradients blow up (Stage C):** CMA-ES fallback is a flag, not a rewrite.
  Calibration (§5.1) showed the per-neuron graph crosses its stability boundary between synapse
  scale 1.5 and 2, so Stage C must bound `scale` (clamp or penalty) from the first run.
- **Substrate does nothing useful (doomfly outcome):** calibration gate + readout-only stage
  isolate whether the problem is the substrate or the training; controls make a null result
  publishable rather than embarrassing.
- **Memory:** only the pipeline touches the full graph, only as sparse; simulation graph ≤ 5k nodes.
- **MPS availability on newer macOS builds has had regressions** (pytorch issues #167679,
  #177819): the bench script checks `torch.backends.mps.is_available()` and falls back to CPU
  with a loud warning rather than crashing.

## 11. Open questions for you

1. Are you OK with the two-engine (TS source of truth + NumPy port with golden fixture) approach,
   versus writing the engine once in Python and driving the browser demo over a WebSocket?
   (I recommend two-engine: the browser demo stays dependency-free and the training loop stays
   fast; the port is a few hundred lines.)
2. Type-pooled first, or go straight to per-neuron? (I recommend type-pooled for M3 to get the
   training loop working, then per-neuron for the real experiment.)
3. Do you want the flyvis pretrained optic lobe as the front end from the start, or the cheaper
   `state`-encoding variant first? (I recommend `state` first; `frame` is the demo-worthy one.)
4. Any preference on the MLX vs PyTorch call beyond "benchmark and see"?
