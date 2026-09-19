/**
 * Browser runtime for agents exported by `flybrain export` (JSON bundles in public/agents/).
 * Mirrors flybrain/train/policies.py + model/substrate.py; checked against each bundle's
 * self-test block (reference states -> logits from PyTorch) in neural.test.ts.
 * Decisions are made every DECISION_EVERY ticks with greedy actions, exactly as in evaluation.
 */
import { ACTIONS, type Action } from "../engine";
import type { Agent } from "./types";

export const DECISION_EVERY = 4;

interface Linear { w: number[][]; b: number[] }
export interface MLPBundle { kind: "mlp"; name: string; layers: Linear[]; pi: Linear; selftest: SelfTest }
export interface FlyBundle {
  kind: "fly"; name: string; dynamics: "rate" | "lif"; substeps: number; input_gain: number; lif_threshold: number; n: number;
  nodes: { name: string[]; type: string[]; group: string[]; side: string[]; alpha: number[]; bias: number[] };
  edges: { pre: number[]; post: number[]; w: number[] };
  encoder: { idx: number[]; w: number[][]; b: number[] };
  readout: { idx: number[]; kind: "linear"; w: number[][]; b: number[] } | { idx: number[]; kind: "fixed"; M: number[][]; gain: number[]; b: number[] }
         | { idx: number[]; kind: "mlp"; h: Linear; w: number[][]; b: number[] };
  graph: { version: string; dataset: string; pooled: boolean; control: string };
  dn_norm: { mean: number[]; var: number[] } | null;
  selftest: SelfTest;
}
export interface SelfTest { states: number[][][]; logits: number[][][] }
export type Bundle = MLPBundle | FlyBundle;

const relu = (x: number) => (x > 0 ? x : 0);
function linear(l: Linear, x: ArrayLike<number>, out: Float64Array): Float64Array {
  for (let i = 0; i < l.w.length; i++) { let s = l.b[i]; const row = l.w[i]; for (let j = 0; j < row.length; j++) s += row[j] * x[j]; out[i] = s; }
  return out;
}
const argmax = (v: ArrayLike<number>) => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; };

export class MLPAgent implements Agent {
  name: string;
  private h: Float64Array[]; private logits = new Float64Array(5); private last: Action = "noop";
  constructor(private b: MLPBundle) { this.name = b.name; this.h = b.layers.map(l => new Float64Array(l.w.length)); }
  /** Pure policy step (no tick gating) — used by the self-test. */
  logitsFor(state: ArrayLike<number>): Float64Array {
    let x: ArrayLike<number> = state;
    for (let i = 0; i < this.b.layers.length; i++) { linear(this.b.layers[i], x, this.h[i]); for (let k = 0; k < this.h[i].length; k++) this.h[i][k] = Math.tanh(this.h[i][k]); x = this.h[i]; }
    return linear(this.b.pi, x, this.logits);
  }
  act(state: Float32Array, tick: number): Action {
    if (tick % DECISION_EVERY !== 0) return "noop";
    return (this.last = ACTIONS[argmax(this.logitsFor(state))]);
  }
  reset() { this.last = "noop"; }
}

export class FlyAgent implements Agent {
  name: string;
  readonly n: number;
  v: Float64Array; rates: Float64Array;           // membrane state and last window-mean rates (for the activity panel)
  logits = new Float64Array(5); lastAction = 0;
  private x: Float64Array; private inp: Float64Array; private r: Float64Array; private drive: Float64Array;
  private pre: Int32Array; private post: Int32Array; private w: Float64Array; private alpha: Float64Array; private bias: Float64Array;
  constructor(readonly b: FlyBundle) {
    this.name = b.name; this.n = b.n;
    this.v = new Float64Array(b.n); this.rates = new Float64Array(b.n); this.x = new Float64Array(b.n); this.inp = new Float64Array(b.n); this.r = new Float64Array(b.n);
    this.drive = new Float64Array(b.encoder.idx.length);
    this.pre = Int32Array.from(b.edges.pre); this.post = Int32Array.from(b.edges.post); this.w = Float64Array.from(b.edges.w);
    this.alpha = Float64Array.from(b.nodes.alpha); this.bias = Float64Array.from(b.nodes.bias);
  }
  reset() { this.v.fill(0); this.rates.fill(0); this.lastAction = 0; }
  /** One decision window: encode state, integrate `substeps`, read out DN rates. Returns logits. */
  logitsFor(state: ArrayLike<number>): Float64Array {
    const b = this.b, n = this.n, T = b.substeps, gain = b.input_gain;
    linear(b.encoder, state, this.drive);
    this.x.fill(0); for (let i = 0; i < this.drive.length; i++) this.x[b.encoder.idx[i]] = gain * relu(this.drive[i]);
    const { v, r, inp, pre, post, w, alpha, bias, rates } = this; rates.fill(0);
    for (let t = 0; t < T; t++) {
      if (b.dynamics === "rate") { for (let i = 0; i < n; i++) r[i] = relu(v[i]); }
      else { for (let i = 0; i < n; i++) { const s = v[i] - b.lif_threshold > 0 ? 1 : 0; r[i] = s; v[i] *= 1 - s; } }
      inp.fill(0); for (let e = 0; e < w.length; e++) inp[post[e]] += w[e] * r[pre[e]];
      for (let i = 0; i < n; i++) { v[i] += alpha[i] * (-v[i] + bias[i] + inp[i] + this.x[i]); rates[i] += r[i] / T; }
    }
    const ro = b.readout, dn = ro.idx;
    const dnr = new Float64Array(dn.length);
    for (let k = 0; k < dn.length; k++) dnr[k] = b.dn_norm ? (rates[dn[k]] - b.dn_norm.mean[k]) / Math.sqrt(b.dn_norm.var[k] + 1e-8) : rates[dn[k]];
    if (ro.kind === "linear") { for (let a = 0; a < 5; a++) { let s = ro.b[a]; for (let k = 0; k < dn.length; k++) s += ro.w[a][k] * dnr[k]; this.logits[a] = s; } }
    else if (ro.kind === "mlp") {
      const h = linear(ro.h, dnr, new Float64Array(ro.h.w.length)); for (let k = 0; k < h.length; k++) h[k] = Math.tanh(h[k]);
      for (let a = 0; a < 5; a++) { let s = ro.b[a]; for (let k = 0; k < h.length; k++) s += ro.w[a][k] * h[k]; this.logits[a] = s; }
    } else { for (let a = 0; a < 5; a++) { let s = 0; for (let k = 0; k < dn.length; k++) s += ro.M[a][k] * rates[dn[k]]; this.logits[a] = s * ro.gain[a] + ro.b[a]; } }
    return this.logits;
  }
  act(state: Float32Array, tick: number): Action {
    if (tick % DECISION_EVERY !== 0) return "noop";
    this.lastAction = argmax(this.logitsFor(state));
    return ACTIONS[this.lastAction];
  }
  /** Mean window rate per node group / per DN, for the live panel. */
  groupActivity(): { groups: Record<string, number>; dns: { name: string; rate: number }[] } {
    const groups: Record<string, number> = {}, counts: Record<string, number> = {};
    for (let i = 0; i < this.n; i++) { const g = this.b.nodes.group[i]; groups[g] = (groups[g] ?? 0) + this.rates[i]; counts[g] = (counts[g] ?? 0) + 1; }
    for (const g in groups) groups[g] /= counts[g];
    const dnMap = new Map<string, { sum: number; n: number }>();
    for (const i of this.b.readout.idx) { const key = `${this.b.nodes.type[i]}_${this.b.nodes.side[i]}`; const m = dnMap.get(key) ?? { sum: 0, n: 0 }; m.sum += this.rates[i]; m.n++; dnMap.set(key, m); }
    return { groups, dns: [...dnMap].map(([name, m]) => ({ name, rate: m.sum / m.n })).sort((a, b) => a.name.localeCompare(b.name)) };
  }
}

export function agentFromBundle(b: Bundle): MLPAgent | FlyAgent { return b.kind === "mlp" ? new MLPAgent(b) : new FlyAgent(b); }

/** Replays the bundle's PyTorch self-test; returns the max |Δlogit|. */
export function selfTest(b: Bundle): number {
  const ag = agentFromBundle(b); let maxErr = 0;
  const { states, logits } = b.selftest;
  const n = states[0].length;
  const agents = Array.from({ length: n }, () => agentFromBundle(b)); void ag;
  for (let t = 0; t < states.length; t++) for (let i = 0; i < n; i++) {
    const lg = agents[i].logitsFor(states[t][i]);
    for (let a = 0; a < 5; a++) maxErr = Math.max(maxErr, Math.abs(lg[a] - logits[t][i][a]));
  }
  return maxErr;
}
