/**
 * Headless throughput benchmark. Run with: npm run bench [-- --n 64 --steps 200000]
 */
import { VecRunner, STATE_SIZE, Runner } from "../engine";
import { HeuristicAgent, RandomAgent } from "../agents";

const argv = process.argv.slice(2);
const arg = (k: string, d: number) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? Number(argv[i + 1]) : d; };
const N = arg("n", 64), STEPS = arg("steps", 200_000);

function bench(label: string, fn: () => number) {
  const t0 = performance.now(); const steps = fn(); const s = (performance.now() - t0) / 1000;
  console.log(`${label.padEnd(36)} ${(steps / s / 1000).toFixed(0).padStart(7)}k steps/s`);
}

bench("single env, noop", () => { const r = new Runner(1); for (let i = 0; i < STEPS; i++) { if (!r.alive) r.reset(i); r.step(); } return STEPS; });
bench("single env, heuristic + getState", () => {
  const r = new Runner(1), a = new HeuristicAgent(), st = new Float32Array(STATE_SIZE);
  for (let i = 0; i < STEPS; i++) { if (!r.alive) r.reset(i); r.act(a.act(r.getState(st), i)); r.step(); } return STEPS;
});
bench("single env, heuristic + getFrame", () => {
  const r = new Runner(1), a = new HeuristicAgent(), st = new Float32Array(STATE_SIZE), fr = new Uint8Array(64 * 48);
  const n = STEPS / 10;
  for (let i = 0; i < n; i++) { if (!r.alive) r.reset(i); r.act(a.act(r.getState(st), i)); r.step(); r.getFrame(fr); } return n;
});
bench(`vec ${N} envs, random`, () => {
  const v = new VecRunner(N, 1), a = new RandomAgent(); const per = Math.ceil(STEPS / N);
  for (let t = 0; t < per; t++) { for (let i = 0; i < N; i++) v.act(i, a.act()); v.step(); } return per * N;
});
bench(`vec ${N} envs, heuristic`, () => {
  const v = new VecRunner(N, 1), a = new HeuristicAgent(); const per = Math.ceil(STEPS / N);
  for (let t = 0; t < per; t++) { for (let i = 0; i < N; i++) v.act(i, a.act(v.stateOf(i), t)); v.step(); } return per * N;
});
