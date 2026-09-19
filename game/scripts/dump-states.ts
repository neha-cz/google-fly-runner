/**
 * Dump getState()/getFrame() samples from headless play into flybrain/data/cache for
 * substrate calibration (Phase 3). Mixed random + heuristic policies, several seeds.
 *   npm run dump-states -- --steps 20000 --frames 2000
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { Runner, STATE_SIZE, FRAME_W, FRAME_H, DT } from "../src/engine";
import { HeuristicAgent, RandomAgent } from "../src/agents";

const argv = process.argv.slice(2);
const arg = (k: string, d: number) => { const i = argv.indexOf(`--${k}`); return i >= 0 ? Number(argv[i + 1]) : d; };
const STEPS = arg("steps", 20000), FRAMES = arg("frames", 2000), EVERY = 4; // decisions at 15 Hz

const outDir = new URL("../../flybrain/data/cache/", import.meta.url);
mkdirSync(outDir, { recursive: true });
const states = new Float32Array(STEPS * STATE_SIZE), frames = new Uint8Array(FRAMES * FRAME_W * FRAME_H);
const policy = new Uint8Array(STEPS), st = new Float32Array(STATE_SIZE);
let n = 0, nf = 0, seed = 1000;
while (n < STEPS) {
  const r = new Runner(seed++), useH = seed % 2 === 0, agent = useH ? new HeuristicAgent() : new RandomAgent(seed, 0.7);
  for (let t = 0; t < 60 * 120 && r.alive && n < STEPS; t++) {
    if (t % EVERY === 0) {
      states.set(r.getState(st), n * STATE_SIZE); policy[n] = useH ? 1 : 0;
      if (nf < FRAMES && n % Math.ceil(STEPS / FRAMES) === 0) { r.getFrame(frames.subarray(nf * FRAME_W * FRAME_H, (nf + 1) * FRAME_W * FRAME_H)); nf++; }
      n++;
    }
    r.act(agent.act(st, t)); r.step();
  }
}
writeFileSync(new URL("states.f32", outDir), Buffer.from(states.buffer));
writeFileSync(new URL("frames.u8", outDir), Buffer.from(frames.buffer, 0, nf * FRAME_W * FRAME_H));
writeFileSync(new URL("policy.u8", outDir), Buffer.from(policy.buffer));
writeFileSync(new URL("meta.json", outDir), JSON.stringify({ steps: n, stateSize: STATE_SIZE, frames: nf, frameW: FRAME_W, frameH: FRAME_H, decisionEvery: EVERY, dt: DT }));
console.log(`wrote ${n} states, ${nf} frames to flybrain/data/cache/`);
