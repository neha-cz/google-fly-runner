/**
 * Regenerates fixtures/golden.json: the reference trajectory both the TS engine and the
 * NumPy port (flybrain/env) must reproduce bit-for-bit. Only re-run this when the engine
 * rules intentionally change, and say so in the commit message.
 */
import { writeFileSync } from "node:fs";
import { Runner, STATE_SIZE } from "../src/engine";
import { HeuristicAgent } from "../src/agents";

const SEEDS = [1, 42, 1234];
const STEPS = 1800; // 30 s
const EVERY = 60;

const out: any = { steps: STEPS, every: EVERY, stateSize: STATE_SIZE, episodes: [] };
for (const seed of SEEDS) {
  const r = new Runner(seed), a = new HeuristicAgent(), st = new Float32Array(STATE_SIZE);
  const actions: string[] = [], checkpoints: any[] = [];
  for (let i = 0; i < STEPS; i++) {
    const act = a.act(r.getState(st), i); actions.push(act); r.act(act); r.step();
    if ((i + 1) % EVERY === 0) checkpoints.push({
      step: i + 1, alive: r.alive, distance: +r.distance.toFixed(6), score: +r.score.toFixed(6),
      coins: r.coinsCollected, x: +r.px.toFixed(6), y: +r.py.toFixed(6), lane: r.lane,
      nObstacles: r.obstacles.length, state: Array.from(r.getState(st)).map(v => +v.toFixed(6)),
    });
    if (!r.alive) break;
  }
  out.episodes.push({ seed, actions, checkpoints, final: { alive: r.alive, distance: +r.distance.toFixed(6), score: +r.score.toFixed(6) } });
}
writeFileSync(new URL("../fixtures/golden.json", import.meta.url), JSON.stringify(out));
console.log("wrote fixtures/golden.json", out.episodes.map((e: any) => e.final));
