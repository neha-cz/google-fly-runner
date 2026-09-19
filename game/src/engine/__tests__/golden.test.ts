import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { Runner, STATE_SIZE } from "..";

const path = new URL("../../../fixtures/golden.json", import.meta.url);

describe("golden trajectory fixture", () => {
  it.skipIf(!existsSync(path))("engine reproduces fixtures/golden.json exactly (shared with the NumPy port)", () => {
    const g = JSON.parse(readFileSync(path, "utf8"));
    expect(g.stateSize).toBe(STATE_SIZE);
    for (const ep of g.episodes) {
      const r = new Runner(ep.seed), st = new Float32Array(STATE_SIZE);
      let ci = 0;
      for (let i = 0; i < ep.actions.length; i++) {
        r.act(ep.actions[i]); r.step();
        if ((i + 1) % g.every === 0) {
          const c = ep.checkpoints[ci++];
          expect(r.alive).toBe(c.alive);
          expect(r.distance).toBeCloseTo(c.distance, 5);
          expect(r.score).toBeCloseTo(c.score, 5);
          expect(r.coinsCollected).toBe(c.coins);
          expect(r.lane).toBe(c.lane);
          const s = Array.from(r.getState(st));
          for (let k = 0; k < STATE_SIZE; k++) expect(s[k]).toBeCloseTo(c.state[k], 5);
        }
      }
      expect(r.distance).toBeCloseTo(ep.final.distance, 5);
    }
  });
});
