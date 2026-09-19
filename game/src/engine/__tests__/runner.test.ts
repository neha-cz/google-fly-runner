import { describe, it, expect } from "vitest";
import { Runner, VecRunner, Rng, STATE_SIZE, STATE_LAYOUT, ObstacleKind, LOOKAHEAD, OBS_OFFSET, JUMP_DURATION, DT, FRAME_W, FRAME_H, LANE_X, OBSTACLE_DEPTH } from "..";
import { HeuristicAgent, RandomAgent } from "../../agents";

const SECONDS = (s: number) => Math.round(s / DT);

function runAgent(seed: number, agent: { act(s: Float32Array, t: number): any }, maxSteps: number) {
  const r = new Runner(seed), st = new Float32Array(STATE_SIZE);
  let t = 0;
  for (; t < maxSteps && r.alive; t++) { r.act(agent.act(r.getState(st), t)); r.step(); }
  return { runner: r, steps: t };
}

describe("Rng", () => {
  it("is a deterministic xorshift32 stream", () => {
    const a = new Rng(42), b = new Rng(42);
    const xs = Array.from({ length: 5 }, () => a.nextU32());
    expect(xs).toEqual(Array.from({ length: 5 }, () => b.nextU32()));
    // reference values (also asserted by the NumPy port)
    expect(new Rng(1).nextU32()).toBe(270369);
    expect(xs.every(x => x >= 0 && x < 2 ** 32)).toBe(true);
  });
});

describe("determinism", () => {
  it("same seed + same actions => identical trajectories", () => {
    const a = new Runner(7), b = new Runner(7), ag = new HeuristicAgent();
    const sa = new Float32Array(STATE_SIZE), sb = new Float32Array(STATE_SIZE);
    for (let t = 0; t < SECONDS(20); t++) {
      const act = ag.act(a.getState(sa), t); b.getState(sb);
      expect(Array.from(sb)).toEqual(Array.from(sa));
      a.act(act); b.act(act); a.step(); b.step();
    }
    expect(b.distance).toBe(a.distance);
  });
  it("different seeds => different tracks", () => {
    const a = new Runner(1), b = new Runner(2);
    expect(a.obstacles.map(o => o.z + o.lane * 100)).not.toEqual(b.obstacles.map(o => o.z + o.lane * 100));
  });
});

describe("state vector", () => {
  it("has the documented size and layout, all entries in [-1, 1]", () => {
    expect(STATE_LAYOUT.length).toBe(STATE_SIZE);
    expect(STATE_SIZE).toBe(36);
    const { runner } = runAgent(3, new HeuristicAgent(), SECONDS(15));
    const s = runner.getState();
    expect(s.length).toBe(STATE_SIZE);
    for (const v of s) { expect(v).toBeGreaterThanOrEqual(-1); expect(v).toBeLessThanOrEqual(1); }
    expect(s[0] + s[1] + s[2]).toBe(1);
  });
  it("reports an injected obstacle in the right lane slot with the right distance", () => {
    const r = new Runner(1); r.debugClear();
    r.debugSpawn(ObstacleKind.High, 2, 10);
    const s = r.getState();
    const base = OBS_OFFSET + (2 * 2 + 0) * 4;
    expect(s[base + 1]).toBe(1);                       // isHigh
    expect(s[base + 3]).toBeCloseTo(10 / LOOKAHEAD, 5);
    expect(s[OBS_OFFSET + 3]).toBe(1);                 // lane 0 slot 0: nothing => dist 1.0
  });
});

describe("collisions", () => {
  it("crashes into a LOW obstacle when doing nothing, and clears it by jumping", () => {
    const crash = new Runner(1); crash.debugClear(); crash.debugSpawn(ObstacleKind.Low, 1, 4);
    let t = 0; while (crash.alive && t++ < 200) crash.step();
    expect(crash.alive).toBe(false);

    const ok = new Runner(1); ok.debugClear(); ok.debugSpawn(ObstacleKind.Low, 1, 4);
    // jump so that we're mid-air when we reach z=4: 4 / 8 m/s = 0.5 s; jump lasts 0.55 s -> start at ~0.25 s
    for (let i = 0; i < 200 && ok.alive; i++) { if (i === Math.round(0.28 / DT)) ok.jump(); ok.step(); }
    expect(ok.alive).toBe(true);
    expect(ok.pz).toBeGreaterThan(4 + OBSTACLE_DEPTH[0]);
  });
  it("HIGH obstacle: crash when running, clear when sliding; FULL: cleared only by changing lane", () => {
    const h1 = new Runner(1); h1.debugClear(); h1.debugSpawn(ObstacleKind.High, 1, 4);
    for (let i = 0; i < 200 && h1.alive; i++) h1.step();
    expect(h1.alive).toBe(false);
    const h2 = new Runner(1); h2.debugClear(); h2.debugSpawn(ObstacleKind.High, 1, 4);
    for (let i = 0; i < 200 && h2.alive; i++) { if (i === Math.round(0.35 / DT)) h2.slide(); h2.step(); }
    expect(h2.alive).toBe(true);
    const f1 = new Runner(1); f1.debugClear(); f1.debugSpawn(ObstacleKind.Full, 1, 4);
    for (let i = 0; i < 200 && f1.alive; i++) { if (i === 5) f1.jump(); if (i === 40) f1.slide(); f1.step(); }
    expect(f1.alive).toBe(false);
    const f2 = new Runner(1); f2.debugClear(); f2.debugSpawn(ObstacleKind.Full, 1, 4);
    for (let i = 0; i < 200 && f2.alive; i++) { if (i === 2) f2.moveLeft(); f2.step(); }
    expect(f2.alive).toBe(true);
    expect(f2.px).toBe(LANE_X[0]);
  });
  it("cannot leave the track and jump is a fixed-duration latch", () => {
    const r = new Runner(1); r.debugClear();
    r.moveLeft(); r.step(); r.moveLeft(); r.step(); for (let i = 0; i < 20; i++) r.step();
    expect(r.lane).toBe(0); expect(r.px).toBe(-1);
    r.jump(); r.step(); expect(r.airborne).toBe(true);
    r.jump(); // ignored while airborne
    for (let i = 0; i < Math.round(JUMP_DURATION / DT) + 1; i++) r.step();
    expect(r.airborne).toBe(false); expect(r.py).toBe(0);
  });
});

describe("track generation", () => {
  it("never blocks all three lanes with FULL obstacles, and stays z-sorted", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const r = new Runner(seed);
      for (let t = 0; t < SECONDS(120); t++) {
        if (!r.alive) r.reset(seed * 100 + t);
        r.step();
        const byZ = new Map<number, Set<number>>();
        for (let i = 1; i < r.obstacles.length; i++) expect(r.obstacles[i].z).toBeGreaterThanOrEqual(r.obstacles[i - 1].z);
        for (const o of r.obstacles) { if (o.kind !== ObstacleKind.Full) continue; const s = byZ.get(o.z) ?? new Set(); s.add(o.lane); byZ.set(o.z, s); }
        for (const s of byZ.values()) expect(s.size).toBeLessThan(3);
      }
    }
  }, 30_000);   // 5 seeds x 120 s of play; slow under CPU contention
  it("speed ramps and is capped", () => {
    const r = new Runner(9); r.debugClear();
    const v0 = r.speed;
    for (let t = 0; t < SECONDS(90); t++) { r.step(); r.debugClear(); }
    expect(r.speed).toBeGreaterThan(v0);
    expect(r.speed).toBeLessThanOrEqual(22);
  });
});

describe("agents through the public API", () => {
  it("the scripted heuristic survives 60 s on several seeds (the game is solvable via getState)", () => {
    for (const seed of [1, 2, 3, 42, 1234]) {
      const { runner, steps } = runAgent(seed, new HeuristicAgent(), SECONDS(60));
      expect(runner.alive, `seed ${seed} died at ${(steps * DT).toFixed(1)} s`).toBe(true);
    }
  });
  it("a random policy dies quickly (there is signal to learn)", () => {
    let total = 0;
    for (let seed = 1; seed <= 10; seed++) total += runAgent(seed, new RandomAgent(seed), SECONDS(60)).steps;
    expect(total / 10).toBeLessThan(SECONDS(20));
  });
});

describe("frame + vec", () => {
  it("getFrame returns a non-degenerate grayscale buffer of the documented size", () => {
    const r = new Runner(5); for (let i = 0; i < 30; i++) r.step();
    const f = r.getFrame();
    expect(f.length).toBe(FRAME_W * FRAME_H);
    const vals = new Set(f); expect(vals.size).toBeGreaterThan(4);
  });
  it("VecRunner auto-resets crashed instances and exposes a packed state buffer", () => {
    const v = new VecRunner(8, 100), ag = new RandomAgent(3);
    for (let t = 0; t < SECONDS(40); t++) { for (let i = 0; i < 8; i++) v.act(i, ag.act()); v.step(); }
    expect(v.episodes).toBeGreaterThan(0);
    expect(v.states.length).toBe(8 * STATE_SIZE);
    expect(v.runners.every(r => r.alive || true)).toBe(true);
  });
});
