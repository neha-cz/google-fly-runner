import { Rng } from "./rng";
import { Spawner, speedAt } from "./spawner";
import { ObstacleKind, type Action, type Obstacle, type Coin, type StepResult, type Snapshot } from "./types";
import {
  DT, LANE_X, LANE_COUNT, JUMP_DURATION, JUMP_HEIGHT, LOW_HEIGHT, SLIDE_DURATION, LANE_CHANGE_DURATION,
  PLAYER_HALF_DEPTH, PLAYER_HALF_WIDTH, OBSTACLE_HALF_WIDTH, COIN_RADIUS, COIN_VALUE, DESPAWN_BEHIND,
} from "./constants";
import { STATE_SIZE, fillState } from "./state";
import { rasterize } from "./frame";
import { FRAME_W, FRAME_H } from "./constants";

const smoothstep = (t: number) => t * t * (3 - 2 * t);

/**
 * The FlyRunner engine: a deterministic, headless, fixed-timestep 3-lane runner.
 *
 * Programmatic API (the point of this project):
 *   jump() slide() moveLeft() moveRight()  — latch an action for the next step
 *   act(action)                            — same, by name
 *   step()                                 — advance exactly one tick (DT)
 *   getState(out?)                         — compact Float32Array(36), see state.ts
 *   getFrame(out?, w?, h?)                 — grayscale Uint8Array, software-rasterised view
 *   snapshot()                             — full world view for renderers
 *   reset(seed)
 */
export class Runner {
  // world
  private rng!: Rng;
  private spawner!: Spawner;
  obstacles: Obstacle[] = [];
  coins: Coin[] = [];
  // scalar state
  time = 0; alive = true; distance = 0; score = 0; coinsCollected = 0; speed = 0;
  crashedInto: Obstacle | null = null;
  // player
  px = 0; py = 0; pz = 0;
  lane = 1; laneT = 1; fromX = 0; leanDir = 0;
  jumpT = -1; slideT = -1;
  // pending action, consumed on the next step
  private pending: Action = "noop";

  constructor(seed = 1) { this.reset(seed); }

  reset(seed: number): void {
    this.rng = new Rng(seed);
    this.obstacles = []; this.coins = [];
    this.time = 0; this.alive = true; this.distance = 0; this.score = 0; this.coinsCollected = 0;
    this.speed = speedAt(0); this.crashedInto = null;
    this.px = 0; this.py = 0; this.pz = 0; this.lane = 1; this.laneT = 1; this.fromX = 0; this.leanDir = 0;
    this.jumpT = -1; this.slideT = -1; this.pending = "noop";
    this.spawner = new Spawner(this.rng, 0);
    this.spawner.update(this.pz, this.distance, this.obstacles, this.coins);
  }

  get airborne(): boolean { return this.jumpT >= 0; }
  get sliding(): boolean { return this.slideT >= 0; }

  // ---- action API -------------------------------------------------------
  jump(): void { this.pending = "jump"; }
  slide(): void { this.pending = "slide"; }
  moveLeft(): void { this.pending = "left"; }
  moveRight(): void { this.pending = "right"; }
  act(a: Action): void { this.pending = a; }

  private applyAction(a: Action): void {
    switch (a) {
      case "jump":
        if (!this.airborne) { this.slideT = -1; this.jumpT = 0; }
        break;
      case "slide":
        if (this.airborne) { this.jumpT = -1; this.py = 0; } // fast-fall
        if (!this.sliding) this.slideT = 0;
        break;
      case "left": this.changeLane(-1); break;
      case "right": this.changeLane(+1); break;
    }
  }

  private changeLane(dir: -1 | 1): void {
    const target = this.lane + dir;
    if (target < 0 || target >= LANE_COUNT) return;
    this.fromX = this.px; this.lane = target; this.laneT = 0; this.leanDir = dir;
  }

  // ---- simulation -------------------------------------------------------
  /** Advance one fixed tick. Returns what happened this tick. */
  step(): StepResult {
    if (!this.alive) return { alive: false, crashed: false, coins: 0, dz: 0 };
    const a = this.pending; this.pending = "noop";
    this.applyAction(a);

    this.time += DT;
    this.speed = speedAt(this.distance);
    const dz = this.speed * DT;
    this.distance += dz; this.pz += dz; this.score += dz;

    if (this.laneT < 1) {
      this.laneT = Math.min(1, this.laneT + DT / LANE_CHANGE_DURATION);
      this.px = this.fromX + (LANE_X[this.lane] - this.fromX) * smoothstep(this.laneT);
      if (this.laneT >= 1) { this.px = LANE_X[this.lane]; this.leanDir = 0; }
    }
    if (this.jumpT >= 0) {
      this.jumpT += DT;
      if (this.jumpT >= JUMP_DURATION) { this.jumpT = -1; this.py = 0; }
      else this.py = JUMP_HEIGHT * Math.sin(Math.PI * this.jumpT / JUMP_DURATION);
    }
    if (this.slideT >= 0) {
      this.slideT += DT;
      if (this.slideT >= SLIDE_DURATION) this.slideT = -1;
    }

    this.spawner.update(this.pz, this.distance, this.obstacles, this.coins);

    // collisions
    const zLo = this.pz - PLAYER_HALF_DEPTH, zHi = this.pz + PLAYER_HALF_DEPTH;
    let crashed = false;
    for (const o of this.obstacles) {
      if (o.z > zHi || o.z + o.depth < zLo) continue;
      if (Math.abs(this.px - LANE_X[o.lane]) >= PLAYER_HALF_WIDTH + OBSTACLE_HALF_WIDTH) continue;
      if (o.kind === ObstacleKind.Low && this.py > LOW_HEIGHT) continue;
      if (o.kind === ObstacleKind.High && this.sliding) continue;
      crashed = true; this.crashedInto = o; break;
    }
    let got = 0;
    for (const c of this.coins) {
      if (c.taken || Math.abs(c.z - this.pz) > COIN_RADIUS + PLAYER_HALF_DEPTH) continue;
      if (Math.abs(this.px - LANE_X[c.lane]) >= PLAYER_HALF_WIDTH + COIN_RADIUS) continue;
      c.taken = true; got++;
    }
    this.coinsCollected += got; this.score += got * COIN_VALUE;

    // despawn (cheap because arrays are short and z-sorted by construction)
    const cut = this.pz - DESPAWN_BEHIND;
    if (this.obstacles.length && this.obstacles[0].z + this.obstacles[0].depth < cut) {
      let i = 0; while (i < this.obstacles.length && this.obstacles[i].z + this.obstacles[i].depth < cut) i++;
      this.obstacles.splice(0, i);
    }
    if (this.coins.length && this.coins[0].z < cut) {
      let i = 0; while (i < this.coins.length && this.coins[i].z < cut) i++;
      this.coins.splice(0, i);
    }

    if (crashed) this.alive = false;
    return { alive: this.alive, crashed, coins: got, dz };
  }

  // ---- observation API --------------------------------------------------
  getState(out: Float32Array = new Float32Array(STATE_SIZE)): Float32Array { return fillState(this, out); }

  getFrame(out?: Uint8Array, w = FRAME_W, h = FRAME_H): Uint8Array {
    const buf = out ?? new Uint8Array(w * h);
    rasterize(this.snapshot(), buf, w, h);
    return buf;
  }

  snapshot(): Snapshot {
    return {
      time: this.time, alive: this.alive, distance: this.distance, score: this.score,
      coinsCollected: this.coinsCollected, speed: this.speed,
      player: {
        x: this.px, y: this.py, z: this.pz, lane: this.lane, laneT: this.laneT, leanDir: this.leanDir,
        airborne: this.airborne, jumpT: this.jumpT, sliding: this.sliding, slideT: this.slideT,
      },
      obstacles: this.obstacles, coins: this.coins, crashedInto: this.crashedInto,
    };
  }

  /** Test hook: place an obstacle `dist` ahead in `lane`. */
  debugSpawn(kind: ObstacleKind, lane: number, dist: number): Obstacle {
    const o: Obstacle = { kind, lane, z: this.pz + dist, depth: 0.6 };
    this.obstacles.push(o); this.obstacles.sort((a, b) => a.z - b.z);
    return o;
  }
  /** Test hook: remove everything ahead. */
  debugClear(): void { this.obstacles.length = 0; this.coins.length = 0; }
}
