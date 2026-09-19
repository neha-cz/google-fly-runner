import { Runner } from "./runner";
import type { Action, StepResult } from "./types";
import { STATE_SIZE } from "./state";

/**
 * A batch of independent engines stepped in lock-step. Crashed instances are reset with a
 * fresh seed on the following step (auto-reset), mirroring Gymnasium VectorEnv semantics.
 */
export class VecRunner {
  readonly runners: Runner[];
  readonly states: Float32Array;
  private nextSeed: number;
  episodes = 0;
  constructor(readonly n: number, seed = 1, readonly autoReset = true) {
    this.runners = Array.from({ length: n }, (_, i) => new Runner(seed + i));
    this.states = new Float32Array(n * STATE_SIZE);
    this.nextSeed = seed + n;
  }
  act(i: number, a: Action): void { this.runners[i].act(a); }
  step(): StepResult[] {
    const out: StepResult[] = new Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const r = this.runners[i];
      if (!r.alive && this.autoReset) { r.reset(this.nextSeed++); this.episodes++; }
      out[i] = r.step();
      r.getState(this.states.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE));
    }
    return out;
  }
  stateOf(i: number): Float32Array { return this.states.subarray(i * STATE_SIZE, (i + 1) * STATE_SIZE); }
}
