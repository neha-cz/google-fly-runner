import { ACTIONS, type Action, Rng } from "../engine";
import type { Agent } from "./types";

/** Uniform-random policy with a no-op bias; the "does anything beat chance" baseline. */
export class RandomAgent implements Agent {
  name = "random";
  private rng: Rng;
  constructor(seed = 7, private pNoop = 0.85) { this.rng = new Rng(seed); }
  act(_state?: Float32Array, _tick?: number): Action {
    if (this.rng.next() < this.pNoop) return "noop";
    return ACTIONS[1 + this.rng.int(ACTIONS.length - 1)];
  }
}
