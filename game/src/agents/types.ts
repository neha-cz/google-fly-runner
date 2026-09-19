import type { Action } from "../engine";

/** Anything that can drive the runner through the state API. */
export interface Agent {
  name: string;
  /** Called once per tick with the current state vector; returns the action to latch. */
  act(state: Float32Array, tick: number): Action;
  reset?(): void;
}
