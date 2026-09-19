export enum ObstacleKind { Low = 0, High = 1, Full = 2 }
export const OBSTACLE_KIND_NAMES = ["low", "high", "full"] as const;

export const ACTIONS = ["noop", "jump", "slide", "left", "right"] as const;
export type Action = (typeof ACTIONS)[number];
export const ACTION_INDEX: Record<Action, number> = { noop: 0, jump: 1, slide: 2, left: 3, right: 4 };

export interface Obstacle { kind: ObstacleKind; lane: number; z: number; depth: number }
export interface Coin { lane: number; z: number; taken: boolean }

export interface StepResult {
  /** True if the player was alive after this step. */
  alive: boolean;
  /** True on the exact step the crash happened. */
  crashed: boolean;
  /** Coins collected during this step. */
  coins: number;
  /** Distance advanced during this step (world units ≈ metres). */
  dz: number;
}

/** Read-only view of the world for renderers and debugging. Do not mutate. */
export interface Snapshot {
  time: number;
  alive: boolean;
  distance: number;
  score: number;
  coinsCollected: number;
  speed: number;
  player: {
    x: number; y: number; z: number;
    lane: number; laneT: number; leanDir: number;
    airborne: boolean; jumpT: number;
    sliding: boolean; slideT: number;
  };
  obstacles: readonly Obstacle[];
  coins: readonly Coin[];
  crashedInto: Obstacle | null;
}
