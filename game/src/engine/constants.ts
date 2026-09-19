/** Fixed simulation timestep (seconds). The engine is only ever advanced by this. */
export const DT = 1 / 60;

export const LANE_COUNT = 3;
/** Lateral centre of each lane, world units. */
export const LANE_X = [-1, 0, 1] as const;

export const BASE_SPEED = 8;          // world units / s at distance 0
export const MAX_SPEED = 22;
export const SPEED_PER_METER = 0.015; // speed = BASE + distance * this, capped

export const JUMP_DURATION = 0.55;    // s airborne
export const JUMP_HEIGHT = 1.0;
export const LOW_HEIGHT = 0.3;        // player must be above this to clear a LOW obstacle
export const SLIDE_DURATION = 0.6;    // s
export const LANE_CHANGE_DURATION = 0.2; // s

export const PLAYER_HALF_DEPTH = 0.25;
export const PLAYER_HALF_WIDTH = 0.3;
export const OBSTACLE_HALF_WIDTH = 0.45;
/** Depth along the track by obstacle kind (index = ObstacleKind). */
export const OBSTACLE_DEPTH = [0.6, 0.6, 0.8] as const;
export const COIN_RADIUS = 0.35;
export const COIN_VALUE = 10;

export const SPAWN_AHEAD = 70;   // keep the track populated this far ahead of the player
export const DESPAWN_BEHIND = 3;
export const LOOKAHEAD = 40;     // distance normalisation for getState()

export const FRAME_W = 64;
export const FRAME_H = 48;
