import type { Runner } from "./runner";
import { LANE_COUNT, LANE_X, LOOKAHEAD, MAX_SPEED, JUMP_DURATION, SLIDE_DURATION } from "./constants";

export const OBSTACLES_PER_LANE = 2;

/**
 * Layout of the getState() vector. Every entry is in [-1, 1].
 *
 *  [0..2]   lane one-hot (target lane)
 *  [3]      lateral offset x (world units, -1..1)
 *  [4]      airborne (0/1)
 *  [5]      jump phase (0..1)
 *  [6]      sliding (0/1)
 *  [7]      slide phase (0..1)
 *  [8]      speed / MAX_SPEED
 *  [9..32]  per lane (3) × next obstacles (2) × [isLow, isHigh, isFull, dist/LOOKAHEAD]
 *           (all-zero kind + dist 1.0 when there is no obstacle within LOOKAHEAD)
 *  [33..35] per lane: distance to nearest un-taken coin / LOOKAHEAD (1.0 if none)
 */
export const STATE_SIZE = 9 + LANE_COUNT * OBSTACLES_PER_LANE * 4 + LANE_COUNT;

export const STATE_LAYOUT: string[] = (() => {
  const n = ["lane0", "lane1", "lane2", "x", "airborne", "jumpPhase", "sliding", "slidePhase", "speed"];
  for (let l = 0; l < LANE_COUNT; l++) for (let k = 0; k < OBSTACLES_PER_LANE; k++)
    n.push(`L${l}.o${k}.low`, `L${l}.o${k}.high`, `L${l}.o${k}.full`, `L${l}.o${k}.dist`);
  for (let l = 0; l < LANE_COUNT; l++) n.push(`L${l}.coinDist`);
  return n;
})();

export const OBS_OFFSET = 9;
export const COIN_OFFSET = 9 + LANE_COUNT * OBSTACLES_PER_LANE * 4;

export function fillState(r: Runner, out: Float32Array): Float32Array {
  out.fill(0);
  out[r.lane] = 1;
  out[3] = r.px / LANE_X[LANE_COUNT - 1];
  out[4] = r.airborne ? 1 : 0;
  out[5] = r.airborne ? r.jumpT / JUMP_DURATION : 0;
  out[6] = r.sliding ? 1 : 0;
  out[7] = r.sliding ? r.slideT / SLIDE_DURATION : 0;
  out[8] = r.speed / MAX_SPEED;

  for (let l = 0; l < LANE_COUNT; l++)
    for (let k = 0; k < OBSTACLES_PER_LANE; k++) out[OBS_OFFSET + (l * OBSTACLES_PER_LANE + k) * 4 + 3] = 1;
  const count = [0, 0, 0];
  for (const o of r.obstacles) {                     // obstacles are z-sorted
    const l = o.lane;
    if (count[l] >= OBSTACLES_PER_LANE) continue;
    if (o.z + o.depth < r.pz) continue;              // already behind
    const d = Math.max(0, o.z - r.pz);
    if (d > LOOKAHEAD) continue;
    const base = OBS_OFFSET + (l * OBSTACLES_PER_LANE + count[l]) * 4;
    out[base + o.kind] = 1;
    out[base + 3] = d / LOOKAHEAD;
    count[l]++;
  }
  for (let l = 0; l < LANE_COUNT; l++) out[COIN_OFFSET + l] = 1;
  const seen = [false, false, false];
  for (const c of r.coins) {
    if (c.taken || seen[c.lane] || c.z < r.pz) continue;
    const d = c.z - r.pz;
    if (d > LOOKAHEAD) continue;
    out[COIN_OFFSET + c.lane] = d / LOOKAHEAD; seen[c.lane] = true;
  }
  return out;
}
