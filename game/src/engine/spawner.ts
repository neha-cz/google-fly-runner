import { Rng } from "./rng";
import { ObstacleKind, type Obstacle, type Coin } from "./types";
import { BASE_SPEED, MAX_SPEED, SPEED_PER_METER, SPAWN_AHEAD, OBSTACLE_DEPTH, LANE_COUNT } from "./constants";

export function speedAt(distance: number): number {
  return Math.min(MAX_SPEED, BASE_SPEED + distance * SPEED_PER_METER);
}

/**
 * Procedural track generator. Emits "waves" (one row of obstacles across the lanes,
 * always leaving a way through) separated by speed-scaled gaps, with coin trails in between.
 * Invariants that keep the game solvable:
 *   - a wave never has FULL in every lane;
 *   - a 3-lane wave has no FULL at all (dodge by jump/slide);
 *   - gap between waves >= 0.55 s of travel, so a 2-lane change (0.4 s) is always possible.
 */
export class Spawner {
  private nextWaveZ: number;
  private firstWave = true;
  constructor(private rng: Rng, startZ: number) {
    this.nextWaveZ = startZ + 18; // a clear runway at the start
  }

  update(playerZ: number, distance: number, obstacles: Obstacle[], coins: Coin[]): void {
    while (this.nextWaveZ < playerZ + SPAWN_AHEAD) {
      const dAtWave = distance + (this.nextWaveZ - playerZ);
      const v = speedAt(dAtWave);
      const waveDepth = this.spawnWave(this.nextWaveZ, dAtWave, obstacles);
      // Gap shrinks with difficulty but never below 0.55 s of travel.
      const difficulty = Math.max(0.5, 1 - dAtWave / 1500);
      const gap = Math.max(v * 0.55, v * this.rng.range(0.9, 1.5) * difficulty);
      this.maybeCoinTrail(this.nextWaveZ + waveDepth, gap - waveDepth, coins);
      this.nextWaveZ += gap;
    }
  }

  private spawnWave(z: number, distance: number, obstacles: Obstacle[]): number {
    const r = this.rng.next();
    const p3 = this.firstWave ? 0 : Math.min(0.2, distance / 3000);
    const p2 = this.firstWave ? 0 : Math.min(0.6, 0.15 + distance / 1000);
    const nLanes = r < p3 ? 3 : r < p3 + p2 ? 2 : 1;
    this.firstWave = false;

    // choose which lanes are occupied
    const lanes: number[] = [];
    if (nLanes === 3) lanes.push(0, 1, 2);
    else if (nLanes === 2) { const free = this.rng.int(LANE_COUNT); for (let l = 0; l < LANE_COUNT; l++) if (l !== free) lanes.push(l); }
    else lanes.push(this.rng.int(LANE_COUNT));

    let maxDepth = 0;
    for (const lane of lanes) {
      const k = this.rng.next();
      let kind: ObstacleKind = k < 0.4 ? ObstacleKind.Low : k < 0.7 ? ObstacleKind.High : ObstacleKind.Full;
      if (nLanes === 3 && kind === ObstacleKind.Full) kind = k < 0.85 ? ObstacleKind.Low : ObstacleKind.High;
      const depth = OBSTACLE_DEPTH[kind];
      maxDepth = Math.max(maxDepth, depth);
      obstacles.push({ kind, lane, z, depth });
    }
    return maxDepth;
  }

  private maybeCoinTrail(startZ: number, room: number, coins: Coin[]): void {
    if (this.rng.next() > 0.6) return;
    const spacing = 0.9;
    const n = Math.min(3 + this.rng.int(4), Math.floor((room - 2.0) / spacing));
    if (n < 2) return;
    const lane = this.rng.int(LANE_COUNT);
    for (let i = 0; i < n; i++) coins.push({ lane, z: startZ + 1.0 + i * spacing, taken: false });
  }
}
