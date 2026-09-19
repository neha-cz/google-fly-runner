import {
  type Action, LOOKAHEAD, MAX_SPEED, OBS_OFFSET, COIN_OFFSET, OBSTACLES_PER_LANE, LANE_COUNT, ObstacleKind,
} from "../engine";
import type { Agent } from "./types";

interface Ob { kind: ObstacleKind | -1; dist: number }

function nextObstacle(s: Float32Array, lane: number, k = 0): Ob {
  const b = OBS_OFFSET + (lane * OBSTACLES_PER_LANE + k) * 4;
  const kind = s[b] ? ObstacleKind.Low : s[b + 1] ? ObstacleKind.High : s[b + 2] ? ObstacleKind.Full : -1;
  return { kind, dist: s[b + 3] * LOOKAHEAD };
}

/**
 * Scripted policy that reads only the public state vector. Exists to prove the API carries
 * enough information to play well, to generate the golden fixture, and as a baseline.
 * Timing constants are derived from the engine's jump/slide/lane-change durations.
 */
export class HeuristicAgent implements Agent {
  name = "heuristic";
  act(s: Float32Array, _tick?: number): Action {
    const lane = s[0] ? 0 : s[1] ? 1 : 2;
    const speed = s[8] * MAX_SPEED;
    const airborne = s[4] > 0.5, sliding = s[6] > 0.5;
    const midTween = Math.abs(s[3] * 1 - [-1, 0, 1][lane]) > 0.05;
    const o = nextObstacle(s, lane);

    // Threat handling in the current lane.
    if (o.kind === ObstacleKind.Full && o.dist < speed * 0.6 + 1) {
      const best = this.bestNeighbour(s, lane, speed);
      if (best !== lane) return best < lane ? "left" : "right";
    }
    if (o.kind === ObstacleKind.Low && !airborne && o.dist < speed * 0.2 + 0.4) return "jump";
    if (o.kind === ObstacleKind.High && !sliding && o.dist < speed * 0.25 + 0.4) return "slide";

    // Greedy coin collection when the neighbouring lane is safe.
    if (!airborne && !sliding && !midTween && o.dist > speed * 0.9) {
      const here = s[COIN_OFFSET + lane] * LOOKAHEAD;
      let best = lane, bestD = here - 2;
      for (const l of [lane - 1, lane + 1]) {
        if (l < 0 || l >= LANE_COUNT) continue;
        const d = s[COIN_OFFSET + l] * LOOKAHEAD;
        const no = nextObstacle(s, l);
        const safe = no.kind === -1 || no.dist > speed * 1.0;
        if (safe && d < bestD) { best = l; bestD = d; }
      }
      if (best !== lane) return best < lane ? "left" : "right";
    }
    return "noop";
  }

  private bestNeighbour(s: Float32Array, lane: number, speed: number): number {
    let best = lane, bestScore = -Infinity;
    for (const l of [lane - 1, lane + 1]) {
      if (l < 0 || l >= LANE_COUNT) continue;
      const o = nextObstacle(s, l);
      let score = o.kind === -1 ? 1000 : o.dist;
      if (o.kind === ObstacleKind.Low || o.kind === ObstacleKind.High) score += o.dist > speed * 0.15 ? 50 : 0;
      if (score > bestScore) { bestScore = score; best = l; }
    }
    return best;
  }
}
