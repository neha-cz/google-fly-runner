import type { Snapshot } from "./types";
import { ObstacleKind } from "./types";
import { LANE_X, OBSTACLE_HALF_WIDTH, COIN_RADIUS } from "./constants";
import { makeCam, project } from "./projection";

/** World-space heights of obstacle silhouettes, by kind: [bottom, top]. */
export const OBSTACLE_HEIGHT: Record<ObstacleKind, [number, number]> = {
  [ObstacleKind.Low]: [0, 0.35],
  [ObstacleKind.High]: [0.7, 1.7],
  [ObstacleKind.Full]: [0, 1.4],
};

function fillRect(buf: Uint8Array, w: number, h: number, x0: number, y0: number, x1: number, y1: number, v: number) {
  const ax = Math.max(0, Math.floor(Math.min(x0, x1))), bx = Math.min(w, Math.ceil(Math.max(x0, x1)));
  const ay = Math.max(0, Math.floor(Math.min(y0, y1))), by = Math.min(h, Math.ceil(Math.max(y0, y1)));
  for (let y = ay; y < by; y++) for (let x = ax; x < bx; x++) buf[y * w + x] = v;
}

/**
 * Software rasteriser producing a grayscale view from behind the player. This is the
 * "retina" input for the visual-system variant of the agent. It has no DOM dependency
 * and is identical in Node and the browser.
 */
export function rasterize(s: Snapshot, buf: Uint8Array, w: number, h: number): void {
  const cam = makeCam(w, h, s.player.z);
  // sky / ground
  const hz = Math.floor(cam.horizon);
  for (let y = 0; y < h; y++) {
    const v = y < hz ? 20 : 45 + Math.floor(40 * (y - hz) / (h - hz));
    buf.fill(v, y * w, y * w + w);
  }
  // lane edges
  for (const ex of [-1.5, -0.5, 0.5, 1.5]) {
    for (let z = s.player.z - 2; z < s.player.z + 60; z += 0.5) {
      const p = project(cam, ex, 0, z); if (!p) continue;
      const xi = Math.round(p[0]), yi = Math.round(p[1]);
      if (xi >= 0 && xi < w && yi >= 0 && yi < h) buf[yi * w + xi] = 80;
    }
  }
  // obstacles & coins, far to near
  const items: { z: number; draw: () => void }[] = [];
  for (const o of s.obstacles) {
    const [b, t] = OBSTACLE_HEIGHT[o.kind];
    const lum = o.kind === ObstacleKind.Low ? 140 : o.kind === ObstacleKind.High ? 200 : 255;
    items.push({ z: o.z, draw: () => {
      const x = LANE_X[o.lane];
      const a = project(cam, x - OBSTACLE_HALF_WIDTH, t, o.z), c = project(cam, x + OBSTACLE_HALF_WIDTH, b, o.z);
      if (a && c) fillRect(buf, w, h, a[0], a[1], c[0], c[1], lum);
    } });
  }
  for (const c of s.coins) {
    if (c.taken) continue;
    items.push({ z: c.z, draw: () => {
      const x = LANE_X[c.lane];
      const a = project(cam, x - COIN_RADIUS * 0.5, 0.45, c.z), b = project(cam, x + COIN_RADIUS * 0.5, 0.15, c.z);
      if (a && b) fillRect(buf, w, h, a[0], a[1], b[0], b[1], 230);
    } });
  }
  items.sort((p, q) => q.z - p.z);
  for (const it of items) it.draw();
  // player silhouette
  const P = s.player;
  const ph = P.sliding ? 0.25 : 0.55;
  const a = project(cam, P.x - 0.28, P.y + ph, P.z), b = project(cam, P.x + 0.28, P.y, P.z);
  if (a && b) fillRect(buf, w, h, a[0], a[1], b[0], b[1], 100);
}
