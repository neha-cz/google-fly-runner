/** Shared pseudo-3D camera used by both the retinal rasteriser and the canvas renderer. */
export const CAM_HEIGHT = 1.6;
export const CAM_BACK = 3.2;
export const HORIZON_FRAC = 0.42;

export interface Cam { w: number; h: number; f: number; camZ: number; cx: number; horizon: number }

export function makeCam(w: number, h: number, playerZ: number): Cam {
  return { w, h, f: h * 0.95, camZ: playerZ - CAM_BACK, cx: w / 2, horizon: h * HORIZON_FRAC };
}

/** Project a world point; returns null if behind the camera. */
export function project(c: Cam, x: number, y: number, z: number): [number, number, number] | null {
  const zr = z - c.camZ;
  if (zr < 0.15) return null;
  return [c.cx + c.f * x / zr, c.horizon - c.f * (y - CAM_HEIGHT) / zr, zr];
}
