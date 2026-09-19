import { type Snapshot, ObstacleKind, LANE_X, OBSTACLE_HALF_WIDTH, COIN_RADIUS, MAX_SPEED } from "../engine";
import { makeCam, project, type Cam } from "../engine/projection";
import { drawFly, drawJar, drawNewspaper, drawStickyStrip, drawDroplet } from "./sprites";
import type { GameRenderer } from "./three-renderer";

interface Particle { x: number; y: number; vx: number; vy: number; life: number }

/**
 * Canvas 2D fallback renderer (night-kitchen theme, `?renderer=2d`), used when WebGL is unavailable.
 * Purely cosmetic — the engine never depends on it.
 */
export class Renderer implements GameRenderer {
  private ctx: CanvasRenderingContext2D;
  private particles: Particle[] = [];
  private lastCoins = 0;
  private shake = 0;
  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  draw(s: Snapshot, label: string, fps: number): void {
    const { ctx } = this; const w = this.canvas.width, h = this.canvas.height;
    const cam = makeCam(w, h, s.player.z);
    if (s.coinsCollected > this.lastCoins) this.burst(cam, s);
    this.lastCoins = s.coinsCollected;
    if (!s.alive && this.shake === 0 && s.crashedInto) this.shake = 18;

    ctx.save();
    if (this.shake > 0) { ctx.translate((Math.random() - 0.5) * this.shake, (Math.random() - 0.5) * this.shake); this.shake -= 1; }
    this.background(cam, s);
    this.counter(cam, s);
    this.world(cam, s);
    this.particlesStep(cam);
    ctx.restore();
    this.hud(s, label, fps);
  }

  private background(cam: Cam, s: Snapshot) {
    const { ctx } = this; const { w, h, horizon } = cam;
    const g = ctx.createLinearGradient(0, 0, 0, horizon);
    g.addColorStop(0, "#050612"); g.addColorStop(1, "#1a1f3a");
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, horizon);
    // window with moonlight
    ctx.fillStyle = "#0e1330"; ctx.fillRect(w * 0.62, horizon * 0.18, w * 0.22, horizon * 0.62);
    ctx.fillStyle = "#e8ecff"; ctx.beginPath(); ctx.arc(w * 0.73, horizon * 0.42, 22, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#0e1330"; ctx.beginPath(); ctx.arc(w * 0.735, horizon * 0.40, 18, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#2a3160"; ctx.lineWidth = 4;
    ctx.strokeRect(w * 0.62, horizon * 0.18, w * 0.22, horizon * 0.62);
    ctx.beginPath(); ctx.moveTo(w * 0.73, horizon * 0.18); ctx.lineTo(w * 0.73, horizon * 0.80); ctx.stroke();
    // cupboards silhouette
    ctx.fillStyle = "#0b0e1f";
    for (let i = 0; i < 6; i++) ctx.fillRect(i * w / 6 + 6, horizon * 0.05, w / 6 - 12, horizon * 0.5 * (i % 2 ? 1 : 0.9));
    // fog band at horizon
    const f = ctx.createLinearGradient(0, horizon - 40, 0, horizon + 10);
    f.addColorStop(0, "rgba(26,31,58,0)"); f.addColorStop(1, "rgba(60,45,30,1)");
    ctx.fillStyle = f; ctx.fillRect(0, horizon - 40, w, 50);
    void s;
  }

  private counter(cam: Cam, s: Snapshot) {
    const { ctx } = this; const { w, h, horizon } = cam;
    const g = ctx.createLinearGradient(0, horizon, 0, h);
    g.addColorStop(0, "#3d2b1a"); g.addColorStop(1, "#8a5a32");
    ctx.fillStyle = g; ctx.fillRect(0, horizon, w, h - horizon);
    // lane edges
    ctx.strokeStyle = "rgba(255,220,170,0.25)"; ctx.lineWidth = 2;
    for (const ex of [-1.5, -0.5, 0.5, 1.5]) {
      const a = project(cam, ex, 0, s.player.z - 2.5), b = project(cam, ex, 0, s.player.z + 80);
      if (a && b) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    }
    // wood grain ticks scrolling with z
    ctx.strokeStyle = "rgba(0,0,0,0.18)"; ctx.lineWidth = 1;
    const start = Math.floor(s.player.z / 2) * 2;
    for (let z = start - 2; z < s.player.z + 80; z += 2) {
      const a = project(cam, -2.2, 0, z), b = project(cam, 2.2, 0, z);
      if (a && b) { ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
    }
    // edge of counter
    ctx.fillStyle = "#2a1d10";
    const l0 = project(cam, -2.2, 0, s.player.z - 2.5), l1 = project(cam, -2.2, 0, s.player.z + 80);
    const r0 = project(cam, 2.2, 0, s.player.z - 2.5), r1 = project(cam, 2.2, 0, s.player.z + 80);
    if (l0 && l1) { ctx.beginPath(); ctx.moveTo(0, h); ctx.lineTo(l0[0], l0[1]); ctx.lineTo(l1[0], l1[1]); ctx.lineTo(0, l1[1]); ctx.fill(); }
    if (r0 && r1) { ctx.beginPath(); ctx.moveTo(w, h); ctx.lineTo(r0[0], r0[1]); ctx.lineTo(r1[0], r1[1]); ctx.lineTo(w, r1[1]); ctx.fill(); }
  }

  private world(cam: Cam, s: Snapshot) {
    const { ctx } = this;
    type Item = { z: number; draw: () => void };
    const items: Item[] = [];
    for (const o of s.obstacles) {
      items.push({ z: o.z, draw: () => {
        const x = LANE_X[o.lane];
        const nl = project(cam, x - OBSTACLE_HALF_WIDTH, 0, o.z), nr = project(cam, x + OBSTACLE_HALF_WIDTH, 0, o.z);
        const fl = project(cam, x - OBSTACLE_HALF_WIDTH, 0, o.z + o.depth);
        if (!nl || !nr || !fl) return;
        const sw = nr[0] - nl[0], unit = cam.f / nl[2]; // pixels per world unit at the near face
        const flash = s.crashedInto === o;
        if (o.kind === ObstacleKind.Full) drawJar(ctx, nl[0], nl[1], sw, unit, fl[1] - nl[1], flash);
        else if (o.kind === ObstacleKind.Low) drawNewspaper(ctx, nl[0], nl[1], sw, unit, fl[1] - nl[1], flash);
        else drawStickyStrip(ctx, nl[0], nl[1], sw, unit, flash);
      } });
    }
    for (const c of s.coins) {
      if (c.taken) continue;
      items.push({ z: c.z, draw: () => {
        const p = project(cam, LANE_X[c.lane], 0.35, c.z); if (!p) return;
        drawDroplet(ctx, p[0], p[1], cam.f / p[2] * COIN_RADIUS, s.time);
      } });
    }
    const P = s.player;
    items.push({ z: P.z, draw: () => {
      const p = project(cam, P.x, P.y, P.z); if (!p) return;
      const unit = cam.f / p[2];
      const shadow = project(cam, P.x, 0, P.z);
      if (shadow) { ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.beginPath(); ctx.ellipse(shadow[0], shadow[1], unit * 0.35, unit * 0.12, 0, 0, Math.PI * 2); ctx.fill(); }
      drawFly(ctx, p[0], p[1], unit, {
        t: s.time, airborne: P.airborne, sliding: P.sliding, lean: P.leanDir * (1 - Math.abs(P.laneT * 2 - 1)), dead: !s.alive,
        speedFrac: s.speed / MAX_SPEED,
      });
    } });
    items.sort((a, b) => b.z - a.z);
    for (const it of items) it.draw();
  }

  private burst(cam: Cam, s: Snapshot) {
    const p = project(cam, s.player.x, s.player.y + 0.3, s.player.z); if (!p) return;
    for (let i = 0; i < 14; i++) {
      const a = Math.random() * Math.PI * 2, v = 1 + Math.random() * 3;
      this.particles.push({ x: p[0], y: p[1], vx: Math.cos(a) * v, vy: Math.sin(a) * v - 2, life: 1 });
    }
  }
  private particlesStep(cam: Cam) {
    const { ctx } = this;
    for (const p of this.particles) { p.x += p.vx; p.y += p.vy; p.vy += 0.15; p.life -= 0.04; }
    this.particles = this.particles.filter(p => p.life > 0);   // filter AFTER decrement: arc() throws on a negative radius
    for (const p of this.particles) {
      ctx.fillStyle = `rgba(200,230,255,${p.life})`; ctx.beginPath(); ctx.arc(p.x, p.y, 3 * p.life, 0, Math.PI * 2); ctx.fill();
    }
    void cam;
  }

  private hud(s: Snapshot, label: string, fps: number) {
    const { ctx } = this; const w = this.canvas.width, h = this.canvas.height;
    ctx.font = "bold 22px system-ui, sans-serif"; ctx.textBaseline = "top";
    ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fillRect(12, 12, 250, 84);
    ctx.fillStyle = "#fff";
    ctx.fillText(`${Math.floor(s.score).toLocaleString()} pts`, 24, 20);
    ctx.font = "15px system-ui, sans-serif"; ctx.fillStyle = "#cfd6ff";
    ctx.fillText(`${s.distance.toFixed(0)} m   ·   ${s.coinsCollected} sugar   ·   ${s.speed.toFixed(1)} m/s`, 24, 50);
    ctx.fillText(`driver: ${label}   ·   ${fps.toFixed(0)} fps`, 24, 72);
    if (!s.alive) {
      ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(0, h * 0.35, w, h * 0.3);
      ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.font = "bold 40px system-ui, sans-serif";
      ctx.fillText("SPLAT", w / 2, h * 0.40);
      ctx.font = "18px system-ui, sans-serif"; ctx.fillStyle = "#cfd6ff";
      ctx.fillText(`${s.distance.toFixed(0)} m · ${Math.floor(s.score)} pts · press R`, w / 2, h * 0.52);
      ctx.textAlign = "left";
    }
  }
}
