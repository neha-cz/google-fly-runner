/** Procedural sprites. All sizes are in pixels; `unit` = pixels per world unit at the sprite's depth. */

export interface FlyPose { t: number; airborne: boolean; sliding: boolean; lean: number; dead: boolean; speedFrac: number }

export function drawFly(ctx: CanvasRenderingContext2D, x: number, y: number, unit: number, p: FlyPose): void {
  const s = unit * 0.55; // body length in px
  ctx.save();
  ctx.translate(x, y - s * 0.35);
  ctx.rotate(p.lean * 0.45);
  if (p.sliding) ctx.scale(1.25, 0.45);
  if (p.dead) ctx.rotate(Math.PI);
  const bob = p.airborne || p.sliding ? 0 : Math.sin(p.t * 22) * s * 0.05;
  ctx.translate(0, bob);

  // legs (run cycle)
  ctx.strokeStyle = "#2b1d12"; ctx.lineWidth = Math.max(1, s * 0.05);
  for (let i = 0; i < 3; i++) {
    const ph = Math.sin(p.t * 26 + i * 2.1) * (p.airborne ? 0.2 : 0.5);
    for (const side of [-1, 1]) {
      ctx.beginPath(); ctx.moveTo(side * s * 0.15, s * 0.1 - i * s * 0.08);
      ctx.lineTo(side * s * (0.45 + ph * 0.1), s * (0.32 + ph * 0.15));
      ctx.lineTo(side * s * 0.5, s * 0.5); ctx.stroke();
    }
  }
  // wings
  const flap = p.airborne ? 0.9 : 0.35 + Math.abs(Math.sin(p.t * 60)) * 0.5;
  ctx.fillStyle = "rgba(200,220,255,0.35)"; ctx.strokeStyle = "rgba(220,235,255,0.6)"; ctx.lineWidth = 1;
  for (const side of [-1, 1]) {
    ctx.save(); ctx.rotate(side * flap * 0.9);
    ctx.beginPath(); ctx.ellipse(side * s * 0.42, -s * 0.15, s * 0.42, s * 0.16, side * 0.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  // abdomen + thorax
  const g = ctx.createRadialGradient(-s * 0.05, -s * 0.05, s * 0.05, 0, 0, s * 0.5);
  g.addColorStop(0, "#8b6b3e"); g.addColorStop(1, "#3a2814");
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.ellipse(0, s * 0.12, s * 0.22, s * 0.34, 0, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = "#1e140a"; ctx.lineWidth = Math.max(1, s * 0.03);
  for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(-s * 0.18, s * (0.15 + i * 0.09)); ctx.lineTo(s * 0.18, s * (0.15 + i * 0.09)); ctx.stroke(); }
  ctx.fillStyle = "#5a4025"; ctx.beginPath(); ctx.ellipse(0, -s * 0.18, s * 0.2, s * 0.2, 0, 0, Math.PI * 2); ctx.fill();
  // head + eyes
  ctx.fillStyle = "#4a3320"; ctx.beginPath(); ctx.arc(0, -s * 0.4, s * 0.15, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = p.dead ? "#777" : "#d6323a";
  ctx.beginPath(); ctx.arc(-s * 0.1, -s * 0.42, s * 0.08, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(s * 0.1, -s * 0.42, s * 0.08, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

export function drawJar(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, unit: number, depthPx: number, flash: boolean): void {
  const hgt = unit * 1.4;
  ctx.save();
  // back face offset for cheap depth
  ctx.fillStyle = "rgba(150,200,230,0.25)"; ctx.fillRect(x + w * 0.08, y + depthPx - hgt * 0.98, w * 0.84, hgt * 0.98);
  const g = ctx.createLinearGradient(x, 0, x + w, 0);
  g.addColorStop(0, "rgba(190,225,245,0.55)"); g.addColorStop(0.35, "rgba(240,250,255,0.75)"); g.addColorStop(1, "rgba(160,200,230,0.55)");
  ctx.fillStyle = flash ? "rgba(255,120,120,0.85)" : g;
  ctx.beginPath(); ctx.roundRect(x, y - hgt, w, hgt, w * 0.12); ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = Math.max(1, unit * 0.03); ctx.stroke();
  // lid
  ctx.fillStyle = "#b8863b"; ctx.fillRect(x - w * 0.05, y - hgt - unit * 0.12, w * 1.1, unit * 0.16);
  ctx.fillStyle = "#8a6128"; ctx.fillRect(x - w * 0.05, y - hgt + unit * 0.02, w * 1.1, unit * 0.04);
  // label
  ctx.fillStyle = "rgba(230,120,90,0.8)"; ctx.fillRect(x + w * 0.15, y - hgt * 0.62, w * 0.7, hgt * 0.28);
  ctx.restore();
}

export function drawNewspaper(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, unit: number, depthPx: number, flash: boolean): void {
  const r = unit * 0.34;
  ctx.save();
  ctx.fillStyle = flash ? "#ff8080" : "#d9cfb8"; ctx.strokeStyle = "#6b6252"; ctx.lineWidth = Math.max(1, unit * 0.02);
  // cylinder body lying across the lane
  ctx.beginPath(); ctx.roundRect(x, y - r * 2 + depthPx * 0.2, w, r * 2 - depthPx * 0.2 + 1, r * 0.9); ctx.fill(); ctx.stroke();
  // rolled end
  ctx.fillStyle = "#e8e0cc"; ctx.beginPath(); ctx.ellipse(x + w * 0.5, y - r + depthPx * 0.1, w * 0.5, r * 0.95, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.strokeStyle = "#8a8172";
  for (let i = 1; i <= 3; i++) { ctx.beginPath(); ctx.ellipse(x + w * 0.5, y - r + depthPx * 0.1, w * 0.5 * (i / 4), r * 0.95 * (i / 4), 0, 0, Math.PI * 2); ctx.stroke(); }
  // headline lines
  ctx.strokeStyle = "#3a3630"; ctx.lineWidth = Math.max(1, unit * 0.015);
  for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(x + w * 0.1, y - r * 1.5 + i * r * 0.35); ctx.lineTo(x + w * 0.9, y - r * 1.5 + i * r * 0.35); ctx.stroke(); }
  ctx.restore();
}

export function drawStickyStrip(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, unit: number, flash: boolean): void {
  const bottom = y - unit * 0.7, top = y - unit * 1.7;
  ctx.save();
  // hanging cord to top of screen
  ctx.strokeStyle = "rgba(220,220,220,0.5)"; ctx.lineWidth = Math.max(1, unit * 0.02);
  ctx.beginPath(); ctx.moveTo(x + w / 2, top); ctx.lineTo(x + w / 2, -10); ctx.stroke();
  const g = ctx.createLinearGradient(x, 0, x + w, 0);
  g.addColorStop(0, "#c9a227"); g.addColorStop(0.5, "#f3d35a"); g.addColorStop(1, "#b8901f");
  ctx.fillStyle = flash ? "#ff8080" : g;
  ctx.fillRect(x + w * 0.2, top, w * 0.6, bottom - top);
  // curl at the bottom
  ctx.beginPath(); ctx.ellipse(x + w / 2, bottom, w * 0.32, unit * 0.06, 0, 0, Math.PI * 2); ctx.fill();
  // specks (caught flies, ominously)
  ctx.fillStyle = "#3a2a10";
  for (let i = 0; i < 6; i++) { const yy = top + ((i * 37) % 100) / 100 * (bottom - top); ctx.beginPath(); ctx.arc(x + w * (0.3 + ((i * 53) % 40) / 100), yy, Math.max(1, unit * 0.02), 0, Math.PI * 2); ctx.fill(); }
  ctx.restore();
}

export function drawDroplet(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, t: number): void {
  const wob = 1 + Math.sin(t * 6 + x) * 0.08;
  ctx.save();
  ctx.shadowColor = "rgba(160,220,255,0.9)"; ctx.shadowBlur = r * 1.5;
  const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
  g.addColorStop(0, "#ffffff"); g.addColorStop(0.5, "#bfe7ff"); g.addColorStop(1, "#5fb3e6");
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.moveTo(x, y - r * 1.4 * wob);
  ctx.bezierCurveTo(x + r * 1.1, y - r * 0.2, x + r * 0.9, y + r, x, y + r);
  ctx.bezierCurveTo(x - r * 0.9, y + r, x - r * 1.1, y - r * 0.2, x, y - r * 1.4 * wob);
  ctx.fill();
  ctx.restore();
}
