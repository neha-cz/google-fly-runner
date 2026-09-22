import * as THREE from "three";

/** Small procedural sprites drawn with Canvas2D at startup (the big surfaces are generated assets, see assets.ts). */

function canvasTexture(w: number, h: number, paint: (ctx: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  paint(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** Cheap deterministic noise so sprites are identical run to run. */
function noise(seed: number) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

/** Cut end of a log: concentric growth rings. */
export function logEndTexture(): THREE.CanvasTexture {
  return canvasTexture(128, 128, ctx => {
    const r = noise(5);
    ctx.fillStyle = "#c9a66b"; ctx.fillRect(0, 0, 128, 128);
    for (let i = 60; i > 2; i -= 3 + r() * 3) {
      ctx.strokeStyle = r() < 0.5 ? "#9d7a45" : "#b48f58"; ctx.lineWidth = 1 + r() * 1.5;
      ctx.beginPath(); ctx.arc(64 + r() * 2, 64 + r() * 2, i, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.strokeStyle = "#6e4e2a"; ctx.lineWidth = 2;
    for (let i = 0; i < 4; i++) { const a = r() * Math.PI * 2; ctx.beginPath(); ctx.moveTo(64, 64); ctx.lineTo(64 + Math.cos(a) * 60, 64 + Math.sin(a) * 60); ctx.stroke(); }
    ctx.strokeStyle = "#3b2a17"; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(64, 64, 61, 0, Math.PI * 2); ctx.stroke();
  });
}

/** A single falling leaf, alpha-masked, for the particle system. */
export function leafSpriteTexture(): THREE.CanvasTexture {
  return canvasTexture(64, 64, ctx => {
    ctx.clearRect(0, 0, 64, 64);
    ctx.translate(32, 32); ctx.rotate(0.6);
    const g = ctx.createLinearGradient(-20, 0, 20, 0); g.addColorStop(0, "#d9a83a"); g.addColorStop(1, "#8f6a1c");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(0, -24); ctx.quadraticCurveTo(18, -6, 0, 24); ctx.quadraticCurveTo(-18, -6, 0, -24); ctx.fill();
    ctx.strokeStyle = "rgba(90,60,15,0.7)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(0, -22); ctx.lineTo(0, 22); ctx.stroke();
  });
}

/** Soft round glow for coin sparkles. */
export function sparkTexture(): THREE.CanvasTexture {
  return canvasTexture(32, 32, ctx => {
    const g = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    g.addColorStop(0, "rgba(255,245,200,1)"); g.addColorStop(0.4, "rgba(255,210,90,0.8)"); g.addColorStop(1, "rgba(255,180,40,0)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, 32, 32);
  });
}

/** Translucent fly wing with dark veins; alpha carries the wing outline. Root at the left edge. */
export function wingTexture(): THREE.CanvasTexture {
  return canvasTexture(256, 128, ctx => {
    ctx.clearRect(0, 0, 256, 128);
    ctx.beginPath(); ctx.moveTo(4, 64);
    ctx.bezierCurveTo(60, 4, 200, 6, 250, 50); ctx.bezierCurveTo(240, 100, 120, 126, 40, 96); ctx.bezierCurveTo(20, 88, 8, 78, 4, 64); ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 256, 0); g.addColorStop(0, "rgba(225,235,255,0.7)"); g.addColorStop(1, "rgba(200,220,255,0.45)");
    ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = "rgba(60,50,40,0.85)"; ctx.lineWidth = 2; ctx.stroke();
    ctx.lineWidth = 1.3;
    for (const [x1, y1, x2, y2] of [[4, 64, 250, 50], [4, 64, 200, 22], [4, 64, 230, 84], [30, 60, 150, 112], [90, 40, 120, 90], [150, 30, 175, 96]]) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
  });
}

/** Banded fly abdomen: dark chitin with lighter tan stripes running around the body. */
export function abdomenTexture(): THREE.CanvasTexture {
  return canvasTexture(64, 128, ctx => {
    const r = noise(9);
    ctx.fillStyle = "#3a2a18"; ctx.fillRect(0, 0, 64, 128);
    for (let y = 22; y < 118; y += 20) { ctx.fillStyle = "#6e5230"; ctx.fillRect(0, y, 64, 8); ctx.fillStyle = "#1f1409"; ctx.fillRect(0, y + 8, 64, 3); }
    ctx.fillStyle = "rgba(0,0,0,0.25)"; for (let i = 0; i < 60; i++) ctx.fillRect(r() * 64, r() * 128, 1, 2 + r() * 3);
  });
}
