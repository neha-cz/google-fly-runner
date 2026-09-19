import * as THREE from "three";

/** Procedural textures drawn with Canvas2D at startup. No external assets. */

function canvasTexture(w: number, h: number, paint: (ctx: CanvasRenderingContext2D) => void, repeat = 1): THREE.CanvasTexture {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  paint(c.getContext("2d")!);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

/** Cheap deterministic noise so textures are identical run to run. */
function noise(seed: number) { let s = seed >>> 0 || 1; return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }

export function stoneFloorTexture(): THREE.CanvasTexture {
  return canvasTexture(256, 256, ctx => {
    const r = noise(11);
    ctx.fillStyle = "#6b6153"; ctx.fillRect(0, 0, 256, 256);
    const tile = 64;
    for (let y = 0; y < 256; y += tile) for (let x = 0; x < 256; x += tile) {
      const g = 80 + r() * 40, m = r() < 0.25;
      ctx.fillStyle = m ? `rgb(${g - 20},${g + 10},${g - 30})` : `rgb(${g + 10},${g},${g - 15})`;
      ctx.fillRect(x + 3, y + 3, tile - 6, tile - 6);
      ctx.fillStyle = "rgba(0,0,0,0.18)";
      for (let i = 0; i < 6; i++) ctx.fillRect(x + 4 + r() * (tile - 12), y + 4 + r() * (tile - 12), 2 + r() * 6, 1 + r() * 3);
    }
    ctx.strokeStyle = "#2f2a22"; ctx.lineWidth = 3;
    for (let i = 0; i <= 4; i++) { ctx.beginPath(); ctx.moveTo(i * tile, 0); ctx.lineTo(i * tile, 256); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * tile); ctx.lineTo(256, i * tile); ctx.stroke(); }
  });
}

export function brickTexture(): THREE.CanvasTexture {
  return canvasTexture(256, 256, ctx => {
    const r = noise(23);
    ctx.fillStyle = "#3a3128"; ctx.fillRect(0, 0, 256, 256);
    const bh = 32, bw = 64;
    for (let row = 0; row < 8; row++) {
      const off = row % 2 ? bw / 2 : 0;
      for (let x = -bw; x < 256 + bw; x += bw) {
        const g = 125 + r() * 50, mossy = r() < 0.2;
        ctx.fillStyle = mossy ? `rgb(${g - 35},${g + 5},${g - 45})` : `rgb(${g},${g - 8},${g - 25})`;
        ctx.fillRect(x + off + 2, row * bh + 2, bw - 4, bh - 4);
        ctx.fillStyle = "rgba(0,0,0,0.15)"; ctx.fillRect(x + off + 2, row * bh + bh - 8, bw - 4, 6);
      }
    }
  });
}

export function barkTexture(): THREE.CanvasTexture {
  return canvasTexture(64, 128, ctx => {
    const r = noise(37);
    ctx.fillStyle = "#4a3320"; ctx.fillRect(0, 0, 64, 128);
    for (let i = 0; i < 40; i++) { ctx.fillStyle = r() < 0.5 ? "#3a2716" : "#5a4028"; ctx.fillRect(r() * 64, r() * 128, 2 + r() * 3, 8 + r() * 30); }
  }, 1);
}

export function leafTexture(): THREE.CanvasTexture {
  return canvasTexture(64, 64, ctx => {
    const r = noise(41);
    ctx.fillStyle = "#1f4a22"; ctx.fillRect(0, 0, 64, 64);
    for (let i = 0; i < 80; i++) { ctx.fillStyle = r() < 0.5 ? "#2c6a2e" : "#173a1a"; ctx.beginPath(); ctx.arc(r() * 64, r() * 64, 3 + r() * 5, 0, Math.PI * 2); ctx.fill(); }
  });
}

export function goldTexture(): THREE.CanvasTexture {
  return canvasTexture(64, 64, ctx => {
    const g = ctx.createRadialGradient(32, 32, 4, 32, 32, 32);
    g.addColorStop(0, "#fff2b0"); g.addColorStop(0.6, "#f0b429"); g.addColorStop(1, "#a86f0a");
    ctx.fillStyle = g; ctx.fillRect(0, 0, 64, 64);
    ctx.strokeStyle = "#7a5008"; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(32, 32, 22, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#7a5008"; ctx.font = "bold 26px serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("✦", 32, 33);
  });
}
