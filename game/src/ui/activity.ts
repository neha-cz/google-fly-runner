import type { FlyAgent } from "../agents/neural";
import { ACTIONS } from "../engine";

const GROUP_ORDER = ["photoreceptor", "lamina", "medulla", "motion", "looming", "optic_flow", "anterior_visual_metu", "anterior_visual_tubu", "ring", "central_complex", "descending"];
const SHORT: Record<string, string> = { photoreceptor: "R1-8", lamina: "lamina", medulla: "medulla", motion: "T4/T5", looming: "LC/LPLC", optic_flow: "HS/VS", anterior_visual_metu: "MeTu", anterior_visual_tubu: "TuBu", ring: "ER", central_complex: "CX", descending: "DN" };
const smooth = new Map<string, number>();

/** Live panel: mean window rate per cell-type group (a signal-flow bar row), the descending-neuron
 * pools the readout sees, and the action logits. Values are smoothed for readability. */
export function drawActivity(canvas: HTMLCanvasElement, agent: FlyAgent, label: HTMLElement): void {
  const ctx = canvas.getContext("2d")!, W = canvas.width, H = canvas.height;
  const { groups, dns } = agent.groupActivity();
  const ema = (k: string, v: number) => { const p = smooth.get(k) ?? v; const s = p + (v - p) * 0.3; smooth.set(k, s); return s; };
  label.textContent = `${agent.b.graph.pooled ? "pooled" : "per-neuron"} MaleCNS ${agent.b.graph.version} substrate · ${agent.n} nodes · ${agent.b.edges.w.length} edges`;
  ctx.fillStyle = "#0e1018"; ctx.fillRect(0, 0, W, H);
  ctx.font = "10px system-ui, sans-serif"; ctx.textBaseline = "middle";

  // 1. signal flow: groups in anatomical order
  const gs = GROUP_ORDER.filter(g => g in groups);
  const gMax = Math.max(1e-6, ...gs.map(g => ema("g:" + g, groups[g])));
  const bw = (W - 10) / gs.length;
  ctx.fillStyle = "#8a93b8"; ctx.textAlign = "left"; ctx.fillText("mean rate by cell-type group (input → DN)", 6, 9);
  gs.forEach((g, i) => {
    const v = ema("g:" + g, groups[g]) / gMax, x = 5 + i * bw;
    const hue = 200 - 160 * (i / Math.max(1, gs.length - 1));
    ctx.fillStyle = `hsl(${hue} 70% 55%)`; ctx.fillRect(x + 2, 62 - v * 44, bw - 4, v * 44);
    ctx.fillStyle = "#c8cde6"; ctx.textAlign = "center"; ctx.save(); ctx.translate(x + bw / 2, 72); ctx.fillText(SHORT[g] ?? g, 0, 0); ctx.restore();
  });

  // 2. descending neurons (what the readout sees)
  ctx.fillStyle = "#8a93b8"; ctx.textAlign = "left"; ctx.fillText("descending-neuron pools → readout", 6, 90);
  const dMax = Math.max(1e-6, ...dns.map(d => ema("d:" + d.name, d.rate)));
  const dw = (W - 10) / Math.max(1, dns.length);
  dns.forEach((d, i) => {
    const v = ema("d:" + d.name, d.rate) / dMax, x = 5 + i * dw;
    ctx.fillStyle = d.name.endsWith("_L") ? "#f0a35a" : "#5ab8f0"; ctx.fillRect(x + 1, 150 - v * 50, dw - 2, v * 50);
    if (dns.length <= 24) { ctx.fillStyle = "#c8cde6"; ctx.save(); ctx.translate(x + dw / 2, 158); ctx.rotate(-Math.PI / 2.6); ctx.textAlign = "right"; ctx.fillText(d.name.replace("_", " "), 0, 0); ctx.restore(); }
  });

  // 3. action logits
  const lg = agent.logits; const lmin = Math.min(...lg), lmax = Math.max(...lg), span = Math.max(1e-6, lmax - lmin);
  ctx.fillStyle = "#8a93b8"; ctx.textAlign = "left"; ctx.fillText("action logits", 6, 190);
  const aw = (W - 10) / 5;
  ACTIONS.forEach((a, i) => {
    const v = (lg[i] - lmin) / span, x = 5 + i * aw, chosen = i === agent.lastAction;
    ctx.fillStyle = chosen ? "#ffd76a" : "#3a4160"; ctx.fillRect(x + 6, 222 - v * 24 - 2, aw - 12, v * 24 + 2);
    ctx.fillStyle = chosen ? "#ffd76a" : "#8a93b8"; ctx.textAlign = "center"; ctx.fillText(a, x + aw / 2, 227);
  });
}
