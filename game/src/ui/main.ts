import { Runner, STATE_SIZE, STATE_LAYOUT, FRAME_W, FRAME_H, DT, type Action } from "../engine";
import { Renderer } from "../render/renderer";
import { ThreeRenderer, type GameRenderer } from "../render/three-renderer";
import { HeuristicAgent, RandomAgent, type Agent } from "../agents";
import { agentFromBundle, FlyAgent, type Bundle } from "../agents/neural";
import { drawActivity } from "./activity";

const canvas = document.getElementById("game") as HTMLCanvasElement;
const frameCanvas = document.getElementById("frame") as HTMLCanvasElement;
const modeSel = document.getElementById("mode") as HTMLSelectElement;
const seedInput = document.getElementById("seed") as HTMLInputElement;
const statsEl = document.getElementById("stats")!;
const stateEl = document.getElementById("state")!;

const params = new URLSearchParams(location.search);
let renderer: GameRenderer;
try {
  if (params.get("renderer") === "2d") throw new Error("2d requested");
  renderer = new ThreeRenderer(canvas);
} catch (e) {
  console.warn("falling back to 2D canvas renderer:", (e as Error).message);
  renderer = new Renderer(canvas);
  document.getElementById("hud")!.hidden = true; document.getElementById("overlay")!.hidden = true;
}
const fctx = frameCanvas.getContext("2d")!;
const fimg = fctx.createImageData(FRAME_W, FRAME_H);

let runner = new Runner(Number(seedInput.value) || 1);
const state = new Float32Array(STATE_SIZE);
const frame = new Uint8Array(FRAME_W * FRAME_H);
let agent: Agent | null = null;
let pendingHuman: Action = "noop";
let paused = false, tick = 0, deadSince = -1;
let acc = 0, last = performance.now(), fps = 60;

/** Registry so later phases (MLP baseline, fly agent) can plug in without touching the loop. */
export const drivers: Record<string, () => Agent | null> = {
  human: () => null,
  heuristic: () => new HeuristicAgent(),
  random: () => new RandomAgent(),
};

function setMode() { agent = drivers[modeSel.value]?.() ?? null; document.getElementById("brain")!.hidden = !(agent instanceof FlyAgent); }

/** Trained agents exported by `flybrain export` (public/agents/index.json). */
async function loadExportedAgents() {
  try {
    const idx: { file: string; name: string; kind: string; label: string }[] = await (await fetch("agents/index.json")).json();
    for (const e of idx) {
      const bundle: Bundle = await (await fetch(`agents/${e.file}`)).json();
      drivers[e.name] = () => agentFromBundle(bundle);
      const opt = document.createElement("option"); opt.value = e.name; opt.textContent = (e.kind === "fly" ? "🪰 " : "🤖 ") + e.label; modeSel.appendChild(opt);
    }
  } catch (err) { console.warn("no exported agents found", err); }
}
function restart() { runner.reset(Number(seedInput.value) || 1); tick = 0; deadSince = -1; pendingHuman = "noop"; agent?.reset?.(); }
modeSel.addEventListener("change", () => { setMode(); restart(); });
seedInput.addEventListener("change", restart);
document.getElementById("restart")!.addEventListener("click", restart);

const keymap: Record<string, Action> = { ArrowUp: "jump", Space: "jump", KeyW: "jump", ArrowDown: "slide", KeyS: "slide", ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right" };
window.addEventListener("keydown", e => {
  if (e.code === "KeyR") { restart(); return; }
  if (e.code === "KeyP") { paused = !paused; return; }
  const a = keymap[e.code]; if (!a) return;
  e.preventDefault();
  if (!runner.alive) { restart(); return; }
  pendingHuman = a;
});

function stepOnce() {
  if (!runner.alive) {
    if (deadSince < 0) deadSince = performance.now();
    if (agent && performance.now() - deadSince > 1800) restart();
    return;
  }
  runner.getState(state);
  const a: Action = agent ? agent.act(state, tick) : pendingHuman;
  pendingHuman = "noop";
  runner.act(a); runner.step(); tick++;
}

function drawSide() {
  runner.getFrame(frame);
  for (let i = 0; i < frame.length; i++) { const v = frame[i]; fimg.data[i * 4] = v; fimg.data[i * 4 + 1] = v; fimg.data[i * 4 + 2] = v; fimg.data[i * 4 + 3] = 255; }
  fctx.putImageData(fimg, 0, 0);
  runner.getState(state);
  const s = runner.snapshot();
  statsEl.innerHTML = [["distance", `${s.distance.toFixed(1)} m`], ["score", Math.floor(s.score)], ["sugar", s.coinsCollected], ["speed", `${s.speed.toFixed(2)} m/s`], ["time", `${s.time.toFixed(1)} s`], ["obstacles live", s.obstacles.length], ["status", s.alive ? (paused ? "paused" : "running") : "splat"]]
    .map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");
  if (agent instanceof FlyAgent) drawActivity(document.getElementById("activity") as HTMLCanvasElement, agent, document.getElementById("brain-label")!);
  if (tick % 6 === 0) stateEl.textContent = Array.from(state).map((v, i) => `${STATE_LAYOUT[i].padEnd(12)} ${v >= 0 ? " " : ""}${v.toFixed(3)}`).join("\n");
}

function loop(now: number) {
  const dtReal = Math.min(0.1, (now - last) / 1000); last = now;
  fps = fps * 0.95 + (1 / Math.max(dtReal, 1e-3)) * 0.05;
  if (!paused) { acc += dtReal; while (acc >= DT) { stepOnce(); acc -= DT; } }
  // A cosmetic exception must never stop the simulation loop.
  try { renderer.draw(runner.snapshot(), agent ? agent.name : "human", fps); drawSide(); }
  catch (err) { console.error("render error", err); }
  requestAnimationFrame(loop);
}
// URL params for demos/automation: ?driver=heuristic&seed=42&skip=30&renderer=2d
if (params.get("driver") && drivers[params.get("driver")!]) modeSel.value = params.get("driver")!;
if (params.get("seed")) seedInput.value = params.get("seed")!;
await loadExportedAgents();
if (params.get("driver") && drivers[params.get("driver")!]) modeSel.value = params.get("driver")!;
setMode(); restart();
// ?skip=8 fast-forwards 8 s of simulated play (agent-driven if a driver is selected)
const skip = Number(params.get("skip") || 0);
for (let i = 0; i < Math.round(skip / DT) && runner.alive; i++) stepOnce();
requestAnimationFrame(loop);
