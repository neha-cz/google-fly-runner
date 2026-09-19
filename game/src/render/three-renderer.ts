import * as THREE from "three";
import { type Snapshot, type Obstacle, type Coin, ObstacleKind, LANE_X, OBSTACLE_HALF_WIDTH, MAX_SPEED } from "../engine";
import { stoneFloorTexture, brickTexture, barkTexture, leafTexture, goldTexture } from "./textures";
import { Explorer } from "./explorer";

const SEG = 6;             // length of one recycled wall/jungle segment
const SEGS = 20;           // segments kept alive per side (covers ~120 units ahead)
const FLOOR_LEN = 200, FLOOR_TILE = 2;
const PATH_HALF = 1.9;     // half-width of the stone path
const WALL_X = 2.6;

/** The chase camera looks down +z, which puts world +x on screen-left; mirror x so lane 0 (x=-1) is on the left. */
const X = (x: number) => -x;
const hash = (n: number) => { let x = (n | 0) * 2654435761; x ^= x >>> 15; x = Math.imul(x, 2246822519); x ^= x >>> 13; return (x >>> 0) / 4294967296; };

export interface GameRenderer { draw(s: Snapshot, label: string, fps: number): void }

/**
 * three.js chase-camera renderer: temple ruin path through jungle, running explorer, log /
 * beam / wall obstacles, gold coins. Purely cosmetic; the engine never depends on it.
 */
export class ThreeRenderer implements GameRenderer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private floorTex: THREE.CanvasTexture;
  private ground: THREE.Group;
  private segments: THREE.Group[] = [];
  private explorer: Explorer;
  private shadow: THREE.Mesh;
  private obstacleMeshes = new WeakMap<Obstacle, THREE.Group>();
  private coinMeshes = new WeakMap<Coin, THREE.Mesh>();
  private live = new Set<THREE.Object3D>();
  private geo: { log: THREE.Group; beam: THREE.Group; wall: THREE.Group; coin: THREE.Mesh };
  private sparks: THREE.Points; private sparkVel: Float32Array; private sparkLife: Float32Array;
  private lastCoins = 0; private shake = 0; private deadT = 0;
  private hud = { score: document.getElementById("hud-score")!, line: document.getElementById("hud-line")!, driver: document.getElementById("hud-driver")!, overlay: document.getElementById("overlay")!, sub: document.getElementById("overlay-sub")! };

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(canvas.width, canvas.height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.05;
    this.camera = new THREE.PerspectiveCamera(62, canvas.width / canvas.height, 0.1, 160);

    const sky = new THREE.Color(0xc2d3c0);
    this.scene.background = sky; this.scene.fog = new THREE.Fog(sky, 14, 80);
    this.scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x3a4a2a, 0.9));
    const sun = new THREE.DirectionalLight(0xfff1d0, 1.6); sun.position.set(-6, 14, -8); this.scene.add(sun);

    // ground: a group that follows the player along z; the stone texture offset keeps tiles world-fixed
    this.ground = new THREE.Group(); this.scene.add(this.ground);
    this.floorTex = stoneFloorTexture(); this.floorTex.repeat.set(PATH_HALF * 2 / FLOOR_TILE, FLOOR_LEN / FLOOR_TILE);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(PATH_HALF * 2, FLOOR_LEN), new THREE.MeshStandardMaterial({ map: this.floorTex, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2; this.ground.add(floor);
    const grooveMat = new THREE.MeshStandardMaterial({ color: 0x2b261f, roughness: 1 });
    for (const x of [-0.5, 0.5]) { const g = new THREE.Mesh(new THREE.PlaneGeometry(0.05, FLOOR_LEN), grooveMat); g.rotation.x = -Math.PI / 2; g.position.set(x, 0.004, 0); this.ground.add(g); }
    const curbMat = new THREE.MeshStandardMaterial({ color: 0x7a7060, roughness: 1 });
    for (const x of [-PATH_HALF - 0.12, PATH_HALF + 0.12]) { const c = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.12, FLOOR_LEN), curbMat); c.position.set(x, 0.06, 0); this.ground.add(c); }
    const earth = new THREE.Mesh(new THREE.PlaneGeometry(80, FLOOR_LEN), new THREE.MeshStandardMaterial({ color: 0x24301a, roughness: 1 }));
    earth.rotation.x = -Math.PI / 2; earth.position.y = -0.03; this.ground.add(earth);

    // recycled side segments
    const brick = brickTexture(), bark = barkTexture(), leaf = leafTexture();
    const wallMat = new THREE.MeshStandardMaterial({ map: brick, roughness: 0.9 });
    const barkMat = new THREE.MeshStandardMaterial({ map: bark, roughness: 1 });
    const leafMat = new THREE.MeshStandardMaterial({ map: leaf, roughness: 1 });
    const frondMat = new THREE.MeshStandardMaterial({ color: 0x4a9a42, roughness: 0.9, side: THREE.DoubleSide });
    const pillarGeo = new THREE.CylinderGeometry(0.28, 0.34, 3.2, 8);
    for (let i = 0; i < SEGS * 2; i++) {
      const side = i < SEGS ? -1 : 1;
      const g = new THREE.Group(); g.userData.side = side;
      const wall = new THREE.Mesh(new THREE.BoxGeometry(0.8, 2.2, SEG), wallMat); wall.position.set(side * WALL_X, 1.1, SEG / 2); wall.name = "wall"; g.add(wall);
      const broken = new THREE.Mesh(new THREE.BoxGeometry(0.8, 1.1, SEG), wallMat); broken.position.set(side * WALL_X, 0.55, SEG / 2); broken.name = "broken"; g.add(broken);
      const pillar = new THREE.Mesh(pillarGeo, wallMat); pillar.position.set(side * (WALL_X - 0.2), 1.6, 0.6); pillar.name = "pillar"; g.add(pillar);
      const cap = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.25, 0.9), wallMat); cap.position.set(side * (WALL_X - 0.2), 3.3, 0.6); cap.name = "pillar"; g.add(cap);
      for (let t = 0; t < 3; t++) {
        const tree = new THREE.Group(); tree.name = `tree${t}`;
        if (t === 1) {                                   // palm
          const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.18, 6, 6), barkMat); trunk.position.y = 3; trunk.rotation.z = side * 0.12; tree.add(trunk);
          for (let k = 0; k < 7; k++) { const f = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.04, 2.4), frondMat); f.position.set(side * 0.35, 5.9, 0); f.rotation.set(0.5, k * Math.PI * 2 / 7, 0); f.translateZ(1.1); tree.add(f); }
        } else {                                         // broadleaf
          const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.4, 3.6, 7), barkMat); trunk.position.y = 1.8; tree.add(trunk);
          for (const [dx, dy, dz, r] of [[0, 4.2, 0, 1.7], [1.1, 3.6, 0.4, 1.2], [-1.0, 3.8, -0.5, 1.3], [0.2, 5.0, 0.6, 1.1]]) { const c = new THREE.Mesh(new THREE.SphereGeometry(r, 8, 6), leafMat); c.position.set(dx, dy, dz); tree.add(c); }
        }
        tree.position.set(side * (WALL_X + 1.4 + t * 1.9), 0, (t + 0.5) * SEG / 3); g.add(tree);
      }
      const bush = new THREE.Mesh(new THREE.SphereGeometry(0.7, 7, 6), leafMat); bush.position.set(side * (WALL_X + 0.7), 0.3, SEG * 0.7); bush.name = "bush"; g.add(bush);
      this.segments.push(g); this.scene.add(g);
    }

    // obstacle prototypes
    const logMat = barkMat, stoneMat = new THREE.MeshStandardMaterial({ map: brick, roughness: 0.85, color: 0xb8b0a0 });
    const log = new THREE.Group();
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, OBSTACLE_HALF_WIDTH * 2 + 0.3, 10), logMat); trunk.rotation.z = Math.PI / 2; trunk.position.y = 0.3; log.add(trunk);
    for (const sx of [-1, 1]) { const stub = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.1, 0.5, 5), logMat); stub.position.set(sx * 0.3, 0.55, 0.05 * sx); stub.rotation.z = sx * 0.5; log.add(stub); }
    const beam = new THREE.Group();
    for (const sx of [-1, 1]) { const post = new THREE.Mesh(new THREE.BoxGeometry(0.16, 1.8, 0.16), stoneMat); post.position.set(sx * (OBSTACLE_HALF_WIDTH + 0.02), 0.9, 0); beam.add(post); }
    const bar = new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2 + 0.3, 0.32, 0.36), stoneMat); bar.position.y = 0.92; beam.add(bar);
    const spikes = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.25, 4), stoneMat); for (let i = -2; i <= 2; i++) { const s = spikes.clone(); s.position.set(i * 0.2, 0.62, 0); s.rotation.x = Math.PI; bar.add(s); }
    const vine = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.6, 4), leafMat); for (let i = 0; i < 3; i++) { const v = vine.clone(); v.position.set(-0.3 + i * 0.3, 1.35, 0.2); beam.add(v); }
    const wall = new THREE.Group();
    const block = new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2, 1.45, 0.8), stoneMat); block.position.set(0, 0.725, 0.4); wall.add(block);
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.2, 8, 6), new THREE.MeshStandardMaterial({ color: 0xe8dcc0, roughness: 0.6 })); skull.position.set(0, 1.55, 0.4); wall.add(skull);
    const moss = new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2 + 0.02, 0.35, 0.82), leafMat); moss.position.set(0, 1.3, 0.4); wall.add(moss);
    const coin = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.06, 20), new THREE.MeshStandardMaterial({ map: goldTexture(), metalness: 0.6, roughness: 0.3, emissive: 0x553300 }));
    coin.rotation.x = Math.PI / 2;
    this.geo = { log, beam, wall, coin };

    this.explorer = new Explorer(); this.scene.add(this.explorer.group);
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(0.32, 16), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.4 }));
    this.shadow.rotation.x = -Math.PI / 2; this.shadow.position.y = 0.01; this.scene.add(this.shadow);

    const N = 40; this.sparkVel = new Float32Array(N * 3); this.sparkLife = new Float32Array(N);
    this.sparks = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3)), new THREE.PointsMaterial({ color: 0xffe08a, size: 0.08, transparent: true, opacity: 0.95 }));
    this.sparks.frustumCulled = false; this.scene.add(this.sparks);
  }

  draw(s: Snapshot, label: string, fps: number): void {
    const P = s.player, pz = P.z;
    // floor scrolls under a plane that follows the player (texture offset keeps it world-fixed)
    const cz = pz + FLOOR_LEN / 2 - 12;
    this.ground.position.z = cz; this.floorTex.offset.y = -((cz / FLOOR_TILE) % 1);
    // side segments
    const base = Math.floor(pz / SEG) - 2;
    for (let i = 0; i < this.segments.length; i++) {
      const g = this.segments[i], k = i % SEGS, idx = base + k;
      g.position.z = idx * SEG;
      const h = hash(idx * 7 + (g.userData.side + 1) * 3), h2 = hash(idx * 13 + g.userData.side);
      const variant = h < 0.45 ? "wall" : h < 0.7 ? "broken" : "gap";
      for (const c of g.children) {
        if (c.name === "wall") c.visible = variant === "wall";
        else if (c.name === "broken") c.visible = variant === "broken";
        else if (c.name === "pillar") c.visible = idx % 4 === 0;
        else if (c.name.startsWith("tree")) c.visible = h2 > 0.25 || c.name === "tree1";
        else if (c.name === "bush") c.visible = variant === "gap";
      }
    }
    // obstacles & coins
    const keep = new Set<THREE.Object3D>();
    for (const o of s.obstacles) {
      let m = this.obstacleMeshes.get(o);
      if (!m) { m = (o.kind === ObstacleKind.Low ? this.geo.log : o.kind === ObstacleKind.High ? this.geo.beam : this.geo.wall).clone(); this.obstacleMeshes.set(o, m); this.scene.add(m); }
      m.position.set(X(LANE_X[o.lane]), 0, o.z); m.visible = true; keep.add(m);
      if (s.crashedInto === o) m.traverse(c => { if ((c as THREE.Mesh).isMesh) { const mat = ((c as THREE.Mesh).material as THREE.MeshStandardMaterial).clone(); mat.emissive = new THREE.Color(0x802020); (c as THREE.Mesh).material = mat; } });
    }
    for (const c of s.coins) {
      if (c.taken) continue;
      let m = this.coinMeshes.get(c);
      if (!m) { m = this.geo.coin.clone(); this.coinMeshes.set(c, m); this.scene.add(m); }
      m.position.set(X(LANE_X[c.lane]), 0.55 + Math.sin(s.time * 4 + c.z) * 0.05, c.z); m.rotation.y = s.time * 4; m.visible = true; keep.add(m);
    }
    for (const m of this.live) if (!keep.has(m)) { this.scene.remove(m); }
    this.live = keep;

    // player
    this.explorer.pose({ x: X(P.x), y: P.y, z: pz, t: s.time, airborne: P.airborne, sliding: P.sliding, lean: P.leanDir * (1 - Math.abs(P.laneT * 2 - 1)), dead: !s.alive, deadT: this.deadT, speedFrac: s.speed / MAX_SPEED });
    this.shadow.position.set(X(P.x), 0.01, pz); this.shadow.scale.setScalar(1 - P.y * 0.4);
    if (!s.alive) { if (this.deadT === 0) this.shake = 14; this.deadT += 1 / 60; } else this.deadT = 0;
    if (s.coinsCollected > this.lastCoins) this.burst(X(P.x), P.y + 0.6, pz); this.lastCoins = s.coinsCollected;
    this.stepSparks();

    // chase camera
    const cx = X(P.x) * 0.4, look = new THREE.Vector3(X(P.x) * 0.7, 0.9, pz + 9);
    this.camera.position.set(cx, 2.5 + P.y * 0.15, pz - 4.8);
    if (this.shake > 0) { this.camera.position.x += (Math.random() - 0.5) * 0.05 * this.shake; this.camera.position.y += (Math.random() - 0.5) * 0.05 * this.shake; this.shake--; }
    this.camera.lookAt(look);
    this.renderer.render(this.scene, this.camera);

    // HUD (DOM)
    this.hud.score.textContent = `${Math.floor(s.score).toLocaleString()} pts`;
    this.hud.line.textContent = `${s.distance.toFixed(0)} m · ${s.coinsCollected} coins · ${s.speed.toFixed(1)} m/s`;
    this.hud.driver.textContent = `driver: ${label} · ${fps.toFixed(0)} fps`;
    this.hud.overlay.hidden = s.alive;
    if (!s.alive) this.hud.sub.textContent = `${s.distance.toFixed(0)} m · ${Math.floor(s.score)} pts · press R`;
  }

  private burst(x: number, y: number, z: number) {
    const pos = this.sparks.geometry.getAttribute("position") as THREE.BufferAttribute;
    let placed = 0;
    for (let i = 0; i < this.sparkLife.length && placed < 12; i++) {
      if (this.sparkLife[i] > 0) continue;
      pos.setXYZ(i, x, y, z); const a = Math.random() * Math.PI * 2;
      this.sparkVel[i * 3] = Math.cos(a) * 0.05; this.sparkVel[i * 3 + 1] = 0.06 + Math.random() * 0.05; this.sparkVel[i * 3 + 2] = Math.sin(a) * 0.05 + 0.1;
      this.sparkLife[i] = 1; placed++;
    }
  }
  private stepSparks() {
    const pos = this.sparks.geometry.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < this.sparkLife.length; i++) {
      if (this.sparkLife[i] <= 0) { pos.setY(i, -10); continue; }
      this.sparkLife[i] -= 0.05; this.sparkVel[i * 3 + 1] -= 0.004;
      pos.setXYZ(i, pos.getX(i) + this.sparkVel[i * 3], pos.getY(i) + this.sparkVel[i * 3 + 1], pos.getZ(i) + this.sparkVel[i * 3 + 2]);
    }
    pos.needsUpdate = true;
  }
}
