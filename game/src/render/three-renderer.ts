import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { type Snapshot, type Obstacle, type Coin, ObstacleKind, LANE_X, OBSTACLE_HALF_WIDTH, MAX_SPEED } from "../engine";
import { Assets } from "./assets";
import { logEndTexture, leafSpriteTexture, sparkTexture } from "./textures";
import { Explorer, type Pose } from "./explorer";
import { Fly } from "./fly";

const SEG = 6;             // length of one recycled wall/jungle segment
const SEGS = 22;           // segments kept alive per side (covers ~130 units ahead)
const FLOOR_LEN = 220, PATH_TILE = 3.2, GROUND_TILE = 6;
const PATH_HALF = 1.9;     // half-width of the stone path
const WALL_X = 2.9;
const FOG = 0xbdbaa8;      // sampled from the mist at the bottom of the generated sky painting
const SKY_DIST = 110, SKY_H = 140, SKY_W = SKY_H * 16 / 9;

/** The chase camera looks down +z, which puts world +x on screen-left; mirror x so lane 0 (x=-1) is on the left. */
const X = (x: number) => -x;
const hash = (n: number) => { let x = (n | 0) * 2654435761; x ^= x >>> 15; x = Math.imul(x, 2246822519); x ^= x >>> 13; return (x >>> 0) / 4294967296; };

export interface GameRenderer { draw(s: Snapshot, label: string, fps: number): void }

/** Vignette + saturation grade, applied in linear space before the output pass. */
const GradeShader = {
  uniforms: { tDiffuse: { value: null }, vignette: { value: 0.4 }, saturation: { value: 1.12 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D tDiffuse; uniform float vignette; uniform float saturation; varying vec2 vUv;
    void main(){ vec4 c = texture2D(tDiffuse, vUv); float l = dot(c.rgb, vec3(0.299, 0.587, 0.114)); c.rgb = mix(vec3(l), c.rgb, saturation);
      vec2 d = vUv - 0.5; float v = smoothstep(0.9, 0.2, dot(d, d) * 2.0); c.rgb *= mix(1.0 - vignette, 1.0, v); gl_FragColor = c; }`,
};

/**
 * three.js chase-camera renderer: mossy temple causeway through the jungle, carved ruin walls,
 * idol blocks, a running explorer (human) or fly (agents), real-time shadows, painted sky, bloom + grade.
 * Textures are AI-generated PBR tiles (public/textures). Purely cosmetic; the engine never depends on it.
 */
export class ThreeRenderer implements GameRenderer {
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private assets: Assets;
  private built = false;
  private pathTex: THREE.Texture[] = []; private groundTex: THREE.Texture[] = [];
  private ground = new THREE.Group();
  private sunRig = new THREE.Group();
  private sky!: THREE.Mesh;
  private segments: THREE.Group[] = [];
  private explorer = new Explorer(); private fly = new Fly();   // human drives the explorer, agents drive the fly
  private shadowBlob: THREE.Mesh;
  private obstacleMeshes = new WeakMap<Obstacle, THREE.Group>();
  private coinMeshes = new WeakMap<Coin, THREE.Mesh>();
  private live = new Set<THREE.Object3D>();
  private geo!: { log: THREE.Group; beam: THREE.Group; wall: THREE.Group; coin: THREE.Mesh };
  private sparks!: THREE.Points; private sparkVel = new Float32Array(0); private sparkLife = new Float32Array(0);
  private leaves!: THREE.Points; private leafVel = new Float32Array(0);
  private lastCoins = 0; private shake = 0; private deadT = 0;
  private hud = { score: document.getElementById("hud-score")!, line: document.getElementById("hud-line")!, driver: document.getElementById("hud-driver")!, overlay: document.getElementById("overlay")!, sub: document.getElementById("overlay-sub")! };

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(canvas.width, canvas.height, false);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.18;
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.camera = new THREE.PerspectiveCamera(58, canvas.width / canvas.height, 0.1, 180);

    this.composer = new EffectComposer(this.renderer);
    this.composer.setPixelRatio(dpr); this.composer.setSize(canvas.width, canvas.height);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new UnrealBloomPass(new THREE.Vector2(canvas.width, canvas.height), 0.22, 0.5, 0.92));
    this.composer.addPass(new ShaderPass(GradeShader));
    this.composer.addPass(new OutputPass());

    this.scene.background = new THREE.Color(FOG); this.scene.fog = new THREE.Fog(FOG, 22, 105);
    this.scene.add(new THREE.HemisphereLight(0xd8e6d2, 0x4a5630, 0.95));
    const sun = new THREE.DirectionalLight(0xffe4b4, 2.4); sun.position.set(-9, 17, -7); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
    const sc = sun.shadow.camera; sc.left = -14; sc.right = 14; sc.top = 34; sc.bottom = -10; sc.near = 1; sc.far = 70;
    sun.target.position.set(0, 0, 10); this.sunRig.add(sun, sun.target); this.scene.add(this.sunRig);

    this.shadowBlob = new THREE.Mesh(new THREE.CircleGeometry(0.42, 16), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.22 }));
    this.shadowBlob.rotation.x = -Math.PI / 2; this.shadowBlob.position.y = 0.012; this.scene.add(this.shadowBlob);
    this.scene.add(this.explorer.group, this.fly.group, this.ground);

    this.assets = new Assets(this.renderer.capabilities.getMaxAnisotropy(), () => this.build(), url => console.warn("texture failed to load", url));
  }

  // ---------------------------------------------------------------- scene construction (after textures load)
  private build() {
    const A = this.assets;
    // sky painting on a distant billboard that follows the camera; unaffected by fog
    this.sky = new THREE.Mesh(new THREE.PlaneGeometry(SKY_W, SKY_H), new THREE.MeshBasicMaterial({ map: A.tex("sky"), fog: false, depthWrite: false }));
    this.sky.rotation.y = Math.PI; this.scene.add(this.sky);

    // ground group follows the player along z; texture offsets keep tiles world-fixed
    const pathMat = A.pbr("path", PATH_HALF * 2 / PATH_TILE, FLOOR_LEN / PATH_TILE, { normalScale: new THREE.Vector2(1.1, 1.1) });
    this.pathTex = [pathMat.map!, pathMat.normalMap!, pathMat.roughnessMap!];
    const path = new THREE.Mesh(new THREE.PlaneGeometry(PATH_HALF * 2, FLOOR_LEN), pathMat); path.rotation.x = -Math.PI / 2; path.receiveShadow = true; this.ground.add(path);
    const groundMat = A.pbr("ground", 80 / GROUND_TILE, FLOOR_LEN / GROUND_TILE, { color: 0xb9b3a4 });
    this.groundTex = [groundMat.map!, groundMat.normalMap!, groundMat.roughnessMap!];
    const earth = new THREE.Mesh(new THREE.PlaneGeometry(80, FLOOR_LEN), groundMat); earth.rotation.x = -Math.PI / 2; earth.position.y = -0.05; earth.receiveShadow = true; this.ground.add(earth);
    // carved stone curbs edging the path (world-fixed texture through the same offset trick)
    const curbMat = A.pbr("floor", 0.4, FLOOR_LEN / 1.6, { color: 0xcfc7b4 });
    this.pathTex.push(curbMat.map!, curbMat.normalMap!, curbMat.roughnessMap!);
    for (const x of [-PATH_HALF - 0.16, PATH_HALF + 0.16]) { const c = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.16, FLOOR_LEN), curbMat); c.position.set(x, 0.06, 0); c.castShadow = c.receiveShadow = true; this.ground.add(c); }

    // shared materials
    const wallMat = A.pbr("wall", 2.4, 1), wallEndMat = A.pbr("wall", 0.4, 1);
    const pillarMat = A.pbr("wall", 1.4, 1.5), capMat = A.pbr("floor", 1, 1, { color: 0xd8d0bc });
    const barkMat = A.pbr("bark", 1, 2.2), leafMat = A.pbr("foliage", 1, 1, { color: 0xd4dccc });
    const frondMat = A.pbr("foliage", 0.5, 1.2, { side: THREE.DoubleSide, color: 0xc8d8b8, emissive: 0x2a4a1c, emissiveIntensity: 0.55 });
    const mossMat = A.pbr("foliage", 0.6, 0.6, { color: 0x9fbb78 });
    const idolMat = A.pbr("idol", 1, 1, { color: 0xe0d6c0 });
    const rubbleMat = A.pbr("floor", 0.5, 0.5, { color: 0xb8ad98 });
    const vineMat = new THREE.MeshStandardMaterial({ color: 0x2c3d1c, roughness: 1 });

    // organic canopy blob: jittered icosahedron
    const blob = (r: number, seed: number) => {
      const g = new THREE.IcosahedronGeometry(r, 2), p = g.attributes.position as THREE.BufferAttribute;
      for (let i = 0; i < p.count; i++) { const k = 0.82 + hash(seed + i) * 0.36; p.setXYZ(i, p.getX(i) * k, p.getY(i) * (k * 0.85), p.getZ(i) * k); }
      g.computeVertexNormals(); return g;
    };
    const blobs = [blob(1.7, 1), blob(1.3, 2), blob(1.0, 3), blob(0.75, 4), blob(0.5, 5)];
    const shadowed = (m: THREE.Mesh) => { m.castShadow = true; m.receiveShadow = true; return m; };

    // recycled side segments: temple wall variants, pillars with idol heads, trees, bushes, vines
    const pillarGeo = new THREE.CylinderGeometry(0.34, 0.4, 3.4, 12);
    for (let i = 0; i < SEGS * 2; i++) {
      const side = i < SEGS ? -1 : 1;
      const g = new THREE.Group(); g.userData.side = side;
      const wallMats = [wallEndMat, wallEndMat, capMat, wallMat, wallMat, wallMat];
      const wall = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.9, 2.4, SEG), wallMats)); wall.position.set(side * WALL_X, 1.2, SEG / 2); wall.name = "wall"; g.add(wall);
      const broken = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.9, 1.15, SEG), wallMats)); broken.position.set(side * WALL_X, 0.575, SEG / 2); broken.name = "broken"; g.add(broken);
      for (let k = 0; k < 3; k++) {                    // rubble in front of the broken wall
        const r = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.22 + hash(i * 9 + k) * 0.22, 0.14 + hash(i * 7 + k) * 0.16, 0.22 + hash(i * 5 + k) * 0.3), rubbleMat));
        r.position.set(side * (WALL_X - 0.55 - hash(i + k) * 0.3), 0.1, (k + 0.6) * SEG / 4); r.rotation.set(hash(k) * 0.3, hash(i + k * 3) * 1.5, hash(i * k) * 0.3); r.name = "broken"; g.add(r);
      }
      const pillar = shadowed(new THREE.Mesh(pillarGeo, pillarMat)); pillar.position.set(side * (WALL_X - 0.15), 1.7, 0.7); pillar.name = "pillar"; g.add(pillar);
      const cap = shadowed(new THREE.Mesh(new THREE.BoxGeometry(1, 0.3, 1), capMat)); cap.position.set(side * (WALL_X - 0.15), 3.55, 0.7); cap.name = "pillar"; g.add(cap);
      const head = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.8, 0.7), [wallEndMat, wallEndMat, mossMat, wallEndMat, wallEndMat, idolMat])); head.position.set(side * (WALL_X - 0.15), 4.1, 0.7); head.name = "idolhead"; g.add(head);
      for (let v = 0; v < 3; v++) {                    // vines hanging over the wall
        const vine = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.012, 1 + hash(i * 3 + v) * 1.2, 5), vineMat);
        vine.position.set(side * (WALL_X - 0.5), 2.4 - vine.geometry.parameters.height / 2, (v + 0.5) * SEG / 3 + hash(v + i) * 0.8); vine.rotation.z = side * 0.15; vine.name = "vine"; g.add(vine);
        const tuft = new THREE.Mesh(blobs[4], leafMat); tuft.scale.setScalar(0.3); tuft.position.set(vine.position.x, vine.position.y - vine.geometry.parameters.height / 2, vine.position.z); tuft.name = "vine"; g.add(tuft);
      }
      for (let t = 0; t < 3; t++) {
        const tree = new THREE.Group(); tree.name = `tree${t}`;
        const h = hash(i * 31 + t);
        if (t === 1) {                                   // palm
          const trunk = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.2, 6.5, 8), barkMat)); trunk.position.y = 3.2; trunk.rotation.z = side * 0.14; tree.add(trunk);
          for (let k = 0; k < 9; k++) {
            const f = new THREE.Mesh(new THREE.PlaneGeometry(0.55, 2.8), frondMat); f.castShadow = true; f.position.set(side * 0.42, 6.35, 0);
            f.rotation.set(-0.55 - hash(k + i) * 0.3, k * Math.PI * 2 / 9, 0, "YXZ"); f.translateY(1.3); tree.add(f);
          }
        } else {                                         // broadleaf jungle tree
          const trunk = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.42, 3.8 + h, 9), barkMat)); trunk.position.y = 1.9 + h / 2; tree.add(trunk);
          const root = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.7, 0.5, 9), barkMat)); root.position.y = 0.2; tree.add(root);
          const crown = [[0, 4.6, 0, 0], [1.2, 4.0, 0.5, 1], [-1.1, 4.2, -0.6, 1], [0.3, 5.5, 0.7, 2], [-0.5, 5.1, 1.1, 2], [0.9, 5.4, -0.9, 3]] as const;
          for (const [dx, dy, dz, b] of crown) { const c = shadowed(new THREE.Mesh(blobs[b], leafMat)); c.position.set(dx, dy + h * 0.5, dz); c.rotation.y = hash(b + i) * 3; tree.add(c); }
        }
        tree.position.set(side * (WALL_X + 1.5 + t * 1.9), 0, (t + 0.5) * SEG / 3); tree.rotation.y = h * 6; g.add(tree);
      }
      for (let t = 0; t < 2; t++) {                    // backdrop trees: bigger, further out, always visible
        const big = new THREE.Group(); big.name = "bigtree"; const h = hash(i * 17 + t * 5);
        const trunk = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.6, 5.5, 8), barkMat)); trunk.position.y = 2.7; big.add(trunk);
        for (const [dx, dy, dz, b] of [[0, 6.4, 0, 0], [1.6, 5.6, 0.8, 0], [-1.5, 5.9, -0.7, 1], [0.4, 7.6, 1.0, 1], [-0.6, 7.2, -1.3, 2]] as const) { const c = shadowed(new THREE.Mesh(blobs[b], leafMat)); c.position.set(dx, dy, dz); c.rotation.y = h * 5 + b; big.add(c); }
        big.position.set(side * (WALL_X + 6.5 + t * 3.2 + h * 2), 0, (t + 0.4) * SEG / 2); big.scale.setScalar(1.25 + h * 0.4); big.rotation.y = h * 6; g.add(big);
      }
      const bush = shadowed(new THREE.Mesh(blobs[3], leafMat)); bush.position.set(side * (WALL_X + 0.6), 0.25, SEG * 0.7); bush.name = "bush"; g.add(bush);
      const fern = shadowed(new THREE.Mesh(blobs[4], mossMat)); fern.position.set(side * (PATH_HALF + 0.75), 0.05, SEG * 0.3); fern.scale.set(1.2, 0.5, 1.2); fern.name = "fern"; g.add(fern);
      this.segments.push(g); this.scene.add(g);
    }

    // obstacle prototypes
    const log = new THREE.Group();
    const trunk = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, OBSTACLE_HALF_WIDTH * 2 + 0.4, 14, 1, true), A.pbr("bark", 2, 1, { color: 0xf0e0c8 })));
    trunk.rotation.z = Math.PI / 2; trunk.position.y = 0.36; log.add(trunk);
    const endMat = new THREE.MeshStandardMaterial({ map: logEndTexture(), roughness: 0.9 });
    for (const sx of [-1, 1]) {
      const cap = new THREE.Mesh(new THREE.CircleGeometry(0.36, 14), endMat); cap.position.set(sx * (OBSTACLE_HALF_WIDTH + 0.2), 0.36, 0); cap.rotation.y = sx * Math.PI / 2; log.add(cap);
      const stub = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.09, 0.5, 6), barkMat)); stub.position.set(sx * 0.3, 0.66, 0.05 * sx); stub.rotation.z = sx * 0.6; log.add(stub);
    }
    const mossPatch = new THREE.Mesh(blobs[4], mossMat); mossPatch.scale.set(0.8, 0.35, 0.7); mossPatch.position.set(0.1, 0.62, 0); log.add(mossPatch);

    const beam = new THREE.Group();
    const paleWall = A.pbr("wall", 0.4, 1, { color: 0xe8e0d0 });
    const postMats = [paleWall, paleWall, capMat, capMat, paleWall, paleWall];
    for (const sx of [-1, 1]) { const post = shadowed(new THREE.Mesh(new THREE.BoxGeometry(0.24, 1.9, 0.24), postMats)); post.position.set(sx * (OBSTACLE_HALF_WIDTH + 0.06), 0.95, 0); beam.add(post); }
    const lintel = shadowed(new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2 + 0.4, 0.36, 0.4), [paleWall, paleWall, capMat, capMat, A.pbr("wall", 0.55, 0.22, { color: 0xe8e0d0 }), A.pbr("wall", 0.55, 0.22, { color: 0xe8e0d0 })])); lintel.position.y = 1.02; beam.add(lintel);
    const spikeMat = new THREE.MeshStandardMaterial({ color: 0xc8c0b0, roughness: 0.45, metalness: 0.2 });
    for (let i = -2; i <= 2; i++) { const s = shadowed(new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.28, 5), spikeMat)); s.position.set(i * 0.21, 0.7, 0); s.rotation.x = Math.PI; beam.add(s); }
    for (let i = 0; i < 4; i++) { const v = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.5 + hash(i) * 0.4, 4), vineMat); v.position.set(-0.4 + i * 0.27, 1.45, 0.15); beam.add(v); }
    const lintelMoss = new THREE.Mesh(blobs[4], mossMat); lintelMoss.scale.set(1.2, 0.3, 0.6); lintelMoss.position.set(0.2, 1.25, 0); beam.add(lintelMoss);

    const wall = new THREE.Group();
    // box face order: +x, -x, +y, -y, +z, -z; the camera sees the -z face
    const block = shadowed(new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2, 1.5, 0.8), [wallEndMat, wallEndMat, capMat, capMat, wallEndMat, idolMat])); block.position.set(0, 0.75, 0.4); wall.add(block);
    const crownMoss = new THREE.Mesh(blobs[3], mossMat); crownMoss.scale.set(0.7, 0.3, 0.6); crownMoss.position.set(0.1, 1.55, 0.35); wall.add(crownMoss);
    const plinth = shadowed(new THREE.Mesh(new THREE.BoxGeometry(OBSTACLE_HALF_WIDTH * 2 + 0.2, 0.12, 1.0), capMat)); plinth.position.set(0, 0.06, 0.4); wall.add(plinth);

    const coinFace = new THREE.MeshStandardMaterial({ map: A.tex("coin"), alphaTest: 0.5, metalness: 0.6, roughness: 0.35, emissive: 0x1a1000 });
    const coinRim = new THREE.MeshStandardMaterial({ color: 0xffd25a, metalness: 0.85, roughness: 0.3, emissive: 0x1a1000 });
    const coin = new THREE.Mesh(new THREE.CylinderGeometry(0.27, 0.27, 0.06, 28), [coinRim, coinFace, coinFace]);
    coin.rotation.x = Math.PI / 2; coin.castShadow = true;
    this.geo = { log, beam, wall, coin };

    // particles: coin sparks and drifting jungle leaves
    const N = 48; this.sparkVel = new Float32Array(N * 3); this.sparkLife = new Float32Array(N);
    this.sparks = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(N * 3), 3)),
      new THREE.PointsMaterial({ map: sparkTexture(), color: 0xffe6a0, size: 0.13, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.sparks.frustumCulled = false; this.scene.add(this.sparks);
    const L = 140; this.leafVel = new Float32Array(L * 3);
    const lp = new Float32Array(L * 3);
    for (let i = 0; i < L; i++) { lp[i * 3] = (hash(i) - 0.5) * 12; lp[i * 3 + 1] = hash(i + 99) * 7; lp[i * 3 + 2] = hash(i + 7) * 60; this.leafVel[i * 3] = (hash(i + 3) - 0.5) * 0.01; this.leafVel[i * 3 + 1] = -0.006 - hash(i + 5) * 0.008; this.leafVel[i * 3 + 2] = (hash(i + 11) - 0.5) * 0.01; }
    this.leaves = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(lp, 3)),
      new THREE.PointsMaterial({ map: leafSpriteTexture(), size: 0.16, transparent: true, alphaTest: 0.3, depthWrite: false, sizeAttenuation: true }));
    this.leaves.frustumCulled = false; this.scene.add(this.leaves);
    this.built = true;
  }

  // ---------------------------------------------------------------- per frame
  draw(s: Snapshot, label: string, fps: number): void {
    const P = s.player, pz = P.z;
    if (this.built) {
      // ground planes follow the player; texture offsets keep the tiles world-fixed
      const cz = pz + FLOOR_LEN / 2 - 14;
      this.ground.position.z = cz;
      for (const t of this.pathTex) t.offset.y = -((cz / PATH_TILE) % 1);
      for (const t of this.groundTex) t.offset.y = -((cz / GROUND_TILE) % 1);
      // side segments
      const base = Math.floor(pz / SEG) - 2;
      for (let i = 0; i < this.segments.length; i++) {
        const g = this.segments[i], k = i % SEGS, idx = base + k;
        g.position.z = idx * SEG;
        const h = hash(idx * 7 + (g.userData.side + 1) * 3), h2 = hash(idx * 13 + g.userData.side);
        const variant = h < 0.5 ? "wall" : h < 0.75 ? "broken" : "gap";
        for (const c of g.children) {
          if (c.name === "wall") c.visible = variant === "wall";
          else if (c.name === "broken") c.visible = variant === "broken";
          else if (c.name === "pillar") c.visible = idx % 3 === 0;
          else if (c.name === "idolhead") c.visible = idx % 6 === 0;
          else if (c.name === "vine") c.visible = variant === "wall" && h2 > 0.4;
          else if (c.name.startsWith("tree")) c.visible = h2 > 0.2 || c.name === "tree1";
          else if (c.name === "bush") c.visible = variant !== "wall";
          else if (c.name === "fern") c.visible = h2 < 0.6;
        }
      }
      // obstacles & coins
      const keep = new Set<THREE.Object3D>();
      for (const o of s.obstacles) {
        let m = this.obstacleMeshes.get(o);
        if (!m) { m = (o.kind === ObstacleKind.Low ? this.geo.log : o.kind === ObstacleKind.High ? this.geo.beam : this.geo.wall).clone(); this.obstacleMeshes.set(o, m); this.scene.add(m); }
        m.position.set(X(LANE_X[o.lane]), 0, o.z); m.visible = true; keep.add(m);
        if (s.crashedInto === o && !m.userData.flashed) {
          m.userData.flashed = true;
          const tint = (mm: THREE.Material) => { const cl = (mm as THREE.MeshStandardMaterial).clone(); cl.emissive = new THREE.Color(0x7a1a10); return cl; };
          m.traverse(c => { const mesh = c as THREE.Mesh; if (!mesh.isMesh) return; mesh.material = Array.isArray(mesh.material) ? mesh.material.map(tint) : tint(mesh.material); });
        }
      }
      for (const c of s.coins) {
        if (c.taken) continue;
        let m = this.coinMeshes.get(c);
        if (!m) { m = this.geo.coin.clone(); this.coinMeshes.set(c, m); this.scene.add(m); }
        m.position.set(X(LANE_X[c.lane]), 0.62 + Math.sin(s.time * 4 + c.z) * 0.06, c.z); m.rotation.y = s.time * 3.5 + c.z; m.visible = true; keep.add(m);
      }
      for (const m of this.live) if (!keep.has(m)) this.scene.remove(m);
      this.live = keep;

      // player, blob shadow, sun rig, sky
      const pose: Pose = { x: X(P.x), y: P.y, z: pz, t: s.time, airborne: P.airborne, sliding: P.sliding, lean: P.leanDir * (1 - Math.abs(P.laneT * 2 - 1)), dead: !s.alive, deadT: this.deadT, speedFrac: s.speed / MAX_SPEED };
      const human = label === "human";
      this.explorer.group.visible = human; this.fly.group.visible = !human;
      (human ? this.explorer : this.fly).pose(pose);
      this.shadowBlob.position.set(X(P.x), 0.012, pz); this.shadowBlob.scale.setScalar(1 - P.y * 0.4);
      this.sunRig.position.z = pz;
      if (!s.alive) { if (this.deadT === 0) this.shake = 16; this.deadT += 1 / 60; } else this.deadT = 0;
      if (s.coinsCollected > this.lastCoins) this.burst(X(P.x), P.y + 0.7, pz); this.lastCoins = s.coinsCollected;
      this.stepSparks(); this.stepLeaves(pz);
    }

    // chase camera: low and close behind the runner, widening slightly with speed
    const speedFrac = s.speed / MAX_SPEED;
    const cx = X(P.x) * 0.45, look = new THREE.Vector3(X(P.x) * 0.75, 1.0, pz + 8.5);
    this.camera.position.set(cx, 2.35 + P.y * 0.12, pz - 4.6);
    if (this.shake > 0) { this.camera.position.x += (Math.random() - 0.5) * 0.05 * this.shake; this.camera.position.y += (Math.random() - 0.5) * 0.05 * this.shake; this.shake--; }
    this.camera.lookAt(look);
    const fov = 58 + speedFrac * 7; if (Math.abs(this.camera.fov - fov) > 0.05) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
    if (this.built) { this.sky.position.set(this.camera.position.x * 0.3, 5 + SKY_H * 0.05, pz + SKY_DIST); this.composer.render(); }

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
    for (let i = 0; i < this.sparkLife.length && placed < 10; i++) {
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
  /** Leaves drift down through a volume that travels with the player; recycled when they land or fall behind. */
  private stepLeaves(pz: number) {
    const pos = this.leaves.geometry.getAttribute("position") as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      let x = pos.getX(i) + this.leafVel[i * 3] + Math.sin(i + pos.getY(i) * 2) * 0.004, y = pos.getY(i) + this.leafVel[i * 3 + 1], z = pos.getZ(i) + this.leafVel[i * 3 + 2];
      if (y < 0.05 || z < pz - 6) { x = (Math.random() - 0.5) * 12; y = 5 + Math.random() * 3; z = pz + 4 + Math.random() * 50; }
      pos.setXYZ(i, x, y, z);
    }
    pos.needsUpdate = true;
  }
}
