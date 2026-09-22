import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import type { Pose } from "./explorer";

/**
 * The fly modelled in Blender (scripts/blender_fly.py → public/models/fly.glb). The GLB carries a
 * named node hierarchy; this class animates those nodes with the same pose contract as the
 * primitive rigs: legs "leg{L|R}{0..2}" pivot at the hip with a "_tibia" child at the knee,
 * wings "wingL"/"wingR" pivot at the root, "head" turns, eye materials dim on death.
 */
export class FlyModel {
  group = new THREE.Group();
  private body = new THREE.Group();
  private wingL?: THREE.Object3D; private wingR?: THREE.Object3D; private head?: THREE.Object3D;
  private legs: { hip: THREE.Object3D; knee: THREE.Object3D; side: number; row: number }[] = [];
  private eyeMats: THREE.MeshStandardMaterial[] = [];

  constructor() {
    this.body.scale.setScalar(1.15);
    this.group.add(this.body);
    new GLTFLoader().load("models/fly.glb", gltf => {
      const s = gltf.scene;
      s.traverse(o => {
        const m = o as THREE.Mesh; if (!m.isMesh) return;
        const mat = m.material as THREE.MeshStandardMaterial;
        if (mat.name === "membrane") {
          m.material = new THREE.MeshPhysicalMaterial({ color: mat.color, transparent: true, opacity: 0.5, roughness: 0.15, metalness: 0, iridescence: 0.6, iridescenceIOR: 1.3, side: THREE.DoubleSide, depthWrite: false });
          m.renderOrder = 2;
        } else {
          m.castShadow = true;
          if (mat.name === "eye") this.eyeMats.push(mat);
        }
      });
      this.wingL = s.getObjectByName("wingL"); this.wingR = s.getObjectByName("wingR"); this.head = s.getObjectByName("head");
      for (const [side, S] of [[-1, "L"], [1, "R"]] as const) for (let row = 0; row < 3; row++) {
        const hip = s.getObjectByName(`leg${S}${row}`), knee = s.getObjectByName(`leg${S}${row}_tibia`);
        if (hip && knee) this.legs.push({ hip, knee, side, row });
      }
      this.body.add(s);
    }, undefined, err => console.warn("fly model failed to load", err));
  }

  pose(p: Pose): void {
    this.group.position.set(p.x, p.y, p.z);
    this.group.rotation.set(0, 0, 0); this.body.position.set(0, 0, 0); this.body.rotation.set(0, 0, 0);
    this.head?.rotation.set(0, 0, 0);
    const buzz = p.t * 75, cyc = p.t * (22 + 16 * p.speedFrac);
    // rest sweeps the wing tips backward over the abdomen; f is the flap
    const wing = (rest: number, amp: number) => { const f = Math.sin(buzz) * amp; this.wingL?.rotation.set(0, -rest, f); this.wingR?.rotation.set(0, rest, -f); };
    const leg = (L: typeof this.legs[number], swing: number, lift: number, splay = 0) => { L.hip.rotation.set(swing, 0, L.side * splay); L.knee.rotation.z = L.side * -lift; };
    const eyes = (hex: number) => { for (const m of this.eyeMats) m.color.setHex(hex); };
    if (p.dead) {
      const k = Math.min(1, p.deadT * 3);
      this.body.rotation.z = k * Math.PI; this.body.position.y = -k * 0.15; this.body.rotation.x = k * 0.3;
      for (const L of this.legs) leg(L, Math.sin(L.row + L.side) * 0.3, 1.6 * k, -0.6 * k);   // legs curl up
      wing(0.9, 0); eyes(0x6a3a3a);
      return;
    }
    eyes(0xb31a1f);
    if (p.sliding) {
      // flatten against the path: body drops, legs splay out sideways, wings pressed flat
      this.body.position.y = -0.3; this.body.rotation.x = 0.15;
      for (const L of this.legs) leg(L, (L.row - 1) * 0.6, -0.5, 0.9);
      wing(0.7, 0.05); if (this.head) this.head.rotation.x = -0.3;
    } else if (p.airborne) {
      // wings out and buzzing, legs tucked back, nose up
      this.body.rotation.x = -0.25; this.body.position.y = 0.1;
      for (const L of this.legs) leg(L, -0.9, 0.9, -0.3);
      wing(-0.1, 0.55);
    } else {
      // tripod gait: (L1, R2, L3) swing together, (R1, L2, R3) opposite
      for (const L of this.legs) {
        const ph = ((L.row + (L.side > 0 ? 1 : 0)) % 2) ? 0 : Math.PI, s = Math.sin(cyc + ph);
        leg(L, s * 0.55 + (L.row - 1) * 0.35, Math.max(0, Math.cos(cyc + ph)) * 0.6, 0);
      }
      wing(0.75, 0.12 + p.speedFrac * 0.1);
      this.body.position.y = Math.abs(Math.sin(cyc)) * 0.02; this.body.rotation.x = 0.05 + p.speedFrac * 0.1;
      if (this.head) this.head.rotation.y = Math.sin(p.t * 3) * 0.15;   // looking around
    }
    // lean is +1 when moving toward screen-right (world -x after the renderer's mirror)
    this.body.rotation.z += p.lean * 0.5; this.body.rotation.y += -p.lean * 0.45;
  }
}
