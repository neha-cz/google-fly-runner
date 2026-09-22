import * as THREE from "three";
import { wingTexture, abdomenTexture } from "./textures";
import type { Pose } from "./explorer";

/**
 * The runner as a housefly built from primitives: banded abdomen, thorax, head with red compound
 * eyes and antennae, six three-segment legs in a tripod scurry gait, translucent veined wings
 * that buzz. Faces +z (direction of travel). Same pose contract as the explorer; purely cosmetic.
 */
export class Fly {
  group = new THREE.Group();
  private body = new THREE.Group();
  private wingL: THREE.Group; private wingR: THREE.Group;
  private legs: { hip: THREE.Group; knee: THREE.Group; side: number; row: number }[] = [];
  private head: THREE.Group;
  private eyeMat: THREE.MeshStandardMaterial;

  constructor() {
    const chitin = new THREE.MeshStandardMaterial({ color: 0x3b2a18, roughness: 0.45, metalness: 0.25 });
    const thoraxMat = new THREE.MeshStandardMaterial({ color: 0x5a4126, roughness: 0.5, metalness: 0.2 });
    const abdomenMat = new THREE.MeshStandardMaterial({ map: abdomenTexture(), roughness: 0.4, metalness: 0.25 });
    const legMat = new THREE.MeshStandardMaterial({ color: 0x2a1c0e, roughness: 0.7 });
    this.eyeMat = new THREE.MeshStandardMaterial({ color: 0xc0262e, roughness: 0.25, metalness: 0.1, emissive: 0x3a0608 });
    const wingMat = new THREE.MeshPhysicalMaterial({ map: wingTexture(), transparent: true, opacity: 0.85, side: THREE.DoubleSide, roughness: 0.15, metalness: 0, iridescence: 0.7, iridescenceIOR: 1.3, depthWrite: false });
    const add = (parent: THREE.Object3D, geo: THREE.BufferGeometry, m: THREE.Material, x = 0, y = 0, z = 0) => {
      const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z); mesh.castShadow = true; parent.add(mesh); return mesh;
    };

    // thorax (front/top), abdomen trailing behind with the stripes ringing the z-axis
    const thorax = add(this.body, new THREE.SphereGeometry(0.24, 18, 14), thoraxMat, 0, 0.5, 0.08); thorax.scale.set(1, 0.85, 1.15);
    const abdomen = add(this.body, new THREE.SphereGeometry(0.23, 18, 14), abdomenMat, 0, 0.46, -0.34); abdomen.scale.set(1, 0.8, 1.75); abdomen.rotation.x = Math.PI / 2;
    add(this.body, new THREE.SphereGeometry(0.05, 8, 6), chitin, 0, 0.62, 0.02);   // scutellum bump

    // head, compound eyes, antennae, proboscis
    this.head = new THREE.Group(); this.head.position.set(0, 0.55, 0.34); this.body.add(this.head);
    add(this.head, new THREE.SphereGeometry(0.16, 16, 12), chitin, 0, 0, 0).scale.set(1.1, 0.95, 0.9);
    for (const sx of [-1, 1]) {
      const eye = add(this.head, new THREE.SphereGeometry(0.11, 16, 12), this.eyeMat, sx * 0.11, 0.03, 0.06); eye.scale.set(0.8, 1.1, 1);
      const ant = add(this.head, new THREE.CylinderGeometry(0.008, 0.012, 0.26, 5), legMat, sx * 0.05, 0.12, 0.12); ant.rotation.set(-0.9, 0, sx * 0.5);
      const tip = add(this.head, new THREE.SphereGeometry(0.018, 6, 5), legMat, sx * 0.13, 0.2, 0.28); void tip;
    }
    const prob = add(this.head, new THREE.ConeGeometry(0.03, 0.14, 6), legMat, 0, -0.1, 0.06); prob.rotation.x = 0.3;

    // six legs: hip pivot on the thorax, femur out and down, tibia down to the ground
    for (const side of [-1, 1]) for (let row = 0; row < 3; row++) {
      const hip = new THREE.Group(); hip.position.set(side * 0.16, 0.4, 0.22 - row * 0.2); this.body.add(hip);
      const femur = add(hip, new THREE.CapsuleGeometry(0.028, 0.18, 3, 6), legMat, side * 0.1, 0.03, 0); femur.rotation.z = side * 1.3;
      const knee = new THREE.Group(); knee.position.set(side * 0.2, 0.07, 0); hip.add(knee);
      const tibia = add(knee, new THREE.CapsuleGeometry(0.02, 0.3, 3, 6), legMat, side * 0.04, -0.2, 0); tibia.rotation.z = side * 0.25;
      add(knee, new THREE.SphereGeometry(0.024, 6, 5), legMat, side * 0.08, -0.37, 0);
      this.legs.push({ hip, knee, side, row });
    }

    // wings: root at the thorax, lying flat, swept back at rest
    const wingGeo = new THREE.PlaneGeometry(0.62, 0.3); wingGeo.translate(0.31, 0, 0);
    this.wingL = new THREE.Group(); this.wingR = new THREE.Group();
    for (const [w, sx] of [[this.wingL, -1], [this.wingR, 1]] as const) {
      w.position.set(sx * 0.07, 0.66, 0.0);
      const m = new THREE.Mesh(wingGeo, wingMat); m.rotation.x = -Math.PI / 2; if (sx < 0) m.scale.x = -1; w.add(m);
      const haltere = add(w, new THREE.SphereGeometry(0.02, 6, 5), chitin, sx * 0.12, -0.08, -0.06); void haltere;
      this.body.add(w);
    }
    this.body.scale.setScalar(1.15);
    this.group.add(this.body);
  }

  pose(p: Pose): void {
    this.group.position.set(p.x, p.y, p.z);
    this.group.rotation.set(0, 0, 0); this.body.position.set(0, 0, 0); this.body.rotation.set(0, 0, 0);
    this.head.rotation.set(0, 0, 0);
    const buzz = p.t * 75, cyc = p.t * (22 + 16 * p.speedFrac);
    const wing = (rest: number, amp: number) => { const f = Math.sin(buzz) * amp; this.wingL.rotation.set(0, rest, f); this.wingR.rotation.set(0, -rest, -f); };
    const leg = (i: number, swing: number, lift: number, splay = 0) => { const L = this.legs[i]; L.hip.rotation.set(swing, 0, L.side * splay); L.knee.rotation.z = L.side * -lift; };
    if (p.dead) {
      const k = Math.min(1, p.deadT * 3);
      this.body.rotation.z = k * Math.PI; this.body.position.y = -k * 0.15; this.body.rotation.x = k * 0.3;
      for (let i = 0; i < 6; i++) leg(i, Math.sin(i) * 0.3, 1.6 * k, -0.6 * k);   // legs curl up
      wing(0.9, 0); this.eyeMat.color.setHex(0x6a3a3a);
      return;
    }
    this.eyeMat.color.setHex(0xc0262e);
    if (p.sliding) {
      // flatten against the path: body drops, legs splay out sideways, wings pressed flat
      this.body.position.y = -0.3; this.body.rotation.x = 0.15;
      for (let i = 0; i < 6; i++) leg(i, (this.legs[i].row - 1) * 0.6, -0.5, 0.9);
      wing(0.55, 0.05); this.head.rotation.x = -0.3;
    } else if (p.airborne) {
      // wings wide and buzzing, legs tucked back, nose up
      this.body.rotation.x = -0.25; this.body.position.y = 0.1;
      for (let i = 0; i < 6; i++) leg(i, -0.9, 0.9, -0.3);
      wing(-0.15, 0.55);
    } else {
      // tripod gait: (L1, R2, L3) swing together, (R1, L2, R3) opposite
      for (let i = 0; i < 6; i++) {
        const L = this.legs[i], ph = ((L.row + (L.side > 0 ? 1 : 0)) % 2) ? 0 : Math.PI;
        const s = Math.sin(cyc + ph);
        leg(i, s * 0.55 + (L.row - 1) * 0.35, Math.max(0, Math.cos(cyc + ph)) * 0.6, 0);
      }
      wing(0.6, 0.12 + p.speedFrac * 0.1);
      this.body.position.y = Math.abs(Math.sin(cyc)) * 0.02; this.body.rotation.x = 0.05 + p.speedFrac * 0.1;
      this.head.rotation.y = Math.sin(p.t * 3) * 0.15;   // looking around
    }
    // lean is +1 when moving toward screen-right (world -x after the renderer's mirror)
    this.body.rotation.z += p.lean * 0.5; this.body.rotation.y += -p.lean * 0.45;
  }
}
