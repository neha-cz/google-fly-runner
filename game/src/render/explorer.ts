import * as THREE from "three";

export interface Pose { x: number; y: number; z: number; t: number; airborne: boolean; sliding: boolean; lean: number; dead: boolean; deadT: number; speedFrac: number }

/**
 * Running explorer built from primitives with a two-segment limb rig (shoulder/elbow, hip/knee),
 * fedora, satchel and boots. Every mesh casts a shadow. Purely cosmetic.
 */
export class Explorer {
  group = new THREE.Group();
  private body = new THREE.Group();
  private armL: THREE.Group; private armR: THREE.Group; private legL: THREE.Group; private legR: THREE.Group;
  private foreL: THREE.Group; private foreR: THREE.Group; private shinL: THREE.Group; private shinR: THREE.Group;
  private head: THREE.Group;

  constructor() {
    const mat = (color: number, roughness = 0.85, metalness = 0) => new THREE.MeshStandardMaterial({ color, roughness, metalness });
    const skin = mat(0xd8a57c, 0.7), shirt = mat(0xdccba3, 0.9), vest = mat(0x6b4a2f, 0.95), pants = mat(0x4f4634, 0.95);
    const boots = mat(0x3a2a1c, 0.8), hat = mat(0x5a3d22, 0.9), leather = mat(0x7a5a33, 0.85), brass = mat(0xc9a24a, 0.4, 0.8);
    const add = (parent: THREE.Object3D, geo: THREE.BufferGeometry, m: THREE.Material, x = 0, y = 0, z = 0) => {
      const mesh = new THREE.Mesh(geo, m); mesh.position.set(x, y, z); mesh.castShadow = true; parent.add(mesh); return mesh;
    };

    // torso: shirt block, open vest over it, belt with buckle, satchel on the hip
    add(this.body, new THREE.BoxGeometry(0.36, 0.46, 0.22), shirt, 0, 0.87);
    add(this.body, new THREE.BoxGeometry(0.4, 0.34, 0.26), vest, 0, 0.9);
    add(this.body, new THREE.BoxGeometry(0.38, 0.06, 0.24), leather, 0, 0.66);
    add(this.body, new THREE.BoxGeometry(0.08, 0.07, 0.02), brass, 0, 0.66, 0.125);
    const strap = add(this.body, new THREE.BoxGeometry(0.05, 0.5, 0.25), leather, 0.1, 0.9); strap.rotation.z = 0.5;
    add(this.body, new THREE.BoxGeometry(0.2, 0.16, 0.1), leather, -0.22, 0.68, -0.02);
    add(this.body, new THREE.BoxGeometry(0.24, 0.3, 0.12), leather, 0, 0.9, -0.17);   // backpack

    // head, neck, fedora
    this.head = new THREE.Group(); this.head.position.y = 1.16; this.body.add(this.head);
    add(this.body, new THREE.CylinderGeometry(0.06, 0.07, 0.1, 8), skin, 0, 1.12);
    add(this.head, new THREE.SphereGeometry(0.14, 14, 12), skin, 0, 0.08);
    add(this.head, new THREE.BoxGeometry(0.06, 0.06, 0.02), mat(0x2a1a10), 0, 0.09, 0.135);   // brow shadow
    add(this.head, new THREE.CylinderGeometry(0.23, 0.23, 0.025, 18), hat, 0, 0.16);
    add(this.head, new THREE.CylinderGeometry(0.13, 0.145, 0.15, 18), hat, 0, 0.24);
    add(this.head, new THREE.CylinderGeometry(0.15, 0.15, 0.03, 18), mat(0x2b1d12), 0, 0.185);   // hat band

    // limbs: two segments each; the child pivot sits at the elbow / knee
    const seg = (len: number, r: number, m: THREE.Material) => { const g = new THREE.Group(); add(g, new THREE.CapsuleGeometry(r, len - r * 2, 3, 8), m, 0, -len / 2); return g; };
    const limb = (x: number, y: number, upper: number, lower: number, r: number, mUp: THREE.Material, mLow: THREE.Material) => {
      const g = seg(upper, r, mUp); g.position.set(x, y, 0);
      const f = seg(lower, r * 0.9, mLow); f.position.y = -upper; g.add(f);
      return [g, f] as const;
    };
    [this.armL, this.foreL] = limb(-0.24, 1.05, 0.28, 0.26, 0.055, shirt, skin);
    [this.armR, this.foreR] = limb(0.24, 1.05, 0.28, 0.26, 0.055, shirt, skin);
    [this.legL, this.shinL] = limb(-0.1, 0.64, 0.32, 0.3, 0.07, pants, pants);
    [this.legR, this.shinR] = limb(0.1, 0.64, 0.32, 0.3, 0.07, pants, pants);
    for (const shin of [this.shinL, this.shinR]) add(shin, new THREE.BoxGeometry(0.15, 0.1, 0.24), boots, 0, -0.29, 0.04);
    for (const fore of [this.foreL, this.foreR]) add(fore, new THREE.SphereGeometry(0.06, 8, 6), skin, 0, -0.26);
    this.body.add(this.armL, this.armR, this.legL, this.legR);
    this.group.add(this.body);
  }

  pose(p: Pose): void {
    this.group.position.set(p.x, p.y, p.z);
    this.group.rotation.set(0, 0, 0); this.body.position.set(0, 0, 0); this.body.rotation.set(0, 0, 0);
    this.head.rotation.set(0, 0, 0);
    const cyc = p.t * (15 + 11 * p.speedFrac);
    const set = (g: THREE.Group, x: number) => { g.rotation.x = x; };
    if (p.dead) {
      const k = Math.min(1, p.deadT * 3);
      this.body.rotation.x = -k * 1.5; this.body.position.y = -k * 0.4; this.body.position.z = k * 0.3;
      set(this.armL, -1.2); set(this.armR, -1.2); set(this.foreL, -0.8); set(this.foreR, -0.8);
      return;
    }
    if (p.sliding) {
      this.body.rotation.x = -1.1; this.body.position.y = -0.45; this.body.position.z = -0.1;
      set(this.armL, -2.4); set(this.armR, -2.4); set(this.foreL, -0.4); set(this.foreR, -0.4);
      set(this.legL, 0.3); set(this.legR, 0.1); set(this.shinL, 0.4); set(this.shinR, 0.2);
      this.head.rotation.x = 0.7;
    } else if (p.airborne) {
      set(this.legL, -1.1); set(this.shinL, 1.6); set(this.legR, 0.5); set(this.shinR, 0.9);
      set(this.armL, -2.3); set(this.armR, -1.6); set(this.foreL, -0.9); set(this.foreR, -1.3);
      this.body.rotation.x = -0.2;
    } else {
      const s = Math.sin(cyc), c = Math.cos(cyc);
      // hips swing; the knee folds on the back-swing and straightens as the foot plants
      set(this.legL, s * 1.0); set(this.shinL, Math.max(0, -c) * 1.5 * (s > 0 ? 0.3 : 1) + 0.15);
      set(this.legR, -s * 1.0); set(this.shinR, Math.max(0, c) * 1.5 * (s < 0 ? 0.3 : 1) + 0.15);
      set(this.armL, -s * 0.9 - 0.2); set(this.armR, s * 0.9 - 0.2);
      set(this.foreL, -1.3 + s * 0.3); set(this.foreR, -1.3 - s * 0.3);
      this.body.position.y = Math.abs(c) * 0.05; this.body.rotation.x = -0.14 - p.speedFrac * 0.08;
      this.body.rotation.y = s * 0.12;   // shoulder counter-rotation
    }
    // lean is +1 when moving toward screen-right (world -x after the renderer's mirror)
    this.body.rotation.z += p.lean * 0.35; this.body.rotation.y += -p.lean * 0.4;
  }
}
