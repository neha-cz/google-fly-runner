import * as THREE from "three";

export interface Pose { x: number; y: number; z: number; t: number; airborne: boolean; sliding: boolean; lean: number; dead: boolean; deadT: number; speedFrac: number }

/** Low-poly running explorer built from primitives; limbs pivot at shoulders and hips. */
export class Explorer {
  group = new THREE.Group();
  private body = new THREE.Group();
  private armL: THREE.Group; private armR: THREE.Group; private legL: THREE.Group; private legR: THREE.Group;
  private head: THREE.Mesh;

  constructor() {
    const skin = new THREE.MeshStandardMaterial({ color: 0xd9a983, roughness: 0.8 });
    const shirt = new THREE.MeshStandardMaterial({ color: 0xc8b48a, roughness: 0.9 });
    const pants = new THREE.MeshStandardMaterial({ color: 0x5a4a34, roughness: 0.9 });
    const hat = new THREE.MeshStandardMaterial({ color: 0x6b4a2a, roughness: 0.9 });

    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.42, 0.2), shirt); torso.position.y = 0.72; this.body.add(torso);
    const pack = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.3, 0.14), pants); pack.position.set(0, 0.74, -0.16); this.body.add(pack);
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), skin); this.head.position.y = 1.06; this.body.add(this.head);
    const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.03, 12), hat); brim.position.y = 1.13; this.body.add(brim);
    const crown = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.13, 0.12, 12), hat); crown.position.y = 1.2; this.body.add(crown);

    const limb = (len: number, r: number, mat: THREE.Material, y: number, x: number) => {
      const g = new THREE.Group(); g.position.set(x, y, 0);
      const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len - r * 2, 3, 6), mat); m.position.y = -len / 2; g.add(m);
      return g;
    };
    this.armL = limb(0.38, 0.055, skin, 0.9, -0.22); this.armR = limb(0.38, 0.055, skin, 0.9, 0.22);
    this.legL = limb(0.5, 0.07, pants, 0.5, -0.09); this.legR = limb(0.5, 0.07, pants, 0.5, 0.09);
    this.body.add(this.armL, this.armR, this.legL, this.legR);
    this.group.add(this.body);
  }

  pose(p: Pose): void {
    this.group.position.set(p.x, p.y, p.z);
    this.group.rotation.set(0, 0, 0); this.body.position.set(0, 0, 0); this.body.rotation.set(0, 0, 0); this.body.scale.set(1, 1, 1);
    const cyc = p.t * (16 + 10 * p.speedFrac);
    if (p.dead) {
      const k = Math.min(1, p.deadT * 3);
      this.body.rotation.x = -k * 1.5; this.body.position.y = -k * 0.35; this.body.position.z = k * 0.3;
      return;
    }
    if (p.sliding) {
      this.body.rotation.x = -1.25; this.body.position.y = -0.55; this.body.position.z = -0.15;
      this.armL.rotation.x = this.armR.rotation.x = -2.6; this.legL.rotation.x = this.legR.rotation.x = 0.2;
      return;
    }
    if (p.airborne) {
      this.legL.rotation.x = -0.9; this.legR.rotation.x = 0.6; this.armL.rotation.x = -2.4; this.armR.rotation.x = -2.0;
      this.body.rotation.x = -0.15;
    } else {
      const s = Math.sin(cyc);
      this.legL.rotation.x = s * 0.9; this.legR.rotation.x = -s * 0.9;
      this.armL.rotation.x = -s * 0.8; this.armR.rotation.x = s * 0.8;
      this.body.position.y = Math.abs(Math.cos(cyc)) * 0.04; this.body.rotation.x = -0.12;
    }
    // lean is +1 when moving toward screen-right (world -x after the renderer's mirror)
    this.body.rotation.z = p.lean * 0.35; this.body.rotation.y = -p.lean * 0.4;
  }
}
