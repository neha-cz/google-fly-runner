import * as THREE from "three";

/**
 * Texture set generated with Higgsfield (Z Image) and post-processed into seamless tiles with
 * normal / roughness maps. Files live in `public/textures/`; see docs/README for the pipeline.
 */
export const TEXTURE_FILES = {
  floor: "floor.jpg", floor_n: "floor_n.jpg", floor_r: "floor_r.jpg",
  path: "floor_alt.jpg", path_n: "floor_alt_n.jpg", path_r: "floor_alt_r.jpg",
  ground: "ground.jpg", ground_n: "ground_n.jpg", ground_r: "ground_r.jpg",
  wall: "wall.jpg", wall_n: "wall_n.jpg", wall_r: "wall_r.jpg",
  bark: "bark.jpg", bark_n: "bark_n.jpg", bark_r: "bark_r.jpg",
  foliage: "foliage.jpg", foliage_n: "foliage_n.jpg", foliage_r: "foliage_r.jpg",
  idol: "idol.jpg", idol_n: "idol_n.jpg", idol_r: "idol_r.jpg",
  coin: "coin.png",
  sky: "sky.jpg",
} as const;
export type TextureName = keyof typeof TEXTURE_FILES;

/** Materials that have a full base / normal / roughness set. */
export type PbrName = "floor" | "path" | "ground" | "wall" | "bark" | "foliage" | "idol";

export class Assets {
  private base = new Map<TextureName, THREE.Texture>();
  ready = false;

  constructor(private anisotropy: number, onReady: () => void, onError: (url: string) => void) {
    const manager = new THREE.LoadingManager(() => { this.ready = true; onReady(); }, undefined, onError);
    const loader = new THREE.TextureLoader(manager);
    for (const [name, file] of Object.entries(TEXTURE_FILES) as [TextureName, string][]) {
      const t = loader.load(`textures/${file}`);
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = anisotropy;
      // colour maps are sRGB; normal / roughness data stays linear
      if (!name.endsWith("_n") && !name.endsWith("_r")) t.colorSpace = THREE.SRGBColorSpace;
      this.base.set(name, t);
    }
  }

  /** A texture with its own repeat, sharing the decoded image with the loaded base. */
  tex(name: TextureName, rx = 1, ry = 1): THREE.Texture {
    const t = this.base.get(name)!.clone();
    t.repeat.set(rx, ry); t.anisotropy = this.anisotropy; t.needsUpdate = true;
    return t;
  }

  /** Standard PBR material from a base / normal / roughness set. */
  pbr(name: PbrName, rx = 1, ry = 1, opts: THREE.MeshStandardMaterialParameters = {}): THREE.MeshStandardMaterial {
    return new THREE.MeshStandardMaterial({
      map: this.tex(name, rx, ry),
      normalMap: this.tex(`${name}_n` as TextureName, rx, ry),
      roughnessMap: this.tex(`${name}_r` as TextureName, rx, ry),
      normalScale: new THREE.Vector2(0.9, 0.9),
      roughness: 1, metalness: 0,
      ...opts,
    });
  }
}
