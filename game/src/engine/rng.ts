/**
 * xorshift32 PRNG. Pure uint32 arithmetic so the NumPy port (flybrain/env) can
 * reproduce the exact same stream — see the golden-trajectory fixture test.
 */
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 0x9e3779b9;
  }
  nextU32(): number {
    let x = this.s;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    this.s = x;
    return x;
  }
  /** Uniform float in [0, 1). */
  next(): number { return this.nextU32() / 4294967296; }
  range(lo: number, hi: number): number { return lo + (hi - lo) * this.next(); }
  /** Integer in [0, n). */
  int(n: number): number { return this.nextU32() % n; }
  state(): number { return this.s; }
}
