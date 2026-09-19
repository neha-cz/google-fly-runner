import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { selfTest, type Bundle } from "../neural";

const dir = new URL("../../../public/agents/", import.meta.url);
const files = existsSync(dir) ? readdirSync(dir).filter(f => f.endsWith(".json") && f !== "index.json") : [];

describe("exported agent bundles", () => {
  it.skipIf(files.length === 0)("JS runtime reproduces PyTorch logits for every bundle", () => {
    for (const f of files) {
      const b: Bundle = JSON.parse(readFileSync(new URL(f, dir), "utf8"));
      const err = selfTest(b);
      expect(err, `${f}: max |Δlogit| = ${err}`).toBeLessThan(1e-3);
    }
  });
});
