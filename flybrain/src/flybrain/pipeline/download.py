"""Download the MaleCNS v1.0 flat-connectome bulk files (public GCS bucket, CC BY 4.0).

Resumable; records size + sha256 of each file in data/raw/manifest.json so a graph build can
prove exactly which inputs it used.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return
        raise
    mode = "ab" if resp.status == 206 else "wb"
    total = int(resp.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
    done, t0 = have if mode == "ab" else 0, time.time()
    with open(dest, mode) as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total and time.time() - t0 > 2:
                print(f"\r  {dest.name}: {done/1e6:,.0f}/{total/1e6:,.0f} MB", end="", flush=True); t0 = time.time()
    print(f"\r  {dest.name}: {done/1e6:,.0f} MB done")


def run(config: dict, raw_dir: Path) -> dict:
    src = config["source"]
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, name in src["files"].items():
        dest = raw_dir / name
        if not dest.exists():
            print(f"downloading {name}")
            download(f"{src['base_url']}/{name}", dest)
        entry = manifest.get(name)
        if not entry or entry.get("size") != dest.stat().st_size:
            print(f"hashing {name} ...", end=" ", flush=True)
            entry = {"key": key, "url": f"{src['base_url']}/{name}", "size": dest.stat().st_size, "sha256": sha256(dest)}
            print(entry["sha256"][:16])
            manifest[name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
