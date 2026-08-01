#!/usr/bin/env python3
"""Finalize CLIF-IP checksums after test_summary is complete."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_cross_layer_r3"));a=ap.parse_args()
    manifest_path=a.out/"provenance_manifest.json";manifest=json.loads(manifest_path.read_text())
    if not (a.out/"test_summary.txt").exists(): raise SystemExit("test_summary.txt must be written before checksums")
    manifest["checksum_policy"]="SHA-256 of every artifact file after test_summary finalization; provenance_manifest.json excludes itself"
    manifest["artifact_checksums"]={str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob("*")) if p.is_file() and p!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"verified manifest entries: {len(manifest['artifact_checksums'])}")

if __name__=="__main__":main()
