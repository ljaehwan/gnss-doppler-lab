#!/usr/bin/env python3
"""Build the additive 1 ms GNSS-SDR tracker exporter without touching its dirty source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(args: argparse.Namespace) -> Path:
    source = args.source.resolve(); output = args.output.resolve(); patch = args.patch.resolve()
    if output.exists(): raise FileExistsError(output)
    if not (source / ".git").exists() or not patch.is_file(): raise ValueError("source git tree and exporter patch required")
    output.mkdir(parents=True)
    temp = Path(tempfile.mkdtemp(prefix="acaf-nf-gnss-sdr-"))
    clone = temp / "source"; build = temp / "build"
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True, capture_output=True).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--binary"], cwd=source, check=True, capture_output=True).stdout
        subprocess.run(["git", "clone", "--local", "--no-hardlinks", str(source), str(clone)], check=True)
        subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=clone, input=dirty, check=True)
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=clone, check=True)
        configure = ["cmake", "-G", "Ninja", "-S", str(clone), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     "-DENABLE_UNIT_TESTING=OFF", "-DENABLE_SYSTEM_TESTING=OFF", "-DENABLE_CUDA=OFF",
                     "-DENABLE_OPENCL=OFF", "-DENABLE_FPGA=OFF", "-DENABLE_OSMOSDR=OFF"]
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "gnss-sdr", "-j", str(args.jobs)], check=True)
        built = build / "src/main/gnss-sdr"
        if not built.is_file(): raise RuntimeError("GNSS-SDR binary was not produced")
        destination = output / "gnss-sdr-continuous-1ms"
        shutil.copy2(built, destination)
        source_file = clone / "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc"
        provenance = {
            "schema": "acaf_nf_gnss_sdr_continuous_exporter_build.v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_tree": str(source), "source_git_head": head,
            "source_dirty_diff_sha256": hashlib.sha256(dirty).hexdigest(), "source_dirty_diff_bytes": len(dirty),
            "exporter_patch": str(patch.relative_to(ROOT)), "exporter_patch_sha256": sha256(patch),
            "patched_tracking_source_sha256": sha256(source_file), "configure_command": configure,
            "build_command": ["cmake", "--build", str(build), "--target", "gnss-sdr", "-j", str(args.jobs)],
            "binary": destination.name, "binary_sha256": sha256(destination), "binary_size_bytes": destination.stat().st_size,
        }
        (output / "build_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--patch",type=Path,default=ROOT/"patches/gnss-sdr-valid-1ms-tracking-dump.patch");parser.add_argument("--jobs",type=int,default=18)
    args=parser.parse_args();print(run(args))


if __name__=="__main__": main()
