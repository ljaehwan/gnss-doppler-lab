#!/usr/bin/env python3
"""Replay one pinned TEXBAT scenario with the additive 1 ms tracker exporter."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()


def rewrite_config(source: Path, destination: Path, raw: Path, raw_dir: Path) -> None:
    values=[];seen=set()
    replacements={"SignalSource.filename":str(raw),"Tracking_1C.dump":"true",
                  "Tracking_1C.dump_filename":str(raw_dir/"epl_tracking_ch_"),
                  "Tracking_1C.extend_correlation_symbols":"1","Observables.dump_filename":str(raw_dir/"observables.dat")}
    for line in source.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key=line.split("=",1)[0].strip()
            if key in replacements: line=f"{key}={replacements[key]}";seen.add(key)
        values.append(line)
    for key in sorted(set(replacements)-seen): values.append(f"{key}={replacements[key]}")
    destination.write_text("\n".join(values)+"\n",encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    cfg=json.loads(args.source_binding.read_text(encoding="utf-8"));source=cfg["scenarios"][args.scenario]
    raw=Path(source["raw_path"]);binary=args.gnss_sdr.resolve();output=args.output.resolve()
    if sha256(binary)!=args.binary_sha256:raise RuntimeError("exporter binary SHA256 mismatch")
    if not raw.is_file() or raw.stat().st_size%4:raise RuntimeError("raw IQ unavailable or malformed")
    raw_dir=output/"raw";receiver_config=output/"receiver.conf"
    if args.finalize_existing:
        if not raw_dir.is_dir() or not receiver_config.is_file() or not (output/"receiver.log").is_file():
            raise RuntimeError("existing replay is incomplete")
    else:
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True);raw_dir.mkdir()
        rewrite_config(Path(source["receiver_config_path"]),receiver_config,raw,raw_dir)
    command=[str(binary),f"--config_file={receiver_config}","--keyboard=false"]
    exit_code=0
    if not args.finalize_existing:
        with (output/"receiver.log").open("w",encoding="utf-8") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=output)
        exit_code=result.returncode
        if exit_code!=0:raise RuntimeError(f"receiver failed with exit {exit_code}")
    canonical=lambda p: bool(re.fullmatch(r"(?:epl_tracking_ch_|epl_track|epl_|e)\d+\.(?:mat|dat)",p.name))
    mats=sorted(p for p in raw_dir.glob("*.mat") if canonical(p));dats=sorted(p for p in raw_dir.glob("*.dat") if canonical(p))
    mat_stems={p.stem for p in mats};dat_stems={p.stem for p in dats}
    if not mats or mat_stems!=dat_stems:raise RuntimeError("receiver did not produce paired MAT/DAT files")
    provenance=json.loads(args.build_provenance.read_text(encoding="utf-8"))
    manifest={"schema":"acaf_nf_continuous_receiver_replay.v1","generated_at_utc":datetime.now(timezone.utc).isoformat(),
              "scenario":args.scenario,"source":{"path":str(raw),"sha256":source["raw_sha256"],
              "iq_sha256":source["raw_sha256"],"size_bytes":raw.stat().st_size,
              "sample_rate_hz":25_000_000,"sample_format":"ishort_complex_iq","format":"ishort_complex_iq",
              "first_file_sample":0},
              "receiver":{"command":command,"exit_code":exit_code,"config":str(receiver_config),
              "config_sha256":sha256(receiver_config),"original_config":source["receiver_config_path"],
              "original_config_sha256":source["receiver_config_sha256"],"executable":str(binary),
              "executable_sha256":sha256(binary),"build_provenance":str(args.build_provenance),
              "build_provenance_sha256":sha256(args.build_provenance),"exporter_patch_sha256":provenance["exporter_patch_sha256"]},
              "tracking":{"dump_contract":"valid tracking loop at every 1 ms integration; telemetry output unchanged",
              "extend_correlation_symbols":1,"mat_inventory":{p.name:sha256(p) for p in mats},
              "dat_inventory":{p.name:sha256(p) for p in dats}}}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return output


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--scenario",choices=("cleanStatic","ds3","ds4","ds7","ds8"),required=True);p.add_argument("--source-binding",type=Path,default=ROOT/"configs/acaf_nf_stage1_source_binding.json")
    p.add_argument("--gnss-sdr",type=Path,required=True);p.add_argument("--binary-sha256",required=True);p.add_argument("--build-provenance",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--finalize-existing",action="store_true",help="validate and manifest an already completed receiver replay")
    print(run(p.parse_args()))


if __name__=="__main__":main()
