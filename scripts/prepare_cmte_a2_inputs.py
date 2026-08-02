#!/usr/bin/env python3
"""Prepare named canonical CMTE-A2 node inputs with one converter contract."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_inputs import (CANONICAL_COMPLEX_SCENARIOS,prepare_named_complex_inputs,prepare_ds4_sensitivity)


def parse_scenario_npz_mappings(items):
    # A normal dict preserves repeated-argument order on supported Python versions.
    mappings={}
    for item in items:
        if not isinstance(item,str) or "=" not in item:
            raise ValueError("scenario mapping must be NAME=PATH")
        name,raw=item.split("=",1)
        if not name: raise ValueError("scenario name must be non-empty")
        if not raw: raise ValueError(f"scenario path must be non-empty: {name}")
        if name not in CANONICAL_COMPLEX_SCENARIOS:
            allowed=", ".join(CANONICAL_COMPLEX_SCENARIOS)
            raise ValueError(f"unsupported complex scenario {name}; allowed: {allowed}")
        if name in mappings: raise ValueError(f"duplicate scenario {name}")
        mappings[name]=Path(raw)
    return mappings


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-npz",action="append",default=[],help="NAME=/complex.npz; cleanStatic/DS1-3/DS7/DS8")
    parser.add_argument("--ds4-node",help="optional historical mixed-producer DS4 development sensitivity")
    parser.add_argument("--ds4-manifest")
    parser.add_argument("--out",required=True)
    parser.add_argument("--onset-seconds",type=float,help="authoritative metadata onset copied uniformly; preparation never infers it")
    args=parser.parse_args(argv)
    if not args.scenario_npz and not args.ds4_node: parser.error("at least one source mapping is required")
    if args.scenario_npz and args.ds4_node: raise ValueError("DS4 mixed producer must use a separate output directory/campaign")
    scenario_names=[]
    if args.scenario_npz:
        try: mappings=parse_scenario_npz_mappings(args.scenario_npz)
        except ValueError as exc: parser.error(str(exc))
        manifests=prepare_named_complex_inputs(mappings,args.out,source_metadata={"onset_s":args.onset_seconds} if args.onset_seconds is not None else None)
        scenario_names.extend(manifests)
    if args.ds4_node:
        prepare_ds4_sensitivity(args.ds4_node,args.out,source_manifest=args.ds4_manifest)
        scenario_names.append("DS4")
    print(json.dumps({"out":str(Path(args.out).resolve()),"scenarios":scenario_names},sort_keys=True))
if __name__=="__main__": main()
