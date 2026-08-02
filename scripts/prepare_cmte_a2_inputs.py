#!/usr/bin/env python3
"""Prepare named canonical CMTE-A2 node inputs with one converter contract."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_inputs import prepare_named_complex_inputs, prepare_ds4_sensitivity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-npz",action="append",default=[],help="NAME=/complex.npz; cleanStatic/DS1-3/DS7/DS8")
    parser.add_argument("--ds4-node",help="optional historical mixed-producer DS4 development sensitivity")
    parser.add_argument("--ds4-manifest")
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    if not args.scenario_npz and not args.ds4_node: parser.error("at least one source mapping is required")
    result={}
    if args.scenario_npz: result["complex"]=prepare_named_complex_inputs(args.scenario_npz,args.out)
    if args.ds4_node:
        if args.scenario_npz: raise ValueError("DS4 mixed producer must use a separate output directory/campaign")
        result["DS4"]=prepare_ds4_sensitivity(args.ds4_node,args.out,source_manifest=args.ds4_manifest)
    print(json.dumps({"out":str(Path(args.out).resolve()),"scenarios":sorted(result)},sort_keys=True))
if __name__=="__main__": main()
