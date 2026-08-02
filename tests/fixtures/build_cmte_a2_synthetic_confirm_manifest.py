#!/usr/bin/env python3
"""Committed no-TEXBAT fixture producer for the subprocess campaign test only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import CONVERTER_SHA,EXPORTER_SHA,RECEIVER_SHA,build_confirm_input_manifest,file_sha256

def main():
 p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--uniform",required=True); p.add_argument("--out",required=True); a=p.parse_args()
 source=Path(a.source); uniform=Path(a.uniform); fixture=Path(a.out).with_suffix(".fixture"); fixture.mkdir()
 rendered=fixture/"receiver.conf"; rendered.write_bytes((ROOT/"configs/cmte_a2_ds8_receiver.conf").read_bytes())
 prep=fixture/"ds8-prep.json"; prep.write_text(json.dumps({"status":"prepared","raw_sha256":file_sha256(source/"DS8.npz"),
  "npz":{"sha256":file_sha256(source/"DS8.npz")},"rendered_config_sha256":file_sha256(rendered),"test_fixture":True}))
 doc=build_confirm_input_manifest(a.out,ds7_raw=source/"DS7.npz",ds7_npz=source/"DS7.npz",ds7_node=uniform/"DS7_nodes.csv",
  ds7_input_manifest=uniform/"DS7_manifest.json",ds8_raw=source/"DS8.npz",ds8_npz=source/"DS8.npz",ds8_rendered_config=rendered,
  ds8_prep_manifest=prep,ds8_node=uniform/"DS8_nodes.csv",ds8_input_manifest=uniform/"DS8_manifest.json",
  expected_ds7_raw_sha=file_sha256(source/"DS7.npz"),expected_ds8_raw_sha=file_sha256(source/"DS8.npz"),
  code_hashes={"converter":CONVERTER_SHA,"wrapper":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"),
   "receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,"template":file_sha256(ROOT/"configs/cmte_a2_ds8_receiver.conf")})
 print(json.dumps({"manifest":str(Path(a.out).resolve()),"test_fixture":True,"files":len(doc["files"])}))
if __name__=="__main__": main()
