#!/usr/bin/env python3
"""Bind DS7/DS8 source, preparation, canonical-node and producer provenance."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import (CONVERTER_SHA,DS7_NPZ_SHA,DS7_RAW_SHA,DS8_RAW_SHA,EXPORTER_SHA,
 RECEIVER_SHA,build_confirm_input_manifest,file_sha256)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("ds7-raw","ds7-npz","ds7-node","ds7-input-manifest","ds8-raw","ds8-npz","ds8-rendered-config",
              "ds8-prep-manifest","ds8-node","ds8-input-manifest"): p.add_argument("--"+name,required=True)
 p.add_argument("--output",required=True); a=p.parse_args(argv)
 if file_sha256(a.ds7_npz)!=DS7_NPZ_SHA: raise ValueError("DS7 preregistered prepared NPZ SHA mismatch")
 hashes={"converter":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_inputs.py"),
  "wrapper":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"),"receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,
  "template":file_sha256(ROOT/"configs/cmte_a2_ds8_receiver.conf")}
 if hashes["converter"]!=CONVERTER_SHA: raise ValueError("stale base converter")
 prep=json.loads(Path(a.ds8_prep_manifest).read_text())
 if prep.get("wrapper_sha256")!=file_sha256(ROOT/"scripts/prepare_cmte_a2_ds8_complex.py"): raise ValueError("DS8 prep wrapper SHA mismatch")
 if prep.get("template_content_sha256")!=hashes["template"] or prep.get("binary_sha256")!=RECEIVER_SHA or prep.get("exporter_content_sha256")!=EXPORTER_SHA:
  raise ValueError("DS8 producer provenance mismatch")
 doc=build_confirm_input_manifest(a.output,ds7_raw=a.ds7_raw,ds7_npz=a.ds7_npz,ds7_node=a.ds7_node,
  ds7_input_manifest=a.ds7_input_manifest,ds8_raw=a.ds8_raw,ds8_npz=a.ds8_npz,ds8_rendered_config=a.ds8_rendered_config,
  ds8_prep_manifest=a.ds8_prep_manifest,ds8_node=a.ds8_node,ds8_input_manifest=a.ds8_input_manifest,
  expected_ds7_raw_sha=DS7_RAW_SHA,expected_ds8_raw_sha=DS8_RAW_SHA,code_hashes=hashes)
 print(json.dumps({"manifest":str(Path(a.output).resolve()),"sha256":file_sha256(a.output),"scenarios":doc["scenarios"],"scoring_performed":False},sort_keys=True))
if __name__=="__main__": main()
