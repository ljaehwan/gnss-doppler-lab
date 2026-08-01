#!/usr/bin/env python3
"""Fail-closed CLIF-IP R2 provenance gate; this script never imports an evaluator."""
import argparse,json
from pathlib import Path
REQUIRED_BOOLEAN_PROOFS=("same_recording_proven","sample_rate_match_proven","same_time_origin_proven","same_physical_interval_proven","sample_ranges_calculable","causal_alignment_proven")
REQUIRED_FIELDS=("scenario","raw_iq_path","raw_iq_sha256","sample_rate_hz","b0_source","m1_source","b0_window_sample_range","m1_block_sample_range","alignment_offset_samples")
def scenario_permitted(row):
 origin=row.get("recording_start_sample") is not None or row.get("explicit_time_origin") is not None
 sources=all(isinstance(row.get(k),dict) and row[k].get("path") and row[k].get("sha256") for k in ("b0_source","m1_source"))
 basics=all(row.get(k) is not None for k in REQUIRED_FIELDS)
 return bool(origin and sources and basics and all(row.get(k) is True for k in REQUIRED_BOOLEAN_PROOFS))
def validate_manifest(manifest):
 rows=manifest.get("scenarios",[])
 for row in rows:
  computed=scenario_permitted(row)
  if row.get("permitted") is not computed: raise ValueError(f"{row.get('scenario')}: permitted does not match fail-closed proof")
 return [r["scenario"] for r in rows if r["permitted"]]
def invoke_evaluator_if_permitted(manifest,evaluator):
 names=validate_manifest(manifest)
 if not names:return False
 evaluator(names);return True
def main():
 p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);a=p.parse_args();m=json.loads(a.manifest.read_text());names=validate_manifest(m)
 print(json.dumps({"gate":"PERMITTED" if names else "BLOCKED","permitted_scenarios":names}))
 return 0 if names else 2
if __name__=="__main__":raise SystemExit(main())
