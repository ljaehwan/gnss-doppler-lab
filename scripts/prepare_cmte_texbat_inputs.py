#!/usr/bin/env python3
"""Prepare explicit canonical CMTE node inputs; never searches for data."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from gnss_doppler_lab.cmte_inputs import FEATURES, convert_complex_npz, copy_verified_ds4, sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("cleanstatic","ds1","ds2","ds3"):
        p.add_argument(f"--{name}-npz",required=True); p.add_argument(f"--{name}-source-manifest",required=True)
    p.add_argument("--ds4-node-csv",required=True); p.add_argument("--ds4-source-manifest"); p.add_argument("--clean-historical-node-csv"); p.add_argument("--checkpoint-sha",required=True); p.add_argument("--out",required=True)
    a=p.parse_args(argv); out=Path(a.out); out.mkdir(parents=True,exist_ok=False); results={}
    for arg,scenario in (("cleanstatic_npz","cleanStatic"),("ds1_npz","DS1"),("ds2_npz","DS2"),("ds3_npz","DS3")):
        source_manifest=Path(getattr(a,arg.replace("_npz","_source_manifest"))).resolve(strict=True)
        source_doc=json.loads(source_manifest.read_text())
        source_metadata={"manifest_path":str(source_manifest),"manifest_sha256":sha256(source_manifest),"schema":source_doc.get("schema"),"schema_version":source_doc.get("schema_version"),"causality":source_doc.get("causality"),"feature_schema":source_doc.get("feature_schema")}
        csv=out/f"{scenario}_nodes.csv"; man=out/f"{scenario}_manifest.json"; frame=convert_complex_npz(getattr(a,arg),csv,man,scenario=scenario,source_metadata=source_metadata)
        doc=json.loads(man.read_text()); doc["checkpoint_sha256"]=a.checkpoint_sha.lower(); man.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); results[scenario]={"rows":len(frame),"node":str(csv),"manifest":str(man)}
    csv=out/"DS4_nodes.csv"; man=out/"DS4_manifest.json"; frame=copy_verified_ds4(a.ds4_node_csv,csv,man,source_manifest=a.ds4_source_manifest)
    doc=json.loads(man.read_text()); doc["checkpoint_sha256"]=a.checkpoint_sha.lower(); man.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); results["DS4"]={"rows":len(frame),"node":str(csv),"manifest":str(man)}
    equivalence={"status":"not_supplied","mae":None,"max_abs_difference":None,"matched_rows":0}
    if a.clean_historical_node_csv:
        historical=pd.read_csv(Path(a.clean_historical_node_csv).resolve(strict=True)); reconstructed=pd.read_csv(out/"cleanStatic_nodes.csv")
        keys=[c for c in ("prn","channel","window_start_s","window_end_s") if c in historical and c in reconstructed]
        joined=reconstructed.merge(historical,on=keys,suffixes=("_new","_historical"))
        differences=np.concatenate([(joined[f"{c}_new"]-joined[f"{c}_historical"]).to_numpy(float) for c in FEATURES]) if len(joined) else np.asarray([])
        equivalence={"status":"measured","historical_path":str(Path(a.clean_historical_node_csv).resolve()),"mae":None if not len(differences) else float(np.mean(np.abs(differences))),"max_abs_difference":None if not len(differences) else float(np.max(np.abs(differences))),"matched_rows":int(len(joined))}
    (out/"bundle_manifest.json").write_text(json.dumps({"schema":"cmte-input-bundle-v1","checkpoint_sha256":a.checkpoint_sha.lower(),"inputs":results,
      "historical_equivalence_check":equivalence},indent=2,sort_keys=True)+"\n")
    print(json.dumps(results,sort_keys=True))
if __name__=="__main__": main()
