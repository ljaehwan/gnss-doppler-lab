#!/usr/bin/env python3
"""Prepare missing OAKBAT os1 M1 raw-IQ features for CLIF-IP R3.

This performs feature extraction only; the sole M1 fit occurs in eval_clif_ip.py
on cleanStatic 0--240 s.
"""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("artifacts/clif_ip_cross_layer_r3"));ap.add_argument("--force",action="store_true");a=ap.parse_args();cache=a.out/"input_cache";cache.mkdir(parents=True,exist_ok=True);dest=cache/"oakbat_os1_raw_iq_noise_features.csv"
 raw=Path("/home/ubuntu/unraid_hdd/oakbat/gps_l1ca/raw/os1.bin"); source=Path("/home/ubuntu/projects/gnss-doppler-lab/scripts/iq_noise_continuity_detector.py")
 if a.force or not dest.exists():
  spec=importlib.util.spec_from_file_location("m1extract",source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.FS=5_000_000;df=mod.extract_feature_frame(raw,"os1",block_ms=10.,stride_s=.5,max_s=None);df.to_csv(dest,index=False)
 else:df=pd.read_csv(dest)
 manifest={"schema":"clif-ip.r3.preparation.v1","scenario":"os1","raw":str(raw),"raw_bytes":raw.stat().st_size,"sample_rate_hz":5000000,"sample_format":"interleaved int16 IQ","recording_start_sample":0,"seek_samples":0,"block_ms":10.,"stride_s":.5,"rows":len(df),"output":str(dest),"note":"feature extraction only; no M1 fit"}
 (cache/"os1_extraction_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
