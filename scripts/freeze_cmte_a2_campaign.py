#!/usr/bin/env python3
"""Create the immutable external CMTE-A2 pre-holdout campaign trust anchor."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2_campaign import atomic_json,collect_freeze,file_sha256

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--state-dir",required=True); p.add_argument("--development-dir",required=True)
 p.add_argument("--confirm-input-manifest",required=True); p.add_argument("--out-dir","--out",dest="out",required=True); p.add_argument("--repo",default=str(ROOT)); a=p.parse_args(argv)
 out=Path(a.out).absolute()
 if out.exists(): raise FileExistsError("freeze output is immutable and non-overwriting")
 staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
 try:
  doc=collect_freeze(a.repo,a.state_dir,a.development_dir,a.confirm_input_manifest)
  manifest=staging/"freeze_manifest.json"; atomic_json(manifest,doc)
  digest=file_sha256(manifest); (staging/"freeze_manifest.sha256").write_text(f"{digest}  freeze_manifest.json\n")
  os.chmod(manifest,0o444); os.chmod(staging/"freeze_manifest.sha256",0o444); os.replace(staging,out)
  print(json.dumps({"trust_anchor":str(out/"freeze_manifest.json"),"trust_anchor_sha256":digest,
                    "source_commit":doc["source_commit"],"immutable":True},sort_keys=True))
 except Exception:
  import shutil; shutil.rmtree(staging,ignore_errors=True); raise
if __name__=="__main__": main()
