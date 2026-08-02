#!/usr/bin/env python3
"""One-shot DS7/DS8 confirmation behind a byte-first trust-anchor guard."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
# Security-only imports. The scorer module is dynamically imported only after every guard.
from gnss_doppler_lab.cmte_a2_campaign import (validate_trust_anchor,resolve_confirmatory_inputs,create_ledger,
 update_ledger,file_sha256)

def _load_guarded_evaluator():
 path=ROOT/"scripts/eval_cmte_a2_texbat.py"; spec=importlib.util.spec_from_file_location("cmte_a2_guarded_confirm_evaluator",path)
 if spec is None or spec.loader is None: raise RuntimeError("cannot load guarded evaluator")
 module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trust-anchor",required=True); p.add_argument("--expected-sha256",required=True)
 p.add_argument("--ledger",required=True); p.add_argument("--out",required=True); p.add_argument("--device",default="cpu"); p.add_argument("--repo",default=str(ROOT)); a=p.parse_args(argv)
 # ORDER IS A SECURITY CONTRACT: no DS7/DS8 path resolution/stat/open before this completes.
 anchor=validate_trust_anchor(a.trust_anchor,a.expected_sha256,repo=a.repo)
 inputs=resolve_confirmatory_inputs(anchor)
 ledger=create_ledger(a.ledger,a.expected_sha256)  # O_EXCL before scoring; retained on all failures.
 try:
  state=Path(anchor["state_dir"])
  args=["--tier","confirmatory","--state-dir",str(state),"--freeze-manifest",str(state/"freeze_manifest.json"),
        "--scenario",f"DS7={inputs['DS7']['node']}={inputs['DS7']['input_manifest']}",
        "--scenario",f"DS8={inputs['DS8']['node']}={inputs['DS8']['input_manifest']}","--out",a.out,"--device",a.device]
  _load_guarded_evaluator().main(args)
  result_hash=file_sha256(Path(a.out)/"checksums.json"); update_ledger(a.ledger,status="completed",result_sha256=result_hash)
  print(json.dumps({"out":str(Path(a.out).resolve()),"ledger":str(Path(a.ledger).resolve()),"status":"completed","result_checksums_sha256":result_hash},sort_keys=True))
 except BaseException as exc:
  update_ledger(a.ledger,status="failed",detail=f"{type(exc).__name__}: {exc}"); raise
if __name__=="__main__": main()
