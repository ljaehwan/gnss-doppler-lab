from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def make_npz(path:Path,end:float,phase:float):
 base=np.arange(0,end+.001,.5); times=np.tile(base,5); prn=np.repeat(np.arange(1,6),len(base)); n=len(times)
 iq=np.zeros((n,9,2),np.float32)
 for tap in range(9):
  iq[:,tap,0]=2+tap*.03+np.sin(times*.03+prn*.2+phase)*.05
  iq[:,tap,1]=.2+tap*.01+np.cos(times*.02+prn*.1+phase)*.02
 np.savez(path,complex_iq=iq,prn=prn,time_s=times,segment=np.zeros(n,int),channel=prn-1)

def run(script,*args,expect=0):
 command=[sys.executable,str(ROOT/script),*map(str,args)]
 result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,env={**os.environ,"MPLBACKEND":"Agg"})
 assert result.returncode==expect, f"command={command!r}\nstdout={result.stdout}\nstderr={result.stderr}"
 return result

def test_true_subprocess_synthetic_prep_train_dev_history_attest_freeze_confirm_finalize(tmp_path):
 # No monkeypatching and no TEXBAT bytes: every campaign stage crosses a real process boundary.
 source=tmp_path/"source"; source.mkdir(); names=("cleanStatic","DS1","DS2","DS3","DS7","DS8")
 for i,name in enumerate(names): make_npz(source/f"{name}.npz",365 if name=="cleanStatic" else 190,i*.3)
 uniform=tmp_path/"uniform"; prep_args=[]
 for name in names: prep_args += ["--scenario-npz",f"{name}={source/f'{name}.npz'}"]
 run("scripts/prepare_cmte_a2_inputs.py",*prep_args,"--onset-seconds","110","--out",uniform)
 ds4=tmp_path/"ds4"; run("scripts/prepare_cmte_a2_inputs.py","--ds4-node",uniform/"DS3_nodes.csv","--ds4-manifest",uniform/"DS3_manifest.json","--out",ds4)
 state=tmp_path/"state"; run("scripts/train_cmte_a2_texbat.py","--clean-node-csv",uniform/"cleanStatic_nodes.csv","--clean-manifest",uniform/"cleanStatic_manifest.json","--out",state)
 development=tmp_path/"development"; devargs=["--tier","development","--state-dir",state,"--out",development]
 for name,root in (("DS1",uniform),("DS2",uniform),("DS3",uniform),("DS4",ds4)):
  devargs += ["--scenario",f"{name}={root/f'{name}_nodes.csv'}={root/f'{name}_manifest.json'}"]
 run("scripts/eval_cmte_a2_texbat.py",*devargs)
 history=tmp_path/"historical.json"; run("scripts/check_cmte_a2_historical_b0.py","--output",history)
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 attestation=tmp_path/"preflight.json"; run("scripts/run_cmte_a2_preflight_tests.py","--source-commit",commit,"--out",attestation)
 attest=json.loads(attestation.read_text()); assert attest["exit_code"]==0 and attest["summary"]["failed"]==0 and attest["summary"]["passed"]>=30
 confirm_inputs=tmp_path/"confirm-inputs.json"
 run("tests/fixtures/build_cmte_a2_synthetic_confirm_manifest.py","--source",source,"--uniform",uniform,"--out",confirm_inputs)
 frozen=tmp_path/"freeze"; run("scripts/freeze_cmte_a2_campaign.py","--state-dir",state,"--development-dir",development,
  "--confirm-input-manifest",confirm_inputs,"--test-attestation",attestation,"--out",frozen)
 anchor=frozen/"freeze_manifest.json"; anchor_sha=hashlib.sha256(anchor.read_bytes()).hexdigest()
 rejected=run("scripts/eval_cmte_a2_texbat.py","--tier","confirmatory","--state-dir",tmp_path/"never-opened",
  "--scenario",f"DS7={tmp_path/'holdout'}={tmp_path/'manifest'}","--out",tmp_path/"rejected",expect=2)
 assert "invalid choice" in rejected.stderr and not (tmp_path/"rejected").exists()
 confirmatory=tmp_path/"confirmatory"; ledger=tmp_path/"one-shot-ledger.json"
 run("scripts/confirm_cmte_a2_texbat.py","--trust-anchor",anchor,"--expected-sha256",anchor_sha,"--ledger",ledger,"--out",confirmatory)
 final=tmp_path/"final"; run("scripts/finalize_cmte_a2_campaign.py","--state-dir",state,"--development-dir",development,
  "--confirmatory-dir",confirmatory,"--trust-anchor",anchor,"--expected-anchor-sha256",anchor_sha,"--ledger",ledger,"--out",final)
 for required in ("README.md","scenario_metrics.csv","confirmatory_metrics.csv","development_metrics.csv","baseline_metrics.csv",
  "bootstrap_cis.csv","exact_n_diagnostics.csv","matched_fpr.csv","provenance.json","checksums.json",
  "provenance/trust_anchor.json","provenance/confirm_input_manifest.json","provenance/one_shot_ledger.json",
  "provenance/development.json","provenance/confirmatory.json","provenance/hash_inventory.json","provenance/test_attestation.json"):
  assert (final/required).stat().st_size>0
 checks=json.loads((final/"checksums.json").read_text()); assert all(required in checks for required in ("provenance/trust_anchor.json","provenance/hash_inventory.json","provenance/test_attestation.json"))
 inventory=json.loads((final/"provenance/hash_inventory.json").read_text()); assert set(inventory["producer_hashes"])=={"converter","wrapper","receiver","exporter","template"}
 assert {"scaler_sha256","qcal_sha256","checkpoint_sha256","state_sha256","thresholds_sha256"}.issubset(inventory["frozen_state"])
 for scenario in ("DS1","DS2","DS3","DS4","DS7","DS8"):
  assert (final/"per_epoch"/f"{scenario}.csv").stat().st_size>0 and (final/"per_prn"/f"{scenario}.csv").stat().st_size>0
