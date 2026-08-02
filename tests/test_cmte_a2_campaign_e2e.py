from __future__ import annotations
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def load(name):
 path=ROOT/"scripts"/name; module_name="e2e_"+name.replace(".","_")
 spec=importlib.util.spec_from_file_location(module_name,path); assert spec and spec.loader
 mod=importlib.util.module_from_spec(spec); sys.modules[module_name]=mod; spec.loader.exec_module(mod); return mod


def make_npz(path:Path,end:float,phase:float):
 base=np.arange(0,end+.001,.5); times=np.tile(base,5); prn=np.repeat(np.arange(1,6),len(base)); n=len(times)
 iq=np.zeros((n,9,2),np.float32)
 for tap in range(9):
  iq[:,tap,0]=2+tap*.03+np.sin(times*.03+prn*.2+phase)*.05
  iq[:,tap,1]=.2+tap*.01+np.cos(times*.02+prn*.1+phase)*.02
 np.savez(path,complex_iq=iq,prn=prn,time_s=times,segment=np.zeros(n,int),channel=prn-1)


def test_synthetic_cli_prep_train_dev_freeze_confirm_finalize(tmp_path,monkeypatch):
 # No TEXBAT path is opened: all six recordings and the receiver/exporter are synthetic fixtures.
 source=tmp_path/"source"; source.mkdir(); names=("cleanStatic","DS1","DS2","DS3","DS7","DS8")
 for i,name in enumerate(names): make_npz(source/f"{name}.npz",365 if name=="cleanStatic" else 190,i*.3)

 # Exercise the DS8 source-only CLI with a fake receiver and fake pinned exporter.
 ds8prep=load("prepare_cmte_a2_ds8_complex.py")
 raw=tmp_path/"fake_ds8.raw"; raw.write_bytes(b"FAKE-DS8")
 receiver=tmp_path/"fake_receiver.py"; receiver.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\nPath('raw/epl_tracking_ch_0.mat').write_bytes(b'MAT')\n") ; receiver.chmod(0o755)
 monkeypatch.setattr(ds8prep,"RAW_BYTES",raw.stat().st_size); monkeypatch.setattr(ds8prep,"RAW_SHA",hashlib.sha256(raw.read_bytes()).hexdigest())
 monkeypatch.setattr(ds8prep,"EXEC_SHA",hashlib.sha256(receiver.read_bytes()).hexdigest())
 monkeypatch.setattr(ds8prep,"export_tracking_csv",lambda mats,csv,summary,**kw: (Path(csv).write_text("x\n1\n"),Path(summary).write_text("x\n1\n"),{"rows":1})[-1])
 monkeypatch.setattr(ds8prep,"parse_acquired_prns",lambda text:[1]); monkeypatch.setattr(ds8prep,"parse_receiver_reported_prns",lambda text:[1])
 def fake_exporter(destination):
  def export(run,out):
   shutil.copyfile(source/"DS8.npz",out); manifest=Path(out).with_suffix(".manifest.json"); manifest.write_text(json.dumps({"fixture":True})); return manifest
  return export
 monkeypatch.setattr(ds8prep,"load_pinned_exporter",fake_exporter)
 prepared_ds8=tmp_path/"ds8-prepared"; ds8prep.main(["--iq",str(raw),"--receiver-executable",str(receiver),"--output-root",str(prepared_ds8)])

 prep=load("prepare_cmte_a2_inputs.py"); uniform=tmp_path/"uniform"
 mappings=[]
 for name in names:
  npz=prepared_ds8/"exports/ds8.npz" if name=="DS8" else source/f"{name}.npz"
  mappings += ["--scenario-npz",f"{name}={npz}"]
 prep.main([*mappings,"--onset-seconds","110","--out",str(uniform)])
 ds4=tmp_path/"ds4"; prep.main(["--ds4-node",str(uniform/"DS3_nodes.csv"),"--ds4-manifest",str(uniform/"DS3_manifest.json"),"--out",str(ds4)])

 train=load("train_cmte_a2_texbat.py"); state=tmp_path/"state"
 train.main(["--clean-node-csv",str(uniform/"cleanStatic_nodes.csv"),"--clean-manifest",str(uniform/"cleanStatic_manifest.json"),"--out",str(state)])

 evaluator=load("eval_cmte_a2_texbat.py"); original_boot=evaluator.bootstrap_metrics
 monkeypatch.setattr(evaluator,"bootstrap_metrics",lambda frame,reps=2000,seed=20260802:original_boot(frame,reps=5,seed=seed))
 development=tmp_path/"development"
 devargs=["--tier","development","--state-dir",str(state),"--out",str(development)]
 for name,root in (("DS1",uniform),("DS2",uniform),("DS3",uniform),("DS4",ds4)):
  devargs += ["--scenario",f"{name}={root/f'{name}_nodes.csv'}={root/f'{name}_manifest.json'}"]
 evaluator.main(devargs)

 from gnss_doppler_lab.cmte_a2_campaign import (CONVERTER_SHA,EXPORTER_SHA,RECEIVER_SHA,build_confirm_input_manifest,file_sha256)
 prepdoc=json.loads((prepared_ds8/"manifest.json").read_text()); prepdoc["binary_sha256"]=RECEIVER_SHA
 (prepared_ds8/"synthetic_prep_manifest.json").write_text(json.dumps(prepdoc))
 confirm_inputs=tmp_path/"confirm-inputs.json"
 build_confirm_input_manifest(confirm_inputs,ds7_raw=source/"DS7.npz",ds7_npz=source/"DS7.npz",ds7_node=uniform/"DS7_nodes.csv",
  ds7_input_manifest=uniform/"DS7_manifest.json",ds8_raw=raw,ds8_npz=prepared_ds8/"exports/ds8.npz",
  ds8_rendered_config=prepared_ds8/"receiver/receiver.conf",ds8_prep_manifest=prepared_ds8/"synthetic_prep_manifest.json",
  ds8_node=uniform/"DS8_nodes.csv",ds8_input_manifest=uniform/"DS8_manifest.json",expected_ds7_raw_sha=file_sha256(source/"DS7.npz"),
  expected_ds8_raw_sha=file_sha256(raw),code_hashes={"converter":CONVERTER_SHA,
  "wrapper":file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"),"receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,
  "template":file_sha256(ROOT/"configs/cmte_a2_ds8_receiver.conf")})

 freeze=load("freeze_cmte_a2_campaign.py"); frozen=tmp_path/"freeze"
 freeze.main(["--state-dir",str(state),"--development-dir",str(development),"--confirm-input-manifest",str(confirm_inputs),"--out",str(frozen)])
 anchor=frozen/"freeze_manifest.json"; anchor_sha=file_sha256(anchor)
 confirm=load("confirm_cmte_a2_texbat.py"); monkeypatch.setattr(confirm,"_load_guarded_evaluator",lambda:evaluator)
 confirmatory=tmp_path/"confirmatory"; ledger=tmp_path/"one-shot-ledger.json"
 confirm.main(["--trust-anchor",str(anchor),"--expected-sha256",anchor_sha,"--ledger",str(ledger),"--out",str(confirmatory)])

 finalizer=load("finalize_cmte_a2_campaign.py"); final=tmp_path/"final"
 finalizer.main(["--state-dir",str(state),"--development-dir",str(development),"--confirmatory-dir",str(confirmatory),
  "--trust-anchor",str(anchor),"--expected-anchor-sha256",anchor_sha,"--ledger",str(ledger),"--out",str(final)])
 for required in ("README.md","preregistration.json","scenario_metrics.csv","confirmatory_metrics.csv","development_metrics.csv",
                  "baseline_metrics.csv","bootstrap_cis.csv","exact_n_diagnostics.csv","matched_fpr.csv","provenance.json","checksums.json"):
  assert (final/required).stat().st_size>0
 for scenario in ("DS1","DS2","DS3","DS4","DS7","DS8"):
  assert (final/"per_epoch"/f"{scenario}.csv").stat().st_size>0
  assert (final/"per_prn"/f"{scenario}.csv").stat().st_size>0
 assert list((final/"plots").rglob("*.png"))
 assert not (development/"confirmatory_metrics.csv").exists() and not (confirmatory/"development_metrics.csv").exists()
