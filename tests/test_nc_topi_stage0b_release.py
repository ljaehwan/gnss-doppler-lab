from __future__ import annotations
import ast, importlib.util, json, multiprocessing as mp
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab import nc_topi_stage0b as a
ROOT=Path(__file__).resolve().parents[1]
VERIFY=ROOT/"scripts/summarize_nc_topi_stage0b_audit.py"
def verifier():
 s=importlib.util.spec_from_file_location("standalone_stage0b_verifier",VERIFY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def meta(n,role="normal_train",scenario="cleanStatic",label="0",valid="True"):
 return [{"identity":f"{role}-row-{i}","scenario":scenario,"role":role,"phase":"normal","label":label,"valid":valid} for i in range(n)]

def test_verifier_import_graph_is_standalone_and_bug_divergent(monkeypatch):
 tree=ast.parse(VERIFY.read_text());imports=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Import): imports += [x.name for x in node.names]
  elif isinstance(node,ast.ImportFrom): imports += [node.module or ""]
 assert not any("audit_nc_topi_shortcut" in x or "nc_topi_stage0b" in x or x=="gnss_doppler_lab" for x in imports)
 v=verifier();monkeypatch.setattr(a,"higher_quantile",lambda *_a,**_k:-999.)
 assert v.higher_quantile([0.,1.,2.,3.],.5)==2.

def test_conditioner_requires_row_provenance_and_clean_state_seal():
 x=np.arange(48.,dtype=float).reshape(12,4);y=np.arange(12.)+1
 model=a.TargetConditioner.fit("TOPI",x[:8],y[:8],meta(8))
 assert model.audit["attack_fit"] is False
 assert model.audit["fit_predicate"]=="cleanStatic/normal_train/normal/label0/valid"
 assert model.clean_state_digest
 for bad in (meta(8,scenario="DS7",label="1"),meta(8,valid="False")):
  with pytest.raises(ValueError,match="fit provenance"):a.TargetConditioner.fit("TOPI",x[:8],y[:8],bad)
 model.calibration_bounds(x[8:],meta(4,role="normal_calibration"))
 with pytest.raises(ValueError,match="calibration provenance"):
  model.calibration_bounds(x[8:],meta(4,role="normal_calibration",scenario="DS7"))
 seal=model.clean_state_digest;model.predict_scale(x[8:]);assert model.clean_state_digest==seal

def test_effective_scale_both_zero_uses_two_denominators():
 assert a.check_effective_scale([2.],[.5],[4.],[4.])["ordinary_rows"]==1
 with pytest.raises(ValueError,match="zero division"):a.check_effective_scale([2.],[0.],[4.],[4.])
 assert a.check_effective_scale([0.],[0.],[1e-12],[1e-12])["both_zero_rows"]==1
 with pytest.raises(ValueError,match="both-zero denominator mismatch"):a.check_effective_scale([0.],[0.],[999.],[1e-12])

def test_profile_d_union_support_callback_and_actual_33():
 iq={"event_id":"e","block_start_s":"-1.0;-0.5","block_end_s":"-0.5;0.0","history_blocks":"2","cadence_seconds":"0.5","target_source_start_s":"4.9"}
 event={"event_id":"e","source_start_s":"5.0","source_end_s":"6.0"};prns=[{"event_id":"e","source_start_s":"4.9","source_end_s":"6.1"}]
 assert a.profile_d_effective_support(event,prns,iq,b0_windows=12,cadence=.5)=={"event_id":"e","effective_start_s":-1.,"effective_end_s":6.1}
 times=[*range(50),*range(70,171),*range(190,240)];events=[{"event_id":str(i),"effective_start_s":float(t),"effective_end_s":float(t)+.5} for i,t in enumerate(times)]
 got=a.check_profile_d_support(events,fit_callback=lambda *_:{"fit":{},"clamp":{},"threshold":{},"holdout":{}})
 assert got["status"]=="AVAILABLE" and set(got["available_evidence"])=={"fit","clamp","threshold","holdout"}
 with pytest.raises(ValueError,match="available callback"):a.check_profile_d_support(events,fit_callback=lambda *_:None)
 data=a.load_parent_evidence(ROOT/"artifacts/nc_topi_stage0",verify_binding=False)
 got=a.check_profile_d_support(a.profile_d_support_from_parent(data))
 assert got["status"]=="INSUFFICIENT_NORMAL_SUPPORT" and got["best_counts"]=={"normal_train":33,"normal_calibration":33,"normal_holdout":33}

def test_strict_json_malformed_hash_and_exact_inventory(tmp_path):
 v=verifier();p=tmp_path/"bad.json";p.write_text('{"x":NaN}')
 with pytest.raises(ValueError,match="non-finite"):v.read_json_strict(p)
 root=tmp_path/"artifact";root.mkdir();(root/"hashes.json").write_text('{"files":3}')
 report=v.verify_hashes(root);assert not report["ok"] and "AttributeError" not in " ".join(report["errors"])
 config={"artifact_contract":{"required_root_regular_files_exact":["config.json","hashes.json"],"required_diagnostics_regular_files_exact":[],"required_plots_regular_files_exact":[]}}
 (root/"config.json").write_text(json.dumps(config));(root/"surprise.txt").write_text("x");v.write_hash_manifest(root)
 assert any("unexpected" in e for e in v.verify_exact_inventory(root,config))

def race(final,payload,ready,start,out):
 try:
  with a.ArtifactStage(final) as s:
   (s.path/"payload").write_text(payload);ready.put(1);start.wait(10);s.publish(lambda _:{"ok":True,"errors":[]})
  out.put((payload,"published"))
 except BaseException as e:out.put((payload,type(e).__name__))
def test_artifact_stage_atomic_noreplace_race(tmp_path):
 final=tmp_path/"out";ready=mp.Queue();out=mp.Queue();start=mp.Event();ps=[mp.Process(target=race,args=(final,x,ready,start,out)) for x in ("one","two")]
 for p in ps:p.start()
 ready.get(timeout=10);ready.get(timeout=10);start.set()
 for p in ps:p.join(10);assert p.exitcode==0
 results=[out.get(timeout=5),out.get(timeout=5)];assert sorted(x[1] for x in results)==["FileExistsError","published"]
 winner=next(x[0] for x in results if x[1]=="published");assert (final/"payload").read_text()==winner

def test_bootstrap_2000_and_exact_comparison_inventory():
 labels=np.r_[np.zeros(40),np.ones(40)];times=np.r_[np.arange(40)*.5,100+np.arange(40)*.5];rec=np.array(["normal"]*40+["attack"]*40)
 aa=labels+np.linspace(0,.01,80);bb=.5*labels+np.linspace(0,.01,80)
 one=a.paired_block_bootstrap(labels,aa,bb,rec,times,reps=2000);two=a.paired_block_bootstrap(labels,aa,bb,rec,times,reps=2000)
 assert one["available"] and one["valid_reps"]==2000 and one["replicate_digest_sha256"]==two["replicate_digest_sha256"]
 rows=[{"scenario":s,"comparator":c,**one} for s in ("DS7","DS8") for c in a.COMPARATORS]
 assert a.validate_comparison_inventory(rows)==[];rows.pop();assert a.validate_comparison_inventory(rows)

def test_semantic_verification_report_has_no_pending_success():
 v=verifier();r=v.semantic_verification_report([],source_digests={"runner":"a"*64,"verifier":"b"*64})
 assert r["ok"] and r["status"]=="VERIFIED" and r["errors"]==[] and r["checks"]
 assert "pending" not in json.dumps(r).lower()


def test_runner_exposes_explicit_synthetic_subprocess_mode_only_by_flag():
 import importlib.util
 path=ROOT/"scripts/audit_nc_topi_shortcut.py"
 spec=importlib.util.spec_from_file_location("stage0b_runner_cli",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 args=module.parse_args(["--synthetic-fixture-mode","--out","/tmp/stage0b-tiny"])
 assert args.synthetic_fixture_mode is True
