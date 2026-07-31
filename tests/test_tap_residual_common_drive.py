import json
import numpy as np
import pandas as pd
import pytest
import torch
from gnss_doppler_lab.tap_residual_common_drive import INNOVATION_PREFIX,TIMING_CONTRACT,calibrate_clean_only,causal_smooth_events,extract_b0_innovations,score_common_drive

class Zero(torch.nn.Module):
 def forward(self,x): return torch.zeros((x.shape[0],x.shape[2]),dtype=x.dtype,device=x.device)

def inputs():
 rows=[]
 for pi,p in enumerate(["G02","G01"]):
  for t in range(4): rows.append(dict(run_id="r1",prn=p,window_bin_s=float(t)/2,window_start_s=float(t)/2-.5,window_end_s=float(t)/2+.5,window_mid_s=float(t)/2,f0=10*pi+t,f1=100+10*pi+t))
 return pd.DataFrame(rows)

def innovations(vs,prns=None,time=1.,run="r1"):
 prns=prns or [f"G{i+1:02d}" for i in range(len(vs))]; rows=[]
 for p,v in zip(prns,vs):
  r=dict(run_id=run,prn=p,window_bin_s=time,window_start_s=time-.5,window_end_s=time+.5,window_mid_s=time,b0_prn_node_rmse=float(np.sqrt(np.mean(np.square(v)))))
  r.update({f"{INNOVATION_PREFIX}{i}":float(x) for i,x in enumerate(v)}); rows.append(r)
 return pd.DataFrame(rows)

def test_vector_extraction_shape_ordering_and_b0_unchanged():
 o=extract_b0_innovations(inputs(),Zero(),["f0","f1"],np.zeros(2),np.ones(2),seq_len=2,device="cpu")
 assert list(zip(o.prn,o.window_bin_s))==[("G01",1.),("G01",1.5),("G02",1.),("G02",1.5)]
 assert o[["innovation_0","innovation_1"]].shape==(4,2)
 assert o.loc[0,["innovation_0","innovation_1"]].tolist()==[12.,112.]
 assert o.loc[0,"b0_prn_node_rmse"]==pytest.approx(np.sqrt(np.mean(np.square([12.,112.]))))
 assert o.loc[0,"target_window_index"]==2

def test_leave_one_out_self_exclusion():
 n,_=score_common_drive(innovations([[100.,0.],[0.,2.],[0.,2.]])); r=n.set_index("prn").loc["G01"]
 assert [r.loo_common_0,r.loo_common_1]==[0.,2.]; assert r.positive_alignment==0.

def test_permutation_invariance():
 f=innovations([[1.,2.],[2.,1.],[1.5,1.5]],["G03","G01","G02"]); na,ea=score_common_drive(f); nb,eb=score_common_drive(f.sample(frac=1,random_state=7)); cols=["prn","local_support","common_drive_support","joint_evidence"]
 pd.testing.assert_frame_equal(na[cols].sort_values("prn").reset_index(drop=True),nb[cols].sort_values("prn").reset_index(drop=True)); pd.testing.assert_frame_equal(ea,eb)

def test_duplicate_rejection_and_variable_count_mask():
 with pytest.raises(ValueError,match="duplicate PRN"): score_common_drive(innovations([[1.],[2.]],["G01","G01"]))
 f=pd.concat([innovations([[1.,0.]],time=1.),innovations([[1.,0.],[1.,0.]],time=2.)],ignore_index=True); n,e=score_common_drive(f)
 assert e.tracked_prn_count.tolist()==[1,2] and e.relation_eligible.tolist()==[False,True]
 r=n[n.window_bin_s==1.].iloc[0]; assert not r.relation_eligible and r.common_drive_support==0. and r.joint_evidence==0.

def test_no_future_influence_and_run_reset():
 f=pd.DataFrame({"run_id":["a","a","a","b"],"window_bin_s":[0.,1.,2.,0.],"event_joint_evidence":[1.,2.,3.,10.]}); g=f.copy(); g.loc[2,"event_joint_evidence"]=999.; a=causal_smooth_events(f,.5); b=causal_smooth_events(g,.5)
 assert a.loc[:1,"event_joint_evidence_causal"].tolist()==b.loc[:1,"event_joint_evidence_causal"].tolist(); assert a.loc[3,"event_joint_evidence_causal"]==10.; assert a.loc[1,"event_joint_evidence_causal"]==pytest.approx(1.5)

def test_joint_requires_magnitude_and_positive_alignment():
 _,a=score_common_drive(innovations([[4.,0.],[4.,0.]])); _,o=score_common_drive(innovations([[4.,0.],[-4.,0.]])); _,t=score_common_drive(innovations([[.01,0.],[.01,0.]]))
 assert a.event_joint_evidence.iloc[0]>1 and o.event_joint_evidence.iloc[0]==0 and t.event_joint_evidence.iloc[0]<.02
 assert a.event_local_support.iloc[0]==pytest.approx(np.sqrt(8.)) and a.event_common_drive_support.iloc[0]==pytest.approx(1.)

def test_finite_edges():
 with pytest.raises(ValueError,match="finite"): score_common_drive(innovations([[np.nan],[1.]]))
 n,e=score_common_drive(innovations([[0.,0.],[0.,0.]])); assert np.isfinite(n.select_dtypes(include=np.number)).all().all() and np.isfinite(e.select_dtypes(include=np.number)).all().all()

def test_clean_only_calibration_provenance_no_overwrite(tmp_path):
 e=pd.DataFrame({"run_id":["c"]*3,"window_bin_s":[0.,1.,2.],"event_local_support":[1.,2.,3.],"event_common_drive_support":[.1,.2,.3],"event_joint_evidence":[.1,.4,.9],"event_joint_evidence_causal":[.1,.25,.575]}); p=tmp_path/"c.json"
 d=calibrate_clean_only(e,p,source_kind="cleanStatic",source_paths=["clean.csv"],checkpoint_sha256="abc",quantiles=(.5,.9)); assert d["provenance"]["gate_source"]=="cleanStatic_only" and not d["provenance"]["attack_labels_used"] and not d["provenance"]["attack_prefix_fitted"]
 before=p.read_bytes()
 with pytest.raises(FileExistsError): calibrate_clean_only(e,p,source_kind="cleanStatic",source_paths=["x"],checkpoint_sha256="x")
 assert p.read_bytes()==before and json.loads(p.read_text())["thresholds"]["event_joint_evidence_causal"]["q0.5"]==.25
 with pytest.raises(ValueError,match="cleanStatic"): calibrate_clean_only(e,tmp_path/"x",source_kind="ds4",source_paths=["a"],checkpoint_sha256="x")

def test_exact_timing_contract():
 assert TIMING_CONTRACT=={"score_time_field":"window_start_s","availability_time_field":"window_end_s","availability_offset":"window_end_s - window_start_s","smoothing":"strictly causal, current-and-past only, reset per run","event_time_fields":"window_start_s=min start and window_end_s=max end form the same-bin availability envelope; window_mid_s=window_bin_s representative"}


def test_parallel_scorer_script_exists_and_does_not_modify_b0():
 import ast
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/"scripts"/"score_tap_residual_common_drive.py"
 tree=ast.parse(p.read_text())
 names={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert {"load_frozen_b0","score_node_csv","main"} <= names
 assert TIMING_CONTRACT["availability_offset"] == "window_end_s - window_start_s"


def _scorer_module():
 import importlib.util
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/"scripts"/"score_tap_residual_common_drive.py"; spec=importlib.util.spec_from_file_location("trcd_score_test",p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def metadata_frame(label="texbat_cleanStatic_9tap_w1.0_s0.5",run_id="texbat-cleanStatic-method-a-9tap-external-validation",fingerprint="fingerprint-a"):
 f=inputs(); f["label"]=label; f["source_fingerprint"]=fingerprint; f["tap_count"]=9; f["tap_layout"]="E4,E3,E2,E,P,L,L2,L3,L4"; f["run_id"]=run_id; return f

def test_public_score_rejects_null_and_empty_identifiers():
 for col,bad in [("run_id",None),("run_id",""),("run_id","  "),("prn",None),("prn",""),("prn","  ")]:
  f=innovations([[1.],[2.]]); f.loc[0,col]=bad
  with pytest.raises(ValueError,match="run_id and prn"): score_common_drive(f)

def test_sequence_contract_rejects_duplicates_gaps_and_inconsistent_times():
 base=inputs(); duplicate=pd.concat([base,base.iloc[[0]]],ignore_index=True)
 with pytest.raises(ValueError,match="duplicate.*run_id.*prn.*window_bin_s"): extract_b0_innovations(duplicate,Zero(),["f0","f1"],[0,0],[1,1],seq_len=2)
 gap=base.copy(); gap.loc[(gap.prn=="G01") & (gap.window_bin_s==1.5),"window_bin_s"]=2.0
 with pytest.raises(ValueError,match="0.5.*cadence|gap"): extract_b0_innovations(gap,Zero(),["f0","f1"],[0,0],[1,1],seq_len=2)
 bad=base.copy(); bad.loc[0,"window_end_s"]=bad.loc[0,"window_start_s"]-.1
 with pytest.raises(ValueError,match="timing"): extract_b0_innovations(bad,Zero(),["f0","f1"],[0,0],[1,1],seq_len=2)
 event=innovations([[1.],[2.]]); event.loc[1,"window_end_s"]+=.25
 with pytest.raises(ValueError,match="inconsistent.*timing"): score_common_drive(event)

def test_clean_metadata_verified_and_fingerprint_bound():
 m=_scorer_module(); clean=metadata_frame(); assert m.validate_node_csv_metadata(clean,require_cleanstatic=True)["source_fingerprint"]=="fingerprint-a"
 attacks=metadata_frame(label="texbat_ds4_9tap_w1.0_s0.5",run_id="texbat-ds4-method-a-9tap-external-validation")
 with pytest.raises(ValueError,match="cleanStatic identity"): m.validate_node_csv_metadata(attacks,require_cleanstatic=True)
 for col in ["label","run_id","source_fingerprint","tap_count","tap_layout"]:
  damaged=clean.copy(); damaged.loc[0,col]=None
  with pytest.raises(ValueError,match=col): m.validate_node_csv_metadata(damaged,require_cleanstatic=True)
 multi=clean.copy(); multi.loc[0,"source_fingerprint"]="other"
 with pytest.raises(ValueError,match="single-valued"): m.validate_node_csv_metadata(multi,require_cleanstatic=True)

def test_exact_frozen_feature_and_semantic_checkpoint_contract(monkeypatch,tmp_path):
 m=_scorer_module(); assert m.FROZEN_TAP_FEATURES==[f"tap_{x}_rel_prompt_mean" for x in ["E4","E3","E2","E","P","L","L2","L3","L4"]]
 class Config: seq_len=12; feature_subset="wrong"
 monkeypatch.setattr(m,"_train_module",lambda:type("T",(),{"TrainConfig":lambda **kw:Config(),"PrnLocalGRU":lambda *a:Zero()})); cp={"model_state_dict":{},"config":{"seq_len":12},"node_feature_columns":m.FROZEN_TAP_FEATURES,"standardizer":{"node_mean":[0]*9,"node_std":[1]*9}}; monkeypatch.setattr(m.torch,"load",lambda *a,**k:cp); p=tmp_path/"b0.pt"; p.write_bytes(b"known")
 with pytest.raises(ValueError,match="feature_subset"): m.load_frozen_b0(p,"cpu",expected_sha256=__import__("hashlib").sha256(b"known").hexdigest())
 with pytest.raises(ValueError,match="SHA-256"): m.load_frozen_b0(p,"cpu",expected_sha256="0"*64)

def test_scenario_slug_and_output_preflight(tmp_path):
 m=_scorer_module()
 for slug in ["../escape","a/b","cleanStatic",".","", "a\\b"]:
  with pytest.raises(ValueError,match="scenario"): m._validated_scenario_slug(slug)
 out=tmp_path/"out"; out.mkdir(); (out/"summary.json").write_text("sentinel")
 with pytest.raises(FileExistsError): m._preflight_output_paths(out,"ds4",include_score=True,overwrite=False)
 assert not (tmp_path/"escape_node_scores.csv").exists()

def test_atomic_writes_and_pairwise_no_overwrite(monkeypatch,tmp_path):
 m=_scorer_module(); calls=[]; real=m.os.replace; monkeypatch.setattr(m.os,"replace",lambda a,b:(calls.append((a,b)),real(a,b))[1]); n=innovations([[1.],[2.]])
 m._write_scores(tmp_path,"ds4",n,n,overwrite=False); assert len(calls)==0 and not list(tmp_path.glob("*.tmp")); before=(tmp_path/"ds4_node_scores.csv").read_bytes()
 with pytest.raises(FileExistsError): m._write_scores(tmp_path,"ds4",n.iloc[:1],n,overwrite=False)
 assert (tmp_path/"ds4_node_scores.csv").read_bytes()==before

def test_calibration_atomic_replace_and_fingerprint_provenance(monkeypatch,tmp_path):
 import gnss_doppler_lab.tap_residual_common_drive as tr
 calls=[]; real=tr.os.replace; monkeypatch.setattr(tr.os,"replace",lambda a,b:(calls.append((a,b)),real(a,b))[1]); e=pd.DataFrame({"run_id":["c"]*2,"event_joint_evidence":[1.,2.],"event_joint_evidence_causal":[1.,1.5]}); p=tmp_path/"calibration.json"; d=calibrate_clean_only(e,p,source_kind="cleanStatic",source_paths=["c.csv"],source_fingerprint="fp",checkpoint_sha256="abc")
 assert len(calls)==0 and d["provenance"]["source_fingerprint"]=="fp" and not list(tmp_path.glob("*.tmp"))
 calibrate_clean_only(e,p,source_kind="cleanStatic",source_paths=["c.csv"],source_fingerprint="fp",checkpoint_sha256="abc",overwrite=True)
 assert len(calls)==1 and not list(tmp_path.glob("*.tmp"))

def test_onset_windows_crossing_excluded_and_first_flag_times_same_row():
 m=_scorer_module(); e=pd.DataFrame({"run_id":["a","b","a","b"],"window_start_s":[9.,9.,10.,10.],"window_end_s":[10.,10.5,11.,12.],"event_joint_evidence_causal":[0.,2.,2.,2.]}); r=m._gate_metrics(e,1.,10.)
 assert (r["pre_windows"],r["crossing_windows"],r["post_windows"])==(1,1,2); assert r["first_post_flag_score_time_s"]==10. and r["first_post_flag_available_time_s"]==11.; assert r["score_time_delay_s"]==0. and r["availability_time_delay_s"]==1.


def test_atomic_no_clobber_wins_race_after_temp_is_complete(monkeypatch, tmp_path):
 import gnss_doppler_lab.tap_residual_common_drive as tr
 e=pd.DataFrame({"run_id":["c"],"event_joint_evidence":[1.]})
 target=tmp_path/"calibration.json"; real_link=tr.os.link; raced=False
 def competing_publish(source, destination):
  nonlocal raced
  if not raced:
   raced=True; target.write_text("competitor")
  return real_link(source,destination)
 monkeypatch.setattr(tr.os,"link",competing_publish)
 with pytest.raises(FileExistsError):
  calibrate_clean_only(e,target,source_kind="cleanStatic",source_paths=["c.csv"],checkpoint_sha256="abc")
 assert target.read_text()=="competitor" and not list(tmp_path.glob("*.tmp"))

def test_csv_and_json_atomic_helpers_no_clobber_under_race(monkeypatch,tmp_path):
 m=_scorer_module(); target=tmp_path/"summary.json"; real_link=m.os.link
 def competing_publish(source,destination):
  target.write_text("competitor")
  return real_link(source,destination)
 monkeypatch.setattr(m.os,"link",competing_publish)
 with pytest.raises(FileExistsError): m._atomic_json(target,{"new":True},overwrite=False)
 assert target.read_text()=="competitor" and not list(tmp_path.glob("*.tmp"))

def test_csv_atomic_helper_no_clobber_under_race(monkeypatch,tmp_path):
 m=_scorer_module(); target=tmp_path/"ds4_node_scores.csv"; real_link=m.os.link; raced=False
 def competing_publish(source,destination):
  nonlocal raced
  if not raced:
   raced=True; target.write_text("competitor")
  return real_link(source,destination)
 monkeypatch.setattr(m.os,"link",competing_publish)
 with pytest.raises(FileExistsError): m._write_scores(tmp_path,"ds4",innovations([[1.]]),innovations([[1.]]),overwrite=False)
 assert target.read_text()=="competitor" and not list(tmp_path.glob("*.tmp"))


def test_same_event_rejects_remote_actual_time_but_allows_real_offsets():
 bad=innovations([[1.],[2.]],time=1.5)
 bad.loc[1,["window_start_s","window_end_s","window_mid_s"]]=[101.,102.,101.5]
 with pytest.raises(ValueError,match="same-event|window_bin"): score_common_drive(bad)
 allowed=innovations([[1.],[2.]],time=1.5)
 allowed.loc[0,["window_start_s","window_end_s","window_mid_s"]]=[.81169144,1.81169144,1.31169144]
 allowed.loc[1,["window_start_s","window_end_s","window_mid_s"]]=[1.18830856,2.18830856,1.68830856]
 _,events=score_common_drive(allowed)
 assert len(events)==1 and events.loc[0,"window_mid_s"]==pytest.approx(1.5)


def test_atomic_writer_never_reopens_mkstemp_path(monkeypatch, tmp_path):
 m=_scorer_module(); victim=tmp_path/"victim.txt"; victim.write_text("SENTINEL")
 target=tmp_path/"summary.json"; created={}; real_mkstemp=m.tempfile.mkstemp; real_close=m.os.close
 def tracked_mkstemp(*args,**kwargs):
  fd,path=real_mkstemp(*args,**kwargs); created["path"]=path; return fd,path
 def swap_after_close(fd):
  real_close(fd); path=created["path"]; m.os.unlink(path); m.os.symlink(victim,path)
 monkeypatch.setattr(m.tempfile,"mkstemp",tracked_mkstemp); monkeypatch.setattr(m.os,"close",swap_after_close)
 m._atomic_json(target,{"safe":True},overwrite=False)
 assert victim.read_text()=="SENTINEL"
 assert not target.is_symlink()
 assert json.loads(target.read_text())=={"safe":True}
