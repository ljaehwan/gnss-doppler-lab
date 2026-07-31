import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest
ROOT=Path(__file__).resolve().parents[1]; SPEC=importlib.util.spec_from_file_location("eval_oakbat_trcd",ROOT/"scripts"/"eval_oakbat_trcd.py"); MOD=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=MOD; SPEC.loader.exec_module(MOD)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ef(v=(1.,2.,3.,4.,5.)):
 n=len(v); return pd.DataFrame({"run_id":["r"]*n,"window_start_s":[0.,109.5,110.,129.5,130.][:n],"window_end_s":[1.,110.5,111.,130.5,131.][:n],"event_local_support":v,"event_common_drive_support":np.asarray(v)/10,"event_joint_evidence":np.asarray(v)/20,"event_joint_evidence_causal":np.asarray(v)/25})
def test_hash_manifest_rejection(tmp_path):
 p=tmp_path/"model.pt";p.write_bytes(b"x");m={"artifacts":{"model.pt":sha(p),"split_manifest.json":"0"*64}}
 with pytest.raises(ValueError,match="missing.*split_manifest"):MOD.verify_hash_roster(tmp_path,m)
 m["artifacts"].pop("split_manifest.json");p.write_bytes(b"y")
 with pytest.raises(ValueError,match="hash mismatch"):MOD.verify_hash_roster(tmp_path,m)
def test_campaign_semantics_fail_closed():
 m={"complete":True,"normal_only":True,"attack_inputs_read":False};MOD.verify_campaign_semantics(m)
 for k,v in (("complete",False),("normal_only",False),("attack_inputs_read",True)):
  with pytest.raises(ValueError):MOD.verify_campaign_semantics(dict(m,**{k:v}))
 with pytest.raises(ValueError):MOD.verify_campaign_semantics({"complete":True})
def test_partition_role_leakage():
 split={"clock":"window_start_s","seq_len":12,"history_contract":"each partition forms sequences independently; no history crosses boundaries","boundaries":MOD.EXPECTED_BOUNDARIES,"partition_csvs":{k:{} for k in MOD.EXPECTED_BOUNDARIES}}
 fs={"train":pd.DataFrame({"window_start_s":[0.,239.5]}),"validation":pd.DataFrame({"window_start_s":[250.,329.5]}),"calibration":pd.DataFrame({"window_start_s":[340.,409.5]}),"held_clean":pd.DataFrame({"window_start_s":[420.,479.]})};MOD.verify_partition_roles(split,fs);fs["calibration"].loc[0,"window_start_s"]=250.
 with pytest.raises(ValueError,match="calibration.*boundary"):MOD.verify_partition_roles(split,fs)
 with pytest.raises(ValueError,match="role"):MOD.calibrate_branches(ef(),partition_role="held_clean")
def test_held_not_used_in_calibration():
 before=MOD.calibrate_branches(ef(),partition_role="calibration");r=MOD.evaluate_branches(ef((1000.,)*5),before,region="held");after=MOD.calibrate_branches(ef(),partition_role="calibration");assert before==after;assert r["branches"]["event_local_support_q99"]["flags"]==5
def test_production_run_persists_calibration_before_attack_reader(monkeypatch,tmp_path):
 calls=[]; root=tmp_path/"campaign"; root.mkdir(); (root/"campaign_manifest.json").write_text("{}"); (root/"split_manifest.json").write_text("{}")
 frames={"calibration":object(),"held_clean":object()}
 monkeypatch.setattr(MOD,"verify_frozen_campaign",lambda p,allow_unpinned_manifests=False:{"root":root,"frames":frames})
 monkeypatch.setattr(MOD,"load_model",lambda *a,**k:object())
 monkeypatch.setattr(MOD,"score_frame",lambda *a,**k:(None,None,ef()))
 monkeypatch.setattr(MOD,"create_unique_output_dir",lambda p:tmp_path/"out")
 monkeypatch.setattr(MOD,"atomic_csv",lambda *a,**k:None)
 def write_json(path,doc): calls.append(("persist",Path(path).name))
 monkeypatch.setattr(MOD,"atomic_json",write_json)
 def attack(path,scenario,allow_unpinned_manifests=False):
  calls.append(("attack_read",scenario));return object(),{},{}
 monkeypatch.setattr(MOD,"authenticate_attack",attack)
 monkeypatch.setattr(MOD,"shuffle_innovations_within_prn",lambda x,seed:x)
 monkeypatch.setattr(MOD,"score_common_drive",lambda x:(None,ef()))
 monkeypatch.setattr(MOD,"causal_smooth_events",lambda x,alpha:x)
 monkeypatch.setattr(MOD,"verify_relation_only_diagnostic",lambda *a,**k:None)
 monkeypatch.setattr(MOD,"sha256",lambda p:MOD.EXPECTED_CAMPAIGN_MANIFEST_SHA256 if Path(p).name=="campaign_manifest.json" else MOD.EXPECTED_SPLIT_MANIFEST_SHA256)
 MOD.run(root,tmp_path,tmp_path)
 persisted=next(i for i,x in enumerate(calls) if x==("persist","calibration.json"))
 assert all(i>persisted for i,x in enumerate(calls) if x[0]=="attack_read")
def test_official_masks():
 m=MOD.official_masks(ef());assert m["pre"].tolist()==[1,1,0,0,0];assert m["guard"].tolist()==[0,0,1,1,0];assert m["post"].tolist()==[0,0,0,0,1]
def test_q95_q99_branch_gates():
 t={"event_local_support":{"q95":2.,"q99":4.},"event_common_drive_support":{"q95":.2,"q99":.4},"event_joint_evidence":{"q95":.1,"q99":.2},"event_joint_evidence_causal":{"q95":.08,"q99":.16}};f=MOD.branch_flags(ef(),t);assert f["local_q95_AND_common_q99"].tolist()==[0,0,0,0,1];assert not f["event_local_support_q95"].iloc[1];assert set(MOD.RELATION_GATES).issubset(f)
def test_no_scenario_fitting_and_finite():
 t=MOD.calibrate_branches(ef(),partition_role="calibration");x=json.dumps(t,sort_keys=True)
 for s in MOD.SCENARIOS:
  r=MOD.scenario_report(ef(),t,scenario=s);assert r["fitting"]=="none";assert r["threshold_source_partition"]=="calibration";assert json.dumps(t,sort_keys=True)==x;MOD.assert_finite_json(r)
 with pytest.raises(ValueError,match="non-finite"):MOD.assert_finite_json({"x":float("nan")})
def test_no_overwrite(tmp_path):
 MOD.create_unique_output_dir(tmp_path,stamp="20260730T120000Z")
 with pytest.raises(FileExistsError):MOD.create_unique_output_dir(tmp_path,stamp="20260730T120000Z")
def test_relation_only_shuffle_preserves_local_evidence_but_changes_relations():
 rows=[]; directions=np.eye(9)[:3]
 for p_i,p in enumerate(("G01","G02","G03")):
  for t in range(6):
   norm=0. if (p_i==0 and t==0) else float(1+p_i+2*t); direction=directions[(t+p_i)%3]; vector=direction*norm*3.0
   r={"run_id":"r","prn":p,"window_bin_s":t/2,"window_start_s":t/2-.2,"window_end_s":t/2+.2,"window_mid_s":t/2,"b0_prn_node_rmse":norm};r.update({f"innovation_{i}":vector[i] for i in range(9)});rows.append(r)
 f=pd.DataFrame(rows); s=MOD.shuffle_innovations_within_prn(f,seed=17); cols=[f"innovation_{i}" for i in range(9)]
 original_rmse=np.sqrt(np.mean(f[cols].to_numpy()**2,axis=1)); shuffled_rmse=np.sqrt(np.mean(s[cols].to_numpy()**2,axis=1))
 np.testing.assert_allclose(shuffled_rmse,original_rmse,rtol=1e-12,atol=1e-12);np.testing.assert_array_equal(s.b0_prn_node_rmse.to_numpy(),f.b0_prn_node_rmse.to_numpy())
 assert np.isfinite(s[cols].to_numpy()).all(); assert np.count_nonzero(s[cols].to_numpy()!=f[cols].to_numpy())>0
 _,oe=MOD.score_common_drive(f); _,se=MOD.score_common_drive(s); oe=MOD.causal_smooth_events(oe,alpha=MOD.ALPHA);se=MOD.causal_smooth_events(se,alpha=MOD.ALPHA)
 np.testing.assert_array_equal(se[["run_id","window_bin_s"]].to_numpy(),oe[["run_id","window_bin_s"]].to_numpy());np.testing.assert_allclose(se.event_local_support,oe.event_local_support,rtol=1e-12,atol=1e-12)
 assert np.any(np.abs(se.event_common_drive_support.to_numpy()-oe.event_common_drive_support.to_numpy())>1e-12)
 thresholds={c:{"q95":float(oe[c].quantile(.4)),"q99":float(oe[c].quantile(.7))} for c in MOD.BRANCHES};MOD.verify_relation_only_diagnostic(f,oe,s,se,thresholds);of,sf=MOD.branch_flags(oe,thresholds),MOD.branch_flags(se,thresholds);masks=MOD.official_masks(oe)
 for region in ("pre","guard","post"):
  for q in (95,99):np.testing.assert_array_equal(np.asarray(of[f"event_local_support_q{q}"])[masks[region]],np.asarray(sf[f"event_local_support_q{q}"])[masks[region]])


def test_trusted_manifest_pins_and_fail_closed(tmp_path):
 assert MOD.EXPECTED_CAMPAIGN_MANIFEST_SHA256=="4519aad0cf69a7f50efa71f713525381decff265228a8d2f05ca1dc1087bfcd4"
 assert MOD.EXPECTED_SPLIT_MANIFEST_SHA256=="33b06abf1fe632d6c6f770f3d19870eca038ed86c8727c10bce22a2155579353"
 assert MOD.EXPECTED_ATTACK_NODE_MANIFEST_SHA256=={"os1":"d583a2acf71f230a9a225c1cc14b283c16ed556f21ff8517934494feb3770684","os2":"e4114229a88dd048902e47871e5abf1c79210d978531ef2b8031300bd1751762","os3":"b1ad3dc50edfc5350088ba8480acf764945ded67f5b080eb8f31ed6fe2d97ffd","os4":"a7e003962f54d4ac8f3f5231f539b0a0f9fe5040de636f3e6408666e65c14170"}
 p=tmp_path/"manifest.json";p.write_text("untrusted")
 with pytest.raises(ValueError,match="trusted.*SHA"):MOD.verify_trusted_manifest(p,"0"*64)
 MOD.verify_trusted_manifest(p,"0"*64,allow_unpinned_manifests=True)


def test_post_alarm_delays_are_explicit_relative_to_onset():
 t={c:{"q95":0.,"q99":0.} for c in MOD.BRANCHES};r=MOD.scenario_report(ef(),t,"os1");post=r["branches"]["event_local_support_q95"]["post"]
 assert post["first_alarm_score_time_s"]==130.;assert post["first_alarm_availability_time_s"]==131.;assert post["score_time_delay_s"]==10.;assert post["availability_time_delay_s"]==11.



def test_relation_shuffle_keeps_zero_rows_zero_and_reports_counts():
 rows=[]
 for t,v in enumerate(([0.]*9,[1.]+[0.]*8,[0.,2.]+[0.]*7,[0.]*9)):
  r={"run_id":"r","prn":"G01","window_bin_s":float(t),"b0_prn_node_rmse":float(np.sqrt(np.mean(np.square(v))))}
  r.update({f"innovation_{i}":x for i,x in enumerate(v)}); rows.append(r)
 original=pd.DataFrame(rows)
 shuffled,stats=MOD.shuffle_innovations_within_prn(original,seed=17,return_report=True)
 cols=[f"innovation_{i}" for i in range(9)]; zero=np.linalg.norm(original[cols],axis=1)==0
 assert np.count_nonzero(shuffled.loc[zero,cols].to_numpy())==0
 np.testing.assert_allclose(np.linalg.norm(shuffled.loc[~zero,cols],axis=1),np.linalg.norm(original.loc[~zero,cols],axis=1))
 assert stats["degenerate_zero_rows"]==2
 assert stats["permuted_nonzero_rows"]+stats["fixed_nonzero_rows"]==2


def test_unpinned_override_summary_is_explicitly_nonproduction(monkeypatch,tmp_path):
 root=tmp_path/"campaign"; root.mkdir(); (root/"campaign_manifest.json").write_text("campaign"); (root/"split_manifest.json").write_text("split")
 monkeypatch.setattr(MOD,"verify_frozen_campaign",lambda *a,**k:{"root":root,"frames":{"calibration":object(),"held_clean":object()}})
 monkeypatch.setattr(MOD,"load_model",lambda *a,**k:object())
 monkeypatch.setattr(MOD,"score_frame",lambda *a,**k:(pd.DataFrame(),None,ef()))
 monkeypatch.setattr(MOD,"create_unique_output_dir",lambda p:tmp_path/"out")
 monkeypatch.setattr(MOD,"atomic_csv",lambda *a,**k:None); monkeypatch.setattr(MOD,"atomic_json",lambda *a,**k:None)
 monkeypatch.setattr(MOD,"authenticate_attack",lambda p,s,allow_unpinned_manifests=False:(object(),{}, {"node_manifest":{"sha256":f"observed-{s}"}}))
 monkeypatch.setattr(MOD,"shuffle_innovations_within_prn",lambda x,seed,return_report=False:(x,{"permuted_nonzero_rows":0,"fixed_nonzero_rows":0,"degenerate_zero_rows":0}))
 monkeypatch.setattr(MOD,"score_common_drive",lambda x:(None,ef())); monkeypatch.setattr(MOD,"causal_smooth_events",lambda x,alpha:x); monkeypatch.setattr(MOD,"verify_relation_only_diagnostic",lambda *a,**k:None)
 summary=MOD.run(root,tmp_path,tmp_path,allow_unpinned_manifests=True)
 pins=summary["manifest_pin_provenance"]
 assert not pins["enforced"] and pins["override_used"]
 assert pins["expected"]["campaign_manifest_sha256"]==MOD.EXPECTED_CAMPAIGN_MANIFEST_SHA256
 assert pins["observed"]["campaign_manifest_sha256"]==MOD.sha256(root/"campaign_manifest.json")
 assert not summary["complete"] and summary["production_status"]=="non-production-incomplete"
