import ast, csv, hashlib, importlib.util, json, shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import (
 BASELINE_SELECTORS,CENTER,DELAYS,DOPPLERS,FROZEN_CONFIG,H1_COORDINATES,
 add_awgn,alarm_metrics,amplitude_control,audit_provenance_roles,binary_metrics,
 block_bootstrap_effect,calibrate_scores,choose_pooling,chronological_split,
 consecutive_windows,dense_complex_caf,learn_diagonal_variance,noise_floor,
 normalize_caf,pool_prns,standardized_score,synthesize_same_prn_second_source,
 two_source_wls, H1Template, build_h1_template)
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code

ROOT=Path(__file__).resolve().parents[1]
def script(name):
 p=ROOT/"scripts"/name;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def row(i,scenario="cleanStatic",prn=1,channel="a",role=None):
 d={"scenario":scenario,"recording_sha256":"a"*64,"prn":prn,"channel":channel,"tracker_row":i,"support_start_sample":i*25000,"support_end_sample":i*25000+25000,"support_samples":25000,"cn0_db_hz":35,"carrier_lock":.95}
 if role:d["role"]=role
 return d

def test_canonical_code_and_frozen_lineage():
 code=gps_l1ca_code(1);assert code.shape==(1023,) and set(code)=={-1.,1.}
 assert hashlib.sha256(np.ascontiguousarray(code).view(np.uint8)).hexdigest()=="b201d6e762aac9d6ca916158d4770a38006236cf07cd67842773eed6cdf4b026"
 d=FROZEN_CONFIG.document();assert d["fs_hz"]==25e6 and d["support_samples"]==25000 and d["window_length"]==20
 assert len(DELAYS)==17 and len(DOPPLERS)==11 and CENTER not in H1_COORDINATES

def test_dense_complex_caf_shape_and_normalization_gain_phase_invariance():
 replica=synthesize_same_prn_second_source(1,25000,0,0,.3,1)
 c=dense_complex_caf(replica,1,1.023e6,0,0);assert c.shape==(11,17) and np.iscomplexobj(c)
 y,diag=normalize_caf(c);y2,_=normalize_caf(7*np.exp(1.2j)*c)
 assert np.allclose(y,y2) and diag["center_magnitude"]>0 and y[CENTER]==pytest.approx(1)

def test_normalization_floor_is_explicit():
 y,d=normalize_caf(np.zeros((11,17),complex),floor=.1);assert y[CENTER]==1 and np.count_nonzero(y)==1 and bool(d["floor_applied"])
 with pytest.raises(ValueError):normalize_caf(np.ones((11,17)),floor=0)

def test_chronological_clean_split_overlap_and_attack_exclusion():
 split=chronological_split([row(i) for i in reversed(range(10))],(.5,.2,.3))
 assert [x["support_start_sample"] for x in split["train"]]==list(range(0,125000,25000))
 with pytest.raises(ValueError):chronological_split([row(0,"ds7")])
 audit=audit_provenance_roles([dict(row(0),role="train"),dict(row(1,"ds3"),role="calibration")]);assert audit["status"]=="FAIL"
 overlapping=[row(i) for i in range(5)];overlapping[3]["support_start_sample"]=0
 with pytest.raises(ValueError):chronological_split(overlapping,(.4,.2,.4))

def test_causal_l20_same_prn_consecutive_and_20ms_rejected():
 assert len(consecutive_windows([row(i) for i in range(22)]))==3
 assert not consecutive_windows([row(i*20) for i in range(20)])
 mixed=[row(i,prn=1 if i<10 else 2) for i in range(20)];assert not consecutive_windows(mixed)
 assert not consecutive_windows([row(i) for i in list(range(19))+[20]])
 crossing=[row(i) for i in range(20)];crossing[10]["phase"]="post"
 for x in crossing[:10]:x["phase"]="pre"
 assert not consecutive_windows(crossing)
 bad=[row(i) for i in range(20)];bad[3]["support_end_sample"]+=1
 assert not consecutive_windows(bad)

def test_tracker_sample_binding_fixture(tmp_path):
 m=script("run_acaf_nf_stage1_static_feasibility.py");tracker=tmp_path/"raw";tracker.mkdir()
 with h5py.File(tracker/"epl_tracking_ch_0.mat","w") as f:
  values={"PRN_start_sample_count":np.arange(22)*25000,"PRN":np.ones(22),"CN0_SNV_dB_Hz":np.ones(22)*35,"carrier_lock_test":np.ones(22)*.9}
  values.update({name:np.ones(22) for name in ("carrier_doppler_hz","code_freq_chips","aux1","Prompt_I","Prompt_Q")})
  for name,value in values.items():f[name]=value
 rows,inventory=m.tracker_rows(tracker);assert len(rows)==20 and all(x["delta_previous"]==x["delta_next"]==25000 and x["same_prn_triple"] for x in rows) and inventory[0]["rows"]==22

def test_wls_h1_le_h0_and_second_source_sensitivity():
 rng=np.random.default_rng(4);t0=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));td=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));variance=np.ones_like(t0.real)
 coord=H1_COORDINATES[0];trusted=build_h1_template(coord,td,source_role="normal_train",construction_method="raw_iq_periodic_recorrelation",lineage={"fixture":True})
 plain=two_source_wls(t0,t0,{coord:trusted},variance);mixed=two_source_wls(t0+.6j*td,t0,{coord:trusted},variance)
 assert plain["h1_rss"]<=plain["h0_rss"] and mixed["h1_rss"]<=mixed["h0_rss"] and mixed["raw_s2src"]>plain["raw_s2src"]
 assert mixed["selected_delta"]!=(5,8)

def test_h1_template_authenticity_and_exact_coefficients():
 rng=np.random.default_rng(9);t0=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));td=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17))
 coord=H1_COORDINATES[0];trusted=build_h1_template(coord,td,source_role="normal_train",construction_method="raw_iq_periodic_recorrelation",lineage={"raw_sha256":"a"*64})
 variance=np.linspace(.2,2,t0.size).reshape(t0.shape);result=two_source_wls((2-.3j)*t0+(.4+.8j)*td,t0,{coord:trusted},variance)
 assert result["h1_alpha"]==pytest.approx(2-.3j) and result["h1_beta"]==pytest.approx(.4+.8j)
 for kwargs in ({"coordinate":(999,999)},{"source_role":"attack"},{"construction_method":"array_shift"}):
  args={"coordinate":coord,"surface":td,"source_role":"normal_train","construction_method":"raw_iq_periodic_recorrelation","lineage":{"raw_sha256":"a"*64}};args.update(kwargs)
  with pytest.raises(ValueError):build_h1_template(**args)

def test_variance_and_threshold_are_normal_only():
 rng=np.random.default_rng(3);v=learn_diagonal_variance(rng.normal(size=(10,11,17))+1j*rng.normal(size=(10,11,17)));assert v.shape==(11,17) and np.all(v[np.isfinite(v)]>0) and np.isnan(v[CENTER])
 cal=calibrate_scores([1,2,3,4]);assert cal["source"]=="cleanStatic_calibration_only" and np.isfinite(standardized_score(4,cal))
 with pytest.raises(ValueError):calibrate_scores([1,np.nan,2])

def test_pooling_permutation_invariant_and_variable_prn_count():
 a={1:1.,2:4.,3:2.,4:3.};b=dict(reversed(list(a.items())))
 for method in ("median","top50_mean","trimmed_mean"):
  x,dx=pool_prns(a,method);y,dy=pool_prns(b,method);assert x==y and dx["prn_count"]==dy["prn_count"]==4
 assert choose_pooling([{"scores":a},{"scores":{1:2,2:3,3:4}}]) in {"median","top50_mean","trimmed_mean"}

def test_raw_iq_controls_and_physical_positive_negative_delays():
 a=synthesize_same_prn_second_source(3,1000,.25,75,0,2);b=synthesize_same_prn_second_source(3,1000,-.25,-75,np.pi,1)
 assert np.allclose(np.abs(a),2) and np.allclose(np.abs(b),1) and not np.array_equal(a,b)
 controlled=amplitude_control(a,.5);assert np.sqrt(noise_floor(controlled))==pytest.approx(.5)
 assert not np.array_equal(add_awgn(controlled,.1,7),controlled)

def test_baselines_and_metric_helpers():
 assert BASELINE_SELECTORS["B0"]=="PROVISIONAL_UNAVAILABLE" and len(BASELINE_SELECTORS["fixed_9_delay_tap_complex"])==9
 m=binary_metrics([0,0,1,1],[0,.2,.8,1]);assert m["roc_auc"]==1
 a=alarm_metrics([0,1,2,3],[0,0,2,2],1,2);assert a["pre_onset_fpr"]==0 and a["alarm_delay_s"]==0
 boot=block_bootstrap_effect([0,0,0,0],[1,1,1,1],[0,0,1,1],20);assert boot["effect"]==1 and boot["block_seconds"]==10

def test_standardized_partial_auc_boundary_and_validation():
 assert binary_metrics([0,0,1,1],[0,.1,.9,1],.01)["partial_auc"]==pytest.approx(1)
 tied=binary_metrics([0,0,1,1],[.5,.5,.5,.5],.01)["partial_auc"]
 assert 0<=tied<=1
 for bad in (0,-.1,1.1):
  with pytest.raises(ValueError):binary_metrics([0,1],[0,1],bad)

def test_overlap_identity_and_metric_validation():
 a=dict(row(0),recording_sha256="a"*64);b=dict(row(0),recording_sha256="b"*64)
 # Equal offsets in different recordings are different bytes.
 from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import assert_no_raw_overlap
 assert_no_raw_overlap({"train":[a],"holdout":[b]})
 with pytest.raises(ValueError):alarm_metrics([1,0],[1,1],.5,0)
 with pytest.raises(ValueError):block_bootstrap_effect([0],[1],[0],0)

def test_timeline_mapping():
 m=script("run_acaf_nf_stage1_static_feasibility.py")
 assert m.phase_name("ds3",118.89)=="pre" and m.phase_name("ds3",118.9)=="onset_to_pulloff"
 assert m.phase_name("ds4",128.22)=="transition_only" and m.phase_name("ds8",3_750_000_000)=="time_push"

def test_fail_closed_never_invokes_attack_scoring_callback():
 m=script("run_acaf_nf_stage1_static_feasibility.py");called=False
 def forbidden():
  nonlocal called;called=True;raise AssertionError("attack scoring invoked")
 audits={s:{"matches_expected":True,"counts":m.EXPECTED_PHASE[s]} for s in ("cleanStatic","ds3","ds4","ds7","ds8")}
 result=m._foundation_gate_with_sentinel_for_test(audits,forbidden);assert result["verdict"]=="FOUNDATION_INVALID" and result["no_attack_raw_scoring_performed"] and not called

def _artifact(root):
 m=script("run_acaf_nf_stage1_static_feasibility.py");root.mkdir();(root/"plots").mkdir()
 reason="blocked"
 payload={"README.md":"FOUNDATION_INVALID\n","test_report.txt":"ok\n"}
 for n,text in payload.items():(root/n).write_text(text)
 m.dump(root/"config.json",{"x":1});m.dump(root/"source_binding.json",{"sources":{s:{"raw_sha256":h,"raw_bytes_read_purpose":"full_sha256_only","raw_path":"/none","tracker_path":"/none"} for s,h in m.HASHES.items()},"tracker_support_audits":{"cleanStatic":{},**{s:{"counts":m.EXPECTED_PHASE[s],"matches_expected":True} for s in ("ds3","ds4","ds7","ds8")}}})
 m.dump(root/"r14_frozen_lineage.json",{"artifact":"none"});m.dump(root/"scenario_timeline.json",m.TIMELINES)
 for n in ("normal_model_summary.json","thresholds.json","bootstrap_results.json"):m.dump(root/n,{"status":"NOT_EVALUATED","value":None,"reason":reason})
 for n,f in m.SCIENCE_CSV.items():m.write_csv_header(root/n,f)
 m.dump(root/"go_no_go.json",{"verdict":"FOUNDATION_INVALID","physics_feasibility_status":"NOT_EVALUATED","paper_candidate_status":"NOT_EVALUATED","stage2_justified":False})
 m.dump(root/"execution_validity.json",{"no_attack_raw_scoring_performed":True,"raw_bytes_read_purpose":"full_sha256_only","attack_iq_bytes_read_for_scoring":0})
 m.dump(root/"verification_report.json",{"status":"PENDING_INDEPENDENT_VERIFICATION","producer_verdict_not_authoritative":True})
 files={str(p.relative_to(root)):{"sha256":m.digest(p),"size_bytes":p.stat().st_size} for p in root.iterdir() if p.is_file() and p.name!="checksums.json"};m.dump(root/"checksums.json",{"algorithm":"sha256","files":files})

def test_independent_verifier_and_checksum_tamper(tmp_path):
 root=tmp_path/"artifact";_artifact(root);v=script("verify_acaf_nf_stage1_static_feasibility.py")
 first=v.verify(root,recompute_external=False);assert first["status"]=="INCOMPLETE" and first["external_recomputation_performed"] is False,first
 (root/"README.md").write_text("tampered\n");result=v.verify(root,recompute_external=False);assert result["status"]=="FAIL" and "checksum_tamper" in result["errors"]

def test_verifier_has_no_producer_import_or_verdict_function():
 p=ROOT/"scripts/verify_acaf_nf_stage1_static_feasibility.py";tree=ast.parse(p.read_text())
 imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
 assert not any("acaf_nf_stage1" in x or "run_acaf" in x for x in imports)
