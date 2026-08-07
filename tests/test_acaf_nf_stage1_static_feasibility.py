import ast, csv, hashlib, importlib.util, json, shutil, sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import (
 BASELINE_SELECTORS,CENTER,DELAYS,DOPPLERS,FROZEN_CONFIG,H1_COORDINATES,
 add_awgn,alarm_metrics,amplitude_control,apply_gain_phase,audit_provenance_roles,binary_metrics,
 block_bootstrap_effect,calibrate_scores,choose_pooling,chronological_split,
 consecutive_windows,dense_complex_caf,learn_diagonal_variance,noise_floor,
 normalize_caf,pool_prns,standardized_score,synthesize_same_prn_second_source,
 two_source_wls, H1Template, build_h1_template_from_raw_recorrelation,
 _build_h1_template_from_surface_for_test)
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code

ROOT=Path(__file__).resolve().parents[1]
def script(name):
 p=ROOT/"scripts"/name;spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def row(i,scenario="cleanStatic",prn=1,channel="a",role=None):
 d={"scenario":scenario,"recording_sha256":"a"*64,"prn":prn,"channel":channel,"tracker_row":i,"support_start_sample":i*25000,"support_end_sample":i*25000+25000,"support_samples":25000,"cn0_db_hz":35,"carrier_lock":.95}
 if role:d["role"]=role
 return d

def h1_lineage(raw, start=0):
 interval_sha=hashlib.sha256(np.ascontiguousarray(raw,dtype=np.complex128).view(np.uint8)).hexdigest()
 return {"recording_sha256":"a"*64,"scenario":"cleanStatic","role":"normal_train",
         "raw_intervals":[{"start":start,"end":start+len(raw),"sha256":interval_sha,
                           "recording_sha256":"a"*64}],
         "construction_method":"raw_iq_periodic_recorrelation",
         "algorithm":"dense_complex_caf_periodic_l1ca","version":"1",
         "grid_sha256":hashlib.sha256(json.dumps({"delays":DELAYS,"dopplers":DOPPLERS},separators=(",",":"),sort_keys=True).encode()).hexdigest()}

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
 with pytest.raises(ValueError):normalize_caf(np.zeros((11,17),complex),floor=.1)
 with pytest.raises(ValueError):normalize_caf(np.ones((11,17)),floor=0)
 with pytest.raises(ValueError):normalize_caf(np.ones((11,17)),floor=np.nan)
 bad=np.ones((11,17),complex);bad[0,0]=np.nan
 with pytest.raises(ValueError):normalize_caf(bad)

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
 rows,inventory=m.tracker_rows(tracker);assert len(rows)==20 and all(x["delta_previous"]==x["delta_next"]==25000 and x["same_prn_triple"] for x in rows) and inventory["canonical_mat_files"][0]["rows"]==22

def test_wls_h1_le_h0_and_second_source_sensitivity():
 rng=np.random.default_rng(4);t0=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));td=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));variance=np.ones_like(t0.real)
 coord=H1_COORDINATES[0];lineage=h1_lineage(np.ones(25000,complex));trusted=_build_h1_template_from_surface_for_test(coord,td,source_role="normal_train",construction_method="raw_iq_periodic_recorrelation",lineage=lineage)
 plain=two_source_wls(t0,t0,{coord:trusted},variance,_test_allow_incomplete_grid=True);mixed=two_source_wls(t0+.6j*td,t0,{coord:trusted},variance,_test_allow_incomplete_grid=True)
 assert plain["h1_rss"]<=plain["h0_rss"] and mixed["h1_rss"]<=mixed["h0_rss"] and mixed["raw_s2src"]>plain["raw_s2src"]
 assert mixed["selected_delta"]!=(5,8)

def test_h1_template_authenticity_and_exact_coefficients():
 rng=np.random.default_rng(9);t0=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17));td=rng.normal(size=(11,17))+1j*rng.normal(size=(11,17))
 coord=H1_COORDINATES[0];lineage=h1_lineage(np.ones(25000,complex));trusted=_build_h1_template_from_surface_for_test(coord,td,source_role="normal_train",construction_method="raw_iq_periodic_recorrelation",lineage=lineage)
 assert trusted.surface.flags.writeable is False
 variance=np.linspace(.2,2,t0.size).reshape(t0.shape);result=two_source_wls((2-.3j)*t0+(.4+.8j)*td,t0,{coord:trusted},variance,_test_allow_incomplete_grid=True)
 assert result["h1_alpha"]==pytest.approx(2-.3j) and result["h1_beta"]==pytest.approx(.4+.8j)
 for kwargs in ({"coordinate":(999,999)},{"source_role":"attack"},{"construction_method":"array_shift"}):
  args={"coordinate":coord,"surface":td,"source_role":"normal_train","construction_method":"raw_iq_periodic_recorrelation","lineage":lineage};args.update(kwargs)
  with pytest.raises(ValueError):_build_h1_template_from_surface_for_test(**args)
 forged=object.__new__(H1Template);object.__setattr__(forged,"coordinate",coord);object.__setattr__(forged,"surface",td);object.__setattr__(forged,"source_role","normal_train");object.__setattr__(forged,"construction_method","raw_iq_periodic_recorrelation");object.__setattr__(forged,"digest","0"*64);object.__setattr__(forged,"lineage",lineage)
 with pytest.raises(ValueError):two_source_wls(t0,t0,{coord:forged},variance,_test_allow_incomplete_grid=True)
 with pytest.raises(ValueError):two_source_wls(t0,t0,{coord:trusted},variance)

def test_h1_deep_provenance_digest_and_raw_builder():
 raw=np.ones(25000,dtype=np.complex128); lineage=h1_lineage(raw);coord=H1_COORDINATES[0]
 template=build_h1_template_from_raw_recorrelation(coord,[raw],lineage=lineage,prn=1,
     code_freq_chips=1.023e6,aux1_samples=0,tracker_doppler_hz=0)
 assert isinstance(template.lineage_json,str) and template.surface.flags.writeable is False
 lineage["raw_intervals"][0]["start"]=9
 assert json.loads(template.lineage_json)["raw_intervals"][0]["start"]==0
 forged=object.__new__(H1Template)
 for name,value in template.__dict__.items():object.__setattr__(forged,name,value)
 object.__setattr__(forged,"source_role","selection")
 with pytest.raises(ValueError):two_source_wls(template.surface,template.surface,{coord:forged},np.ones((11,17)),_test_allow_incomplete_grid=True)
 bad=h1_lineage(raw);bad["recording_sha256"]="G"*64
 with pytest.raises(ValueError):build_h1_template_from_raw_recorrelation(coord,[raw],lineage=bad,prn=1,code_freq_chips=1.023e6,aux1_samples=0,tracker_doppler_hz=0)
 bad=h1_lineage(raw);bad["raw_intervals"][0]["sha256"]="0"*64
 with pytest.raises(ValueError):build_h1_template_from_raw_recorrelation(coord,[raw],lineage=bad,prn=1,code_freq_chips=1.023e6,aux1_samples=0,tracker_doppler_hz=0)

def test_variance_and_threshold_are_normal_only():
 rng=np.random.default_rng(3);v=learn_diagonal_variance(rng.normal(size=(10,11,17))+1j*rng.normal(size=(10,11,17)));assert v.shape==(11,17) and np.all(v[np.isfinite(v)]>0) and np.isnan(v[CENTER])
 cal=calibrate_scores([1,2,3,4]);assert cal["source"]=="cleanStatic_calibration_only" and np.isfinite(standardized_score(4,cal))
 with pytest.raises(ValueError):calibrate_scores([1,np.nan,2])

def test_pooling_permutation_invariant_and_variable_prn_count():
 a={1:1.,2:4.,3:2.,4:3.};b=dict(reversed(list(a.items())))
 for method in ("median","top50_mean","trimmed_mean"):
  x,dx=pool_prns(a,method);y,dy=pool_prns(b,method);assert x==y and dx["prn_count"]==dy["prn_count"]==4
 rows=[{"scores":a,"role":"clean_train","scenario":"cleanStatic","recording_sha256":"a"*64},{"scores":{1:2,2:3,3:4},"role":"selection","scenario":"cleanStatic","recording_sha256":"a"*64}]
 assert choose_pooling(rows,cleanstatic_sha256="a"*64) in {"median","top50_mean","trimmed_mean"}
 with pytest.raises(ValueError):choose_pooling([{"scores":a,"role":"calibration"}])
 with pytest.raises(ValueError):choose_pooling([dict(rows[0],scenario="ds4")],cleanstatic_sha256="a"*64)

def test_raw_iq_controls_and_physical_positive_negative_delays():
 a=synthesize_same_prn_second_source(3,1000,.25,75,0,2);b=synthesize_same_prn_second_source(3,1000,-.25,-75,np.pi,1)
 assert np.allclose(np.abs(a),2) and np.allclose(np.abs(b),1) and not np.array_equal(a,b)
 controlled=amplitude_control(a,.5);assert np.sqrt(noise_floor(controlled))==pytest.approx(.5)
 assert not np.array_equal(add_awgn(controlled,.1,7),controlled)
 for call in (lambda:apply_gain_phase(a,np.nan,0),lambda:apply_gain_phase(a,1,np.nan),
              lambda:add_awgn(a,-1),lambda:amplitude_control(a,np.nan),
              lambda:noise_floor([np.nan]),lambda:synthesize_same_prn_second_source(1,10,0,0,0,-1)):
  with pytest.raises(ValueError):call()

def test_baselines_and_metric_helpers():
 assert BASELINE_SELECTORS["B0"]=="PROVISIONAL_UNAVAILABLE" and len(BASELINE_SELECTORS["fixed_9_delay_tap_complex"])==9
 m=binary_metrics([0,0,1,1],[0,.2,.8,1]);assert m["roc_auc"]==1
 a=alarm_metrics([0,1,2,3],[0,0,2,2],1,2);assert a["pre_onset_fpr"]==0 and a["alarm_delay_s"]==0
 boot=block_bootstrap_effect([0,0,0,0],[1,1,1,1],20,times=[0,1,10,11],block_origin_s=0);assert boot["effect"]==1 and boot["block_seconds"]==10

def test_standardized_partial_auc_boundary_and_validation():
 assert binary_metrics([0,0,1,1],[0,.1,.9,1],.01)["partial_auc"]==pytest.approx(1)
 tied=binary_metrics([0,0,1,1],[.5,.5,.5,.5],.01)["partial_auc"]
 assert 0<=tied<=1
 for bad in (0,-.1,1.1):
  with pytest.raises(ValueError):binary_metrics([0,1],[0,1],bad)
 from sklearn.metrics import roc_auc_score
 for scores in ([.5,.5,.5,.5],[0,.4,.3,1],[0,0,1,1]):
  assert binary_metrics([0,0,1,1],scores,.1)["partial_auc"]==pytest.approx(roc_auc_score([0,0,1,1],scores,max_fpr=.1))

def test_overlap_identity_and_metric_validation():
 a=dict(row(0),recording_sha256="a"*64);b=dict(row(0),recording_sha256="b"*64)
 # Equal offsets in different recordings are different bytes.
 from gnss_doppler_lab.acaf_nf_stage1_static_feasibility import assert_no_raw_overlap
 assert_no_raw_overlap({"train":[a],"holdout":[b]})
 with pytest.raises(ValueError):alarm_metrics([1,0],[1,1],.5,0)
 with pytest.raises(ValueError):block_bootstrap_effect([0],[1],0)
 with pytest.raises(ValueError):block_bootstrap_effect([0],[1],10,times=None)
 with pytest.raises(ValueError):block_bootstrap_effect([0,np.nan],[1,2],10,times=[0,1])
 with pytest.raises(ValueError):alarm_metrics([],[],.5,0)
 with pytest.raises(ValueError):alarm_metrics([0],[1],np.nan,0)

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
 m=script("run_acaf_nf_stage1_static_feasibility.py");cfg=json.loads((ROOT/"configs/acaf_nf_stage1_source_binding.json").read_text());root.mkdir();(root/"plots").mkdir()
 reason="blocked"
 (root/"README.md").write_text("FOUNDATION_INVALID is not a physics `NO_GO`; attack IQ was never scored; science is NOT_EVALUATED; B0 PROVISIONAL_UNAVAILABLE.\n")
 sources={}
 for s,h in m.HASHES.items():
  spec=cfg["scenarios"][s];size=Path(spec["raw_path"]).stat().st_size;manifest=m.authenticate_manifest(json.loads(Path(spec["manifest_path"]).read_text()),s);receiver=m.authenticate_receiver_config(spec["receiver_config_path"],{**spec,"raw_size_bytes":size})
  sources[s]={"raw_path":spec["raw_path"],"raw_sha256":h,"expected_raw_sha256":h,"raw_size_bytes":size,"raw_sample_count":size//4,"raw_count_unit":"complex_int16_iq","raw_bytes_read_purpose":"full_sha256_only","raw_checks":{"size_divisible_by_4":{"status":"PASS"},"format":{"status":"PASS"},"rate":{"status":"PASS"},"sample_count":size//4},"tracker_path":spec["tracker_path"],"manifest_path":spec["manifest_path"],"manifest_sha256":spec["manifest_sha256"],"receiver_config_path":spec["receiver_config_path"],"receiver_config_sha256":spec["receiver_config_sha256"],"manifest_checks":manifest,"receiver_config_checks":receiver,"tracker_raw_binding":manifest["tracker_raw_binding"]}
 counts={s:{p:{**v,"dominant_fraction":0.0,"crossing_excluded":1 if s=="ds3" and p=="onset_to_pulloff" else 0} for p,v in phases.items()} for s,phases in m.EXPECTED_PHASE.items()}
 audits={s:{"scenario":s,"counts":counts[s],"expected":m.EXPECTED_PHASE[s],"matches_expected":True,"excluded":{},"boundary_crossing_supports":0,"mat_inventory":{"canonical_mat_files":[],"ignored_alias_symlinks":[],"ignored_regular_noninputs":[]},"accepted_rule":"previous-row support; exact triple; deltas 24999..25001; finite full schema; quality every row; bounds; half-open phase containment","twenty_ms_gaps_interpolated":False,"tracker_raw_binding_status":sources[s]["tracker_raw_binding"]["status"],"receiver_config_status":"PASS"} for s in m.HASHES}
 campaign={"frozen":m.FROZEN_CONFIG.document(),"source_binding_config_sha256":m.SOURCE_BINDING_SHA256,"search_complexity":{"calibration_statistic":"full_minimized_delta_search","scalar_penalty":0.0},"delay_grid":list(np.arange(-1,1.0001,.125)),"doppler_grid_hz":list(range(-250,251,50)),"pooling_candidates":["median","top50_mean","trimmed_mean"],"baseline_B0":"PROVISIONAL_UNAVAILABLE"};m.dump(root/"config.json",campaign);m.dump(root/"source_binding.json",{"sources":sources,"tracker_support_audits":audits,"ds7_ds8_pre_attack_pairing":"paired replay diagnostic only if byte identity authenticated"})
 r14=cfg["r14"];sp=__import__("subprocess");head=sp.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip();dirty=bool(sp.run(["git","status","--porcelain"],cwd=ROOT,text=True,capture_output=True,check=True).stdout);source_hashes={"producer_sha256":m.digest(ROOT/"scripts/run_acaf_nf_stage1_static_feasibility.py"),"module_sha256":m.digest(ROOT/"src/gnss_doppler_lab/acaf_nf_stage1_static_feasibility.py"),"verifier_sha256":m.digest(ROOT/"scripts/verify_acaf_nf_stage1_static_feasibility.py"),"config_sha256":m.digest(ROOT/"configs/acaf_nf_stage1_source_binding.json")};versions={"python":sys.version.split()[0],"numpy":np.__version__,"h5py":h5py.__version__};versions.update({p:__import__(p).__version__ for p in ("scipy","sklearn","matplotlib")});m.dump(root/"r14_frozen_lineage.json",{"artifact":str(ROOT/r14["artifact_path"]),"artifact_checksums_sha256":r14["checksums_sha256"],"verification_report_sha256":r14["verification_report_sha256"],"verifier_sha256":r14["verifier_sha256"],"module_sha256":r14["module_sha256"],"runner_sha256":r14["runner_sha256"],"verifier_required":"PASS","contract":m.FROZEN_CONFIG.document(),"base_sha":cfg["base_commit"],"stage1_source_hashes":source_hashes,"git_head":head,"git_dirty":dirty,"versions":versions,"status":"PASS"});m.dump(root/"scenario_timeline.json",m.TIMELINES)
 reason="DS4_MANIFEST_DOES_NOT_BIND_RAW_SHA;DS4_NO_POST_L20;DS7_NO_POST_SUPPORT;DS8_NO_TRANSITION_OR_HELD_SUPPORT;DS8_TIME_PUSH_ONE_PRN_DIAGNOSTIC_ONLY"
 for n in ("normal_model_summary.json","thresholds.json","bootstrap_results.json"):m.dump(root/n,{"status":"NOT_EVALUATED","value":None,"reason":reason})
 for n,f in m.SCIENCE_CSV.items():m.write_csv_header(root/n,f)
 m.dump(root/"go_no_go.json",{"verdict":"FOUNDATION_INVALID","PHYSICS_FEASIBILITY_GO":False,"physics_feasibility_status":"NOT_EVALUATED","PAPER_CANDIDATE_GO":False,"paper_candidate_status":"NOT_EVALUATED","stage2_justified":False,"reason":reason})
 m.dump(root/"execution_validity.json",{"status":"FOUNDATION_INVALID","no_attack_raw_scoring_performed":True,"raw_bytes_read_purpose":"full_sha256_only","attack_iq_bytes_read_for_scoring":0,"science_csv_semantics":"header_only","plots":{"count":0,"reason":reason},"B0":"PROVISIONAL_UNAVAILABLE"})
 commands=[[sys.executable,"-m","pytest","-q",*targets] for targets in m.TEST_TARGETS];runs=[{"command":c,"stdout":"1 passed in 0.01s\n","exit_code":0,"collected":1,"passed":1,"failed":0} for c in commands];m.dump(root/"test_report.txt",{"schema":"acaf-stage1-test-report-v1","source_head":head,"versions":versions,"environment":{"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},"runs":runs})
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

def test_exact_previous_row_support_and_boundary_exclusion(tmp_path):
 m=script("run_acaf_nf_stage1_static_feasibility.py");tracker=tmp_path/"raw";tracker.mkdir()
 with h5py.File(tracker/"epl_tracking_ch_0.mat","w") as f:
  stamps=np.array([2_972_450_001,2_972_475_001,2_972_500_000,2_972_525_000])
  values={"PRN_start_sample_count":stamps,"PRN":np.ones(4),"CN0_SNV_dB_Hz":np.ones(4)*35,"carrier_lock_test":np.ones(4)*.9}
  values.update({name:np.ones(4) for name in ("carrier_doppler_hz","code_freq_chips","aux1","Prompt_I","Prompt_Q")})
  for name,value in values.items():f[name]=value
 audit=m.support_audit(tracker,"ds3")
 assert audit["counts"]["pre"]["crossing_excluded"]==1
 assert audit["counts"]["onset_to_pulloff"]["triples"]==0

def test_config_pins_exact_metadata_and_ds4_aliases():
 cfg=json.loads((ROOT/"configs/acaf_nf_stage1_source_binding.json").read_text())
 assert set(cfg["scenarios"])=={"cleanStatic","ds3","ds4","ds7","ds8"}
 for value in cfg["scenarios"].values():
  assert all(value[k] for k in ("raw_path","raw_sha256","manifest_path","manifest_sha256","receiver_config_path","receiver_config_sha256","mat_inventory"))
 assert len(cfg["scenarios"]["ds4"]["ignored_alias_symlinks"])==11

def test_receiver_config_contract_and_skip_rejection(tmp_path):
 m=script("run_acaf_nf_stage1_static_feasibility.py");p=tmp_path/"receiver.conf"
 p.write_text("SignalSource.filename=/old/raw.bin\nSignalSource.item_type=ishort\nSignalSource.sampling_frequency=25000000\nSignalSource.samples=0\n")
 assert m.authenticate_receiver_config(p,{"raw_sha256":"a"*64})["first_file_sample_is_raw_sample"]==0
 p.write_text(p.read_text()+"SignalSource.skip_samples=1\n")
 assert m.authenticate_receiver_config(p,{"raw_sha256":"a"*64})["status"]=="FAIL"

def test_real_receiver_sample_count_semantics():
 m=script("run_acaf_nf_stage1_static_feasibility.py");cfg=json.loads((ROOT/"configs/acaf_nf_stage1_source_binding.json").read_text())
 for scenario,spec in cfg["scenarios"].items():
  size=Path(spec["raw_path"]).stat().st_size;audit=m.authenticate_receiver_config(spec["receiver_config_path"],{**spec,"raw_size_bytes":size})
  assert audit["status"]=="PASS" and audit["count_unit"]=="scalar_int16" and audit["covers_full_file"] is True
  assert audit["configured_count"]==(24008196096 if scenario=="cleanStatic" else 0)
  assert audit["full_source_count"]==size//2

def test_bootstrap_exact_derived_blocks_and_fragment_rejection():
 result=block_bootstrap_effect([0,0,0],[1,1,1],10,times=[0,9,10],block_origin_s=0,block_ids=[0,0,1])
 assert result["block_origin_s"]==0
 with pytest.raises(ValueError):block_bootstrap_effect([0,0,0],[1,1,1],10,times=[0,9,10],block_ids=[0,1,0])
 with pytest.raises(ValueError):block_bootstrap_effect([0],[1],10,times=[0],block_origin_s=0,block_seconds=5)

def test_no_removed_numpy_trapz_call():
 source=(ROOT/"src/gnss_doppler_lab/acaf_nf_stage1_static_feasibility.py").read_text()
 assert "np.trapz" not in source

def test_alarm_uses_minimum_alarm_time():
 assert alarm_metrics([0,1,2,3],[0,2,2,0],1,1)["alarm_delay_s"]==0

def test_ds4_manifest_is_unbound_fail_closed():
 m=script("run_acaf_nf_stage1_static_feasibility.py");cfg=json.loads((ROOT/"configs/acaf_nf_stage1_source_binding.json").read_text());doc=json.loads(Path(cfg["scenarios"]["ds4"]["manifest_path"]).read_text())
 assert m.authenticate_manifest(doc,"ds4")["tracker_raw_binding"]=={"status":"FAIL","reason":"MANIFEST_DOES_NOT_BIND_RAW_SHA","pointer":None}

def test_skip_cli_exits_two(tmp_path):
 root=tmp_path/"artifact";_artifact(root)
 result=__import__("subprocess").run([str(Path(sys.executable)),str(ROOT/"scripts/verify_acaf_nf_stage1_static_feasibility.py"),str(root),"--skip-external-recompute"],capture_output=True,text=True)
 assert result.returncode==2 and '"status": "INCOMPLETE"' in result.stdout

def test_unknown_go_field_is_rejected(tmp_path):
 root=tmp_path/"artifact";_artifact(root);m=script("run_acaf_nf_stage1_static_feasibility.py");go=json.loads((root/"go_no_go.json").read_text());go["attack_score"]=1;m.dump(root/"go_no_go.json",go)
 files={str(p.relative_to(root)):{"sha256":m.digest(p),"size_bytes":p.stat().st_size} for p in root.iterdir() if p.is_file() and p.name!="checksums.json"};m.dump(root/"checksums.json",{"algorithm":"sha256","files":files})
 assert "go_schema" in script("verify_acaf_nf_stage1_static_feasibility.py").verify(root,False)["errors"]

def _close_checksums(root):
 m=script("run_acaf_nf_stage1_static_feasibility.py");files={str(p.relative_to(root)):{"sha256":m.digest(p),"size_bytes":p.stat().st_size} for p in root.iterdir() if p.is_file() and p.name!="checksums.json"};m.dump(root/"checksums.json",{"algorithm":"sha256","files":files})

@pytest.mark.parametrize("mutation,error",[
 (lambda r,m:(lambda x:(x["frozen"].__setitem__("fs_hz",1),m.dump(r/"config.json",x)))(json.loads((r/"config.json").read_text())),"campaign_config_schema"),
 (lambda r,m:(lambda x:(x["sources"]["ds3"].__setitem__("manifest_sha256","0"*64),m.dump(r/"source_binding.json",x)))(json.loads((r/"source_binding.json").read_text())),"source_values:ds3"),
 (lambda r,m:(lambda x:(x["sources"]["ds4"].__setitem__("attack_score",1),m.dump(r/"source_binding.json",x)))(json.loads((r/"source_binding.json").read_text())),"source_binding_nested_schema:ds4"),
 (lambda r,m:(lambda x:(x.__setitem__("status","FORGED"),m.dump(r/"r14_frozen_lineage.json",x)))(json.loads((r/"r14_frozen_lineage.json").read_text())),"r14_lineage_schema"),
 (lambda r,m:(lambda x:(x.__setitem__("base_sha","0"*40),m.dump(r/"r14_frozen_lineage.json",x)))(json.loads((r/"r14_frozen_lineage.json").read_text())),"r14_lineage_schema"),
 (lambda r,m:(lambda x:(x["versions"].__setitem__("numpy","forged"),m.dump(r/"r14_frozen_lineage.json",x)))(json.loads((r/"r14_frozen_lineage.json").read_text())),"r14_lineage_schema"),
 (lambda r,m:(lambda x:(x.__setitem__("extra",1),m.dump(r/"r14_frozen_lineage.json",x)))(json.loads((r/"r14_frozen_lineage.json").read_text())),"r14_lineage_schema"),
])
def test_checksum_closed_nested_schema_mutations_fail(tmp_path,mutation,error):
 root=tmp_path/"artifact";_artifact(root);m=script("run_acaf_nf_stage1_static_feasibility.py");mutation(root,m);_close_checksums(root)
 assert error in script("verify_acaf_nf_stage1_static_feasibility.py").verify(root,False)["errors"]

def test_forged_substring_only_test_report_fails(tmp_path):
 root=tmp_path/"artifact";_artifact(root);(root/"test_report.txt").write_text("command: pytest exit_code: 0 collected: 99 passed: 99 failed: 0 Python numpy scipy h5py sklearn matplotlib") ;_close_checksums(root)
 assert script("verify_acaf_nf_stage1_static_feasibility.py").verify(root,False)["status"]=="FAIL"

def test_finalization_copies_before_verification_and_ignores_late_pending_mutation(tmp_path,monkeypatch):
 v=script("verify_acaf_nf_stage1_static_feasibility.py");pending=tmp_path/"pending";pending.mkdir();(pending/"plots").mkdir()
 for name in v.NAMES:(pending/name).write_text("original")
 final=tmp_path/"final";real_copy=shutil.copytree
 def copy_then_mutate(source,destination,*args,**kwargs):
  result=real_copy(source,destination,*args,**kwargs)
  if Path(source)==pending:(pending/"README.md").write_text("mutated-after-copy")
  return result
 seen=[]
 def verified_staging(root,*args,**kwargs):
  seen.append(Path(root));return {"status":"PASS","derived_verdict":"FOUNDATION_INVALID","errors":[],"independent_of_producer_verdict":True,"external_recomputation_performed":True,"integration_diagnostic":False}
 monkeypatch.setattr(v.shutil,"copytree",copy_then_mutate);monkeypatch.setattr(v,"verify",verified_staging)
 with pytest.raises(SystemExit) as exc:v.main([str(pending),"--finalize-to",str(final)])
 assert exc.value.code==0 and len(seen)==2 and all(p!=pending for p in seen)
 assert (final/"README.md").read_text()=="original" and (pending/"README.md").read_text()=="mutated-after-copy"
