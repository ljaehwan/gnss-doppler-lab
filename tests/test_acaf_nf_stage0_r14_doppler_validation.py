import ast, csv, hashlib, importlib.util, json, os, shutil
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest

from gnss_doppler_lab.acaf_nf_stage0_r14_doppler_validation import (
 CANDIDATE_STRING,FROZEN_CONFIG,LENGTHS,R13_REFERENCE,aggregation_gate,bootstrap_paired,
 check_r13_metrics,clean_only_guard,common_anchor_blocks,delay_gate,delay_metrics,
 diagnostic_aggregates,doppler_metrics,final_gates,normalized_noncoherent_power,
 offset_zero_clearly_better,paired_improvements,prompt_evidence,prompt_gate,prompt_metrics)
ROOT=Path(__file__).resolve().parents[1]

def datum(i,prn=1,channel="c",role="train",doppler=0,delay=0):
 return {"prn":prn,"channel":channel,"role":role,"tracker_row":i,"anchor_tracker_row":i,
 "support_start_sample":i*25000,"support_length_samples":25000,"valid_raw_support":True,
 "cn0_db_hz":35,"carrier_lock":.95,"center_magnitude":100+i,"mat_prompt_magnitude":100+i,
 "peak_delay_offset_chips":delay,"peak_doppler_offset_hz":doppler,"peak_magnitude":101+i,
 "peak_center_ratio":(101+i)/(100+i),"delay_boundary":False,"doppler_boundary":False}

def test_frozen_r13_config_and_metric_tolerance():
 assert FROZEN_CONFIG.document()=={"signal":"gps_l1ca_code","fs_hz":25e6,"raw_format":"interleaved_signed_int16_iq","global_offset_samples":0,"nco_row":"previous","aux_row":"previous","remnant_sign":-1,"carrier_sign":-1,"replica_direction":"forward","prompt_row":"current","support_samples":25000,"candidate_string":CANDIDATE_STRING}
 assert check_r13_metrics(R13_REFERENCE); bad=dict(R13_REFERENCE,pooled_spearman=R13_REFERENCE["pooled_spearman"]-1.1e-6); assert not check_r13_metrics(bad)

def test_prompt_relative_error_metrics_and_gates_by_role():
 rows=[datum(i) for i in range(10)]; rows[0]["center_magnitude"]*=1.001
 evidence=prompt_evidence(rows[0]); assert evidence["prompt_abs_relative_error"]==pytest.approx(.001)
 metrics=prompt_metrics(rows); assert metrics["p99_relative_error"]<.0011 and prompt_gate(metrics)
 rows[0]["center_magnitude"]*=1.1; assert not prompt_gate(prompt_metrics(rows))

def test_delay_and_doppler_are_separate_and_verdict_is_fail_closed():
 rows=[datum(i,doppler=250 if i<5 else 0) for i in range(100)]
 d=delay_metrics(rows); f=doppler_metrics(rows)
 assert d["within_0_125_fraction"]==1 and f["within_50_fraction"]==.95
 byp=[{"within_0_125_fraction":1} for _ in range(8)]; byr=[{"role":r,"within_0_125_fraction":1} for r in ("train","calibration","holdout")]
 assert delay_gate(d,byp,byr)
 assert final_gates(True,True,True,True,False)["verdict"]=="PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED"
 assert final_gates(False,True,True,True,True)["verdict"]=="RECONSTRUCTION_IMPLEMENTATION_INVALID"
 assert final_gates(True,True,False,True,True)["verdict"]=="TRACKER_RAW_RECONSTRUCTION_UNRESOLVED"

def test_normalized_noncoherent_formula_and_diagnostics():
 a=np.array([[1+1j,2],[3,4j]]); b=2*a
 got=normalized_noncoherent_power([a,b]); expected=np.mean([abs(a)**2/np.sum(abs(a)**2),abs(b)**2/np.sum(abs(b)**2)],axis=0)
 assert np.allclose(got,expected)
 assert set(diagnostic_aggregates([a,b]))=={"normalized_power_mean","raw_power_sum","magnitude_mean","robust_median"}

def test_l_blocks_common_anchor_and_rejections():
 rows=[datum(i) for i in range(25)]; blocks=common_anchor_blocks(rows)
 assert tuple(blocks)==LENGTHS and {L:len(x) for L,x in blocks.items()}=={1:6,5:6,10:6,20:6}
 assert all([b[-1]["tracker_row"] for b in blocks[L]]==list(range(19,25)) for L in LENGTHS)
 crossing=[datum(i,role="train" if i<10 else "holdout") for i in range(20)]
 assert not common_anchor_blocks(crossing)[20]
 mixed=[datum(i,prn=1 if i<10 else 2) for i in range(20)]
 assert not common_anchor_blocks(mixed)[20]
 duplicate=rows+[dict(rows[-1])]
 with pytest.raises(ValueError,match="duplicate"):common_anchor_blocks(duplicate)
 one_sample=[dict(datum(i),support_start_sample=i*24999) for i in range(20)]
 assert len(common_anchor_blocks(one_sample)[20])==1
 excessive=[dict(datum(i),support_start_sample=i*24998) for i in range(20)]
 assert not common_anchor_blocks(excessive)[20]

def test_paired_improvement_and_fixed_bootstrap_seed():
 l1=[datum(i,prn=i%8+1,doppler=100) for i in range(80)]; l20=[datum(i,prn=i%8+1,doppler=0) for i in range(80)]
 pairs=paired_improvements(l1,l20); assert all(x["difference"]==1 for x in pairs)
 a=bootstrap_paired(pairs,replicates=200); b=bootstrap_paired(pairs,replicates=200); assert a==b and a["ci95_low"]==1
 with pytest.raises(ValueError):paired_improvements(l1,l20[:-1])

def test_aggregation_gate_requires_all_predeclared_conditions():
 l20={"within_50_fraction":.96,"boundary_fraction":0}; boot={"ci95_low":.01}
 byp=[{"difference":.1} for _ in range(8)]; byr=[{"role":r,"difference":.1} for r in ("train","calibration","holdout")]
 assert aggregation_gate(l20,boot,byp,byr); byp[0]["difference"]=-.1; byp[1]["difference"]=-.1; assert not aggregation_gate(l20,boot,byp,byr)

def test_offset_zero_proof_is_diagnostic_not_selection():
 rows=[{"global_offset_samples":o,"pooled_spearman":1 if o==0 else .8} for o in (-1000,-500,0,500,1000)]
 assert offset_zero_clearly_better(rows); rows[0]["pooled_spearman"]=1; assert not offset_zero_clearly_better(rows)

def test_no_attack_data_or_prohibited_verdicts_in_r14_sources():
 clean_only_guard(["cleanStatic"])
 with pytest.raises(ValueError):clean_only_guard(["ds1"])
 for bits in np.ndindex(*(2,)*5):
  assert final_gates(*map(bool,bits))["verdict"] in {"RECONSTRUCTION_IMPLEMENTATION_INVALID","TRACKER_RAW_RECONSTRUCTION_UNRESOLVED","PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED","PHYSICAL_CENTER_VALID"}

def test_independent_verifier_does_not_import_producer_and_tamper_helpers_fail():
 path=ROOT/"scripts/verify_acaf_nf_stage0_static_r14_doppler_validation.py"; tree=ast.parse(path.read_text())
 assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) and "acaf_nf_stage0_r14" in ast.unparse(n) for n in ast.walk(tree))
 spec=importlib.util.spec_from_file_location("r14verify",path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 rows=[datum(i) for i in range(5)]; saved=mod.prompt([{k:str(v) for k,v in r.items()} for r in rows]); assert mod.close(saved,saved)
 tampered=dict(saved,pooled_spearman=.1); assert not mod.close(saved,tampered)

def test_smoke_vectorized_aggregation_is_small_and_finite():
 rng=np.random.default_rng(14); surfaces=rng.normal(size=(20,11,17))+1j*rng.normal(size=(20,11,17))
 out=diagnostic_aggregates(surfaces); assert out["normalized_power_mean"].shape==(11,17) and np.isfinite(out["normalized_power_mean"]).all()

def _load_script(name):
 path=ROOT/"scripts"/name;spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

@pytest.fixture(scope="module")
def synthetic_success_artifact(tmp_path_factory):
 """Exercise run() end-to-end while replacing only authenticated external I/O."""
 runmod=_load_script("run_acaf_nf_stage0_static_r14_doppler_validation.py")
 with (ROOT/"artifacts/acaf_nf_stage0_static_r13_reconstruction/center_validation.csv").open(newline="") as f:frozen=list(csv.DictReader(f))
 triples=[];byid={}
 for r in frozen:
  key=(str(r["channel"]),int(r["prn"]),int(r["tracker_row"]));byid[key]=r
  state={"mat_row":int(r["nco_row_index"]),"code_freq_chips":float(r["code_freq_chips_value"]),"carrier_doppler_hz":float(r["carrier_doppler_hz_value"]),"aux1":float(r["aux1_value"]),"mat_path":r["mat_path"]}
  prompt={"PRN":int(r["prn"]),"channel":str(r["channel"]),"mat_row":int(r["tracker_row"]),"Prompt_I":float(r["mat_prompt_magnitude"]),"Prompt_Q":0.,"CN0_SNV_dB_Hz":35.,"carrier_lock_test":.95,"mat_path":r["mat_path"],"_support":{"start_sample":int(r["support_start_sample"]),"end_sample":int(r["support_end_sample"]),"length_samples":25000}}
  triples.append((r["role"],(state,prompt,{})))
 out=tmp_path_factory.mktemp("r14-success")/"artifact";raw=out.parent/"cleanStatic.bin";raw.write_bytes(b"\0\0\0\0")
 runmod.authenticate_inputs=lambda *a,**k:{"checks":{"raw":True,"tracker":True,"manifest":True}};runmod.load_triples=lambda *a,**k:[];runmod.balanced_sample=lambda *a,**k:triples;runmod.source_support=lambda t,n:t[1]["_support"];runmod.read_iq=lambda *a,**k:np.zeros(8,dtype=np.complex128)
 def surface(iq,row):
  r=byid[(str(row["channel"]),int(row["prn"]),int(row["tracker_row"]))];s=np.zeros((11,17),complex);di=runmod.GRID["doppler_hz"].index(int(float(r["peak_doppler_offset_hz"])));ci=runmod.GRID["delay_chips"].index(float(r["peak_delay_offset_chips"]));s[5,8]=float(r["center_magnitude"]);s[di,ci]=float(r["peak_magnitude"]);return s
 runmod.complex_caf_surface=surface
 verdict=runmod.run(SimpleNamespace(raw=raw,tracker_dir=out.parent,manifest=None,output=out))
 assert verdict["verdict"] in {"TRACKER_RAW_RECONSTRUCTION_UNRESOLVED","PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED","PHYSICAL_CENTER_VALID"}
 return out

def test_run_success_latent_authenticated_io_and_inventory(synthetic_success_artifact):
 root=synthetic_success_artifact;assert len({p.name for p in root.iterdir()})==27;assert len([p for p in root.iterdir() if p.is_file()])==26;assert len(list((root/"plots").glob("*.svg")))==7
 assert (root/"prompt_reproduction_by_channel.csv").is_file()
 rows=list(csv.DictReader((root/"per_block_scores.csv").open()));assert sum(r["record_type"]=="epoch" for r in rows)==969;assert {int(r["L"]) for r in rows if r["record_type"]=="aggregate"}=={1,5,10,20};assert all(v==513 for v in json.loads((root/"execution_validity.json").read_text())["common_anchor_counts"].values())

def _rehash(verifier,root):
 checks={str(x.relative_to(root)):verifier.digest(x) for x in sorted(root.rglob("*")) if x.is_file() and x.name not in {"checksums.json","verification_report.json"}};(root/"checksums.json").write_text(json.dumps({"files":checks},indent=2,sort_keys=True)+"\n")

def test_independent_verifier_accepts_success(synthetic_success_artifact):
 assert _load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py").verify(synthetic_success_artifact)["status"]=="PASS"

@pytest.mark.parametrize("filename,kind,field",[("prompt_reproduction_metrics.json","json","median_relative_error"),("prompt_reproduction_by_channel.csv","csv","pooled_spearman"),("prompt_reproduction_by_time_block.csv","csv","pooled_spearman"),("per_block_scores.csv","constituent","constituent_identities"),("paired_improvement.csv","csv","difference"),("bootstrap_results.json","bootstrap","ci95_low"),("doppler_mainlobe_diagnostics.csv","csv","imag_+50_hz"),("go_no_go.json","json","verdict"),("execution_validity.json","json","caf_executed")])
def test_checksum_rewritten_tampers_fail(synthetic_success_artifact,tmp_path,filename,kind,field):
 import shutil
 verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");work=tmp_path/filename.replace(".","_");shutil.copytree(synthetic_success_artifact,work);p=work/filename
 if kind in {"csv","constituent"}:
  with p.open(newline="") as f:rows=list(csv.DictReader(f));fields=list(rows[0])
  i=969 if kind=="constituent" else 0;rows[i][field]=rows[i+1][field] if kind=="constituent" else "123"
  with p.open("w",newline="") as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 else:
  d=json.loads(p.read_text())
  if kind=="bootstrap":d["20"][field]=.123
  elif field=="caf_executed":d[field]=False
  elif field=="verdict":d[field]="RECONSTRUCTION_IMPLEMENTATION_INVALID" if d[field]!="RECONSTRUCTION_IMPLEMENTATION_INVALID" else "PHYSICAL_CENTER_VALID"
  else:d[field]=.5
  p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
 _rehash(verifier,work);assert verifier.verify(work)["status"]=="FAIL"

def test_r13_verification_report_tamper_fails(synthetic_success_artifact,monkeypatch):
 verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");real=verifier.loadj
 def tampered(path):
  value=real(path)
  if path.name=="verification_report.json" and path.parent.name=="acaf_nf_stage0_static_r13_reconstruction":value=dict(value,errors=["tampered"])
  return value
 monkeypatch.setattr(verifier,"loadj",tampered);result=verifier.verify(synthetic_success_artifact)
 assert result["status"]=="FAIL" and "r13_verification_report" in result["errors"]

def _isolated_r13(tmp_path,for_verifier=False):
 base=tmp_path/"isolated-root";art=base/"artifacts/acaf_nf_stage0_static_r13_reconstruction"
 shutil.copytree(ROOT/"artifacts/acaf_nf_stage0_static_r13_reconstruction",art)
 if for_verifier:
  source=base/"src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py";source.parent.mkdir(parents=True);shutil.copy2(ROOT/"src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py",source)
 return base,art

def _assert_r13_rejected(path_kind,base,art,synthetic_success_artifact):
 if path_kind=="producer":
  producer=_load_script("run_acaf_nf_stage0_static_r14_doppler_validation.py");producer.R13_ARTIFACT=art
  with pytest.raises(RuntimeError):producer.authenticate_r13()
 else:
  verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");verifier.ROOT=base;result=verifier.verify(synthetic_success_artifact)
  assert result["status"]=="FAIL"
  return result

REPORT_MUTATIONS={
 "verdict":(b'"verdict": "TRACKER_RAW_ALIGNMENT_UNRESOLVED"',b'"verdict": "TRACKER_RAW_ALIGNMENT_UNRESOLVEE"'),
 "gates":(b'"A1_SOURCE_BINDING": "PASS"',b'"A1_SOURCE_BINDING": "FAIL"'),
 "recomputed":(b'"n": 969,',b'"n": 968,'),
 "recursive":(b'"README.md": "597c36c3b555bc24128ad496db3a384b86934b344d7787d45d9eda751a1d0016"',b'"README.md": "697c36c3b555bc24128ad496db3a384b86934b344d7787d45d9eda751a1d0016"'),
 "ca":(b'"code_sha256": "b201d6e762aac9d6ca916158d4770a38006236cf07cd67842773eed6cdf4b026"',b'"code_sha256": "a201d6e762aac9d6ca916158d4770a38006236cf07cd67842773eed6cdf4b026"'),
}

@pytest.mark.parametrize("path_kind",["producer","verifier"])
@pytest.mark.parametrize("field",list(REPORT_MUTATIONS))
def test_r13_report_semantic_field_mutation_rejected_by_exact_digest(synthetic_success_artifact,tmp_path,path_kind,field):
 base,art=_isolated_r13(tmp_path,path_kind=="verifier");report=art/"verification_report.json";before=report.read_bytes();old,new=REPORT_MUTATIONS[field]
 assert before.count(old)==1 and len(old)==len(new);report.write_bytes(before.replace(old,new,1))
 assert report.stat().st_size==20868 and hashlib.sha256(report.read_bytes()).hexdigest()!="4a4177a51b2fcd1d552155e5714efbb69afcfa2fa3da51bd3594201bda884591"
 result=_assert_r13_rejected(path_kind,base,art,synthetic_success_artifact)
 if result is not None:assert "r13_verification_report_trust_anchor" in result["errors"]

@pytest.mark.parametrize("path_kind",["producer","verifier"])
def test_r13_report_byte_only_mutation_rejected_by_exact_digest(synthetic_success_artifact,tmp_path,path_kind):
 base,art=_isolated_r13(tmp_path,path_kind=="verifier");report=art/"verification_report.json";payload=report.read_bytes();assert payload.endswith(b"\n")
 report.write_bytes(payload[:-1]+b" ");assert report.stat().st_size==20868 and json.loads(report.read_text())["status"]=="PASS"
 result=_assert_r13_rejected(path_kind,base,art,synthetic_success_artifact)
 if result is not None:assert "r13_verification_report_trust_anchor" in result["errors"]

@pytest.mark.parametrize("path_kind",["producer","verifier"])
@pytest.mark.parametrize("extra_kind",["empty_dir","nested_file","symlink"])
def test_r13_exact_recursive_inventory_rejects_every_extra_entry(synthetic_success_artifact,tmp_path,path_kind,extra_kind):
 base,art=_isolated_r13(tmp_path,path_kind=="verifier")
 if extra_kind=="empty_dir":(art/"unexpected-empty").mkdir()
 elif extra_kind=="nested_file":
  nested=art/"plots/unexpected/nested.txt";nested.parent.mkdir();nested.write_text("unexpected\n")
 else:os.symlink("README.md",art/"unexpected-link")
 result=_assert_r13_rejected(path_kind,base,art,synthetic_success_artifact)
 if result is not None:assert "r13_manifest_inventory" in result["errors"]

@pytest.mark.parametrize("kind",["file","directory","symlink"])
def test_nested_extra_rewritten_checksum_fails_closed(synthetic_success_artifact,tmp_path,kind):
 import os,shutil
 verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");work=tmp_path/("nested-"+kind);shutil.copytree(synthetic_success_artifact,work);extra=work/"plots"/"nested"
 if kind=="file":extra.mkdir();(extra/"extra.txt").write_text("extra\n")
 elif kind=="directory":extra.mkdir()
 else:os.symlink("l-recovery.svg",work/"plots"/"extra-link.svg")
 _rehash(verifier,work)
 assert verifier.verify(work)=={"status":"FAIL","errors":["exact_recursive_inventory"]}

def test_plot_payload_tamper_with_rewritten_checksum_fails(synthetic_success_artifact,tmp_path):
 import shutil
 verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");work=tmp_path/"plot-tamper";shutil.copytree(synthetic_success_artifact,work);p=work/"plots/prompt-scatter.svg"
 text=p.read_text();needle="&quot;label&quot;:&quot;969 epochs&quot;";assert needle in text;p.write_text(text.replace(needle,"&quot;label&quot;:&quot;968 epochs&quot;",1));_rehash(verifier,work)
 result=verifier.verify(work);assert result["status"]=="FAIL" and "plot_semantic_prompt-scatter" in result["errors"]

def test_svg_plots_embed_distinct_canonical_evidence(synthetic_success_artifact):
 verifier=_load_script("verify_acaf_nf_stage0_static_r14_doppler_validation.py");digests=set()
 for name in verifier.PLOTS:
  text=(synthetic_success_artifact/"plots"/(name+".svg")).read_text();assert "canonical-data" in text and "data-sha256" in text and "polyline" not in text;digests.add(text.split('data-sha256="',1)[1].split('"',1)[0])
 assert len(digests)==7
