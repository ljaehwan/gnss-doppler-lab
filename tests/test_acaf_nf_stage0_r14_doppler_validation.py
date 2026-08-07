import ast, importlib.util, json
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
