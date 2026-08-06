import json
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r11_validity import (canonical_prn_identity, sampled_chip_indices, aux_samples_to_chips, raw_iq_s16le, alignment_candidates, parse_tracker_rows, chronological_split, stratified_round_robin_clean, center_gate, normal_only_fit, null_thresholds, classify_verdict, ds78_overlap)

def test_canonical_prns_1_32_are_acquisition_identity():
    assert canonical_prn_identity() == list(range(1,33))
    for p in range(1,33): assert np.array_equal(gps_l1ca_code(f"G{p:02d}"), gps_l1ca_code(p))
def test_25msps_ms_traverses_all_chips():
    x=sampled_chip_indices(25_000_000, 1.023e6); assert len(x)==25000 and set(x)==set(range(1023))
def test_aux_conversion_is_samples_not_assumed_chips(): assert aux_samples_to_chips(25000,1023000,25000000)==1023

def test_alignment_has_required_sign_and_row_candidates():
    names={x['name'] for x in alignment_candidates()}; assert {'interval_k_to_k1_tracker_k1_rem_plus_wipe_minus','interval_end_k_tracker_k_rem_minus_wipe_plus'} <= names

def test_raw_bounds_and_little_endian(tmp_path):
    p=tmp_path/'x.bin'; p.write_bytes(np.array([[1,2],[-3,4]],dtype='<i2').tobytes()); assert raw_iq_s16le(p,0,2)[1]==-3+4j
    with pytest.raises(ValueError): raw_iq_s16le(p,1,2)
def test_parse_all_prns_and_stability():
    rows=[{'PRN':p,'PRN_start_sample_count':p*25000,'carrier_doppler_hz':0,'code_freq_chips':1023000,'aux1':0,'CN0_SNV_dB_Hz':28,'carrier_lock_test':.85,'Prompt_I':1,'Prompt_Q':0} for p in range(1,6)]
    assert len(parse_tracker_rows(rows,25000000))==5

def test_parse_tracker_rows_retains_mat_provenance():
    row={'PRN':1,'PRN_start_sample_count':25000,'carrier_doppler_hz':0,'code_freq_chips':1023000,'aux1':0,'CN0_SNV_dB_Hz':28,'carrier_lock_test':.85,'Prompt_I':1,'Prompt_Q':0,'channel':'ch0','mat_path':'x.mat','mat_sha256':'abc','mat_index':7}
    parsed=parse_tracker_rows([row],25000000)[0]
    assert {key:parsed[key] for key in ('channel','mat_path','mat_sha256','mat_index')} == {'channel':'ch0','mat_path':'x.mat','mat_sha256':'abc','mat_index':7}
def test_chronological_split_no_raw_overlap():
    rows=[{'sample_count':i*25000,'end_sample':(i+1)*25000} for i in range(2000)]
    s=chronological_split(rows); assert [len(s[x]) for x in ('train','calibration','holdout')]==[1000,500,500] and s['train'][-1]['end_sample']<=s['calibration'][0]['sample_count']

def test_stratified_round_robin_clean_covers_all_stable_prns_and_retains_nonoverlap():
    rows=[]
    for epoch in range(20):
        for prn in (1, 2, 3, 4, 5):
            start=(epoch * 5 + prn) * 25_000
            rows.append({'prn':prn, 'sample_count':start, 'end_sample':start + 25_000})
    selected=stratified_round_robin_clean(rows, 15)
    assert len(selected)==15
    assert {r['prn'] for r in selected}=={1,2,3,4,5}
    assert max(sum(r['prn']==p for r in selected) for p in range(1,6)) - min(sum(r['prn']==p for r in selected) for p in range(1,6)) <= 1
    assert all(a['end_sample']<=b['sample_count'] for a,b in zip(selected,selected[1:]))
def test_stratified_round_robin_skips_overlapping_windows_without_prn_collapse():
    rows=[]
    for epoch in range(30):
        for prn in (1, 2, 3, 4, 5):
            start=epoch * 25_000
            rows.append({'prn':prn, 'sample_count':start, 'end_sample':start + 25_000})
    selected=stratified_round_robin_clean(rows, 15)
    assert len(selected)==15
    assert {r['prn'] for r in selected}=={1,2,3,4,5}
    assert all(a['end_sample']<=b['sample_count'] for a,b in zip(selected,selected[1:]))

def test_stratified_round_robin_seeds_scarce_prns_before_dense_middle_prn():
    rows=[]
    for epoch in range(30,70): rows.append({'prn':16,'sample_count':epoch*25_000,'end_sample':(epoch+1)*25_000})
    for epoch in range(2,6): rows.append({'prn':3,'sample_count':epoch*25_000,'end_sample':(epoch+1)*25_000})
    for epoch in range(20,25): rows.append({'prn':19,'sample_count':epoch*25_000,'end_sample':(epoch+1)*25_000})
    selected=stratified_round_robin_clean(rows, 20)
    assert {r['prn'] for r in selected}=={3,16,19}

def test_ds78_behavior_is_not_independent_without_provenance(tmp_path): assert ds78_overlap(None,None)['status']=='UNRECONSTRUCTABLE_RECORDING_RELATIVE_COUNTERS'


def test_center_gate_exact_thresholds():
    good={'n':500,'within_fraction':.95,'spearman':.9,'boundary_fraction':.05,'prn_count':4,'dominant_fraction':.5}; assert center_gate(good)['status']=='PASS'
    good['spearman']=.899; assert center_gate(good)['status']=='FAIL'
def test_normal_only_fit_and_null_no_claim():
    assert normal_only_fit(['clean'],['ds1'])['status']=='OK'; assert null_thresholds([])['performance_claim']==False
def test_blocked_incomplete_distinct_from_physics_no_go():
    assert classify_verdict('FAIL','NOT_EVALUATED_UNTIL_CENTER_VALID')['verdict']=='CENTER_RECONSTRUCTION_INVALID'
    assert classify_verdict('PASS','INCOMPLETE')['verdict']!='PHYSICS_NO_GO'
