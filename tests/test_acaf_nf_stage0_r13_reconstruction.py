import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import (
    Candidate, caf, candidate_fingerprint, clean_only_guard, code_replica,
    filter_stable_triples, gate_verdict, interval_rows, roles_nonoverlap,
    wide_grid,
)

ROOT = Path(__file__).resolve().parents[1]


def legacy_bad_code(prn):
    taps={1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}
    g1=np.ones(10,dtype=np.int8); g2=np.ones(10,dtype=np.int8); out=np.empty(1023,dtype=np.int8); a,b=taps[prn]
    for i in range(1023):
        out[i]=1 if g1[-1] == (g2[a-1] ^ g2[b-1]) else -1
        g1=np.r_[g1[1:],g1[2]^g1[9]]; g2=np.r_[g2[1:],g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]]
    return out


def test_canonical_ca_all_prns_and_no_local_generator():
    for prn in range(1,33):
        a=gps_l1ca_code(prn); b=gps_l1ca_code(f"G{prn:02d}")
        assert np.array_equal(a,b) and len(a)==1023 and set(a)=={-1,1}
        assert np.dot(a,a)==1023
    assert not np.array_equal(legacy_bad_code(1),gps_l1ca_code(1))
    for rel in ('src/gnss_doppler_lab/acaf_nf_stage0_r13_reconstruction.py','scripts/run_acaf_nf_stage0_static_r13_reconstruction.py'):
        tree=ast.parse((ROOT/rel).read_text())
        assert not any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='ca_code' for n in ast.walk(tree))


def row(i, prn=1, cn0=35, lock=.95):
    return {'PRN':prn,'PRN_start_sample_count':[100,25099,50099][i],
            'carrier_doppler_hz':100+i,'code_freq_chips':1_023_000+i,'aux1':.25+i,
            'Prompt_I':3+i,'Prompt_Q':4+i,'CN0_SNV_dB_Hz':cn0,
            'carrier_lock_test':lock,'mat_row':i,'channel':'ch0','mat_sha256':'a'*64}


def test_stable_filter_exact_lengths_quality_and_cross_prn_rejection():
    triples=filter_stable_triples([row(0),row(1),row(2)],raw_samples=100000)
    assert len(triples)==1 and interval_rows(triples[0],'prev_to_cur')[:2]==(100,25099)
    assert interval_rows(triples[0],'cur_to_next')[:2]==(25099,50099)
    assert not filter_stable_triples([row(0),row(1,cn0=27),row(2)],100000)
    assert not filter_stable_triples([row(0),row(1,lock=.84),row(2)],100000)
    assert not filter_stable_triples([row(0,cn0=27),row(1),row(2)],100000)
    assert not filter_stable_triples([row(0),row(1,prn=2),row(2)],100000)


def test_all_candidate_variables_physically_change_fingerprint():
    iq=np.ones(250,dtype=np.complex64); rows=(row(0),row(1),row(2)); base=Candidate()
    basefp=candidate_fingerprint(iq,rows,base,100,25099,25e6)
    variants=[Candidate(aux_row='previous'),Candidate(remnant_sign=-1),Candidate(carrier_sign=1),
              Candidate(code_freq_row='next'),Candidate(interval='cur_to_next'),Candidate(global_offset=500)]
    for candidate in variants:
        fp=candidate_fingerprint(iq,rows,candidate,100+candidate.global_offset,25099+candidate.global_offset,25e6)
        assert fp != basefp


def test_replica_equation_and_caf_known_center(monkeypatch):
    fs=1_023_000; n=1023; rows=(row(0),row(1),row(2)); c=Candidate(carrier_sign=-1)
    replica,chip_hash=code_replica(1,n,fs,1_023_000,.25,1,.125)
    expected=np.floor(.25+.125+np.arange(n)).astype(int)%1023
    assert np.array_equal(replica,gps_l1ca_code(1)[expected]) and chip_hash
    t=np.arange(n)/fs
    iq=gps_l1ca_code(1).astype(complex)*np.exp(1j*2*np.pi*100*t)
    result=caf(iq,1,fs,1_023_000,0,100,c,{'delay_chips':[0], 'doppler_hz':[0]})
    assert result['peak_delay_offset_chips']==0 and result['peak_doppler_offset_hz']==0
    assert result['center_magnitude']>1000


def test_grid_roles_clean_guard_and_fail_closed():
    assert wide_grid()=={'delay_chips':[round(-1+i*.125,3) for i in range(17)],'doppler_hz':list(range(-250,251,50))}
    assert roles_nonoverlap([{'role':'train','start':0,'end':10},{'role':'holdout','start':10,'end':20}])
    assert not roles_nonoverlap([{'role':'train','start':0,'end':11},{'role':'holdout','start':10,'end':20}])
    assert clean_only_guard(['cleanStatic'])
    with pytest.raises(ValueError): clean_only_guard(['ds1'])
    out=gate_verdict(False,True,True)
    assert out['verdict']=='SOURCE_BINDING_INVALID' and out['selected_alignment'] is None and out['physics_no_go_claim'] is False
    assert gate_verdict(True,False,True)['verdict']=='RECONSTRUCTION_IMPLEMENTATION_INVALID'
    assert gate_verdict(True,True,False)['verdict']=='TRACKER_RAW_ALIGNMENT_UNRESOLVED'
    assert gate_verdict(True,True,True)['verdict']=='PHYSICAL_CENTER_VALID'


def test_runner_calls_canonical_and_global_offsets_recompute(monkeypatch,tmp_path):
    path=ROOT/'scripts/run_acaf_nf_stage0_static_r13_reconstruction.py'
    spec=importlib.util.spec_from_file_location('runner',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    calls=[]
    monkeypatch.setattr(mod,'raw_caf',lambda *a,**kw: calls.append((kw['start'],kw['end'])) or {'center_magnitude':1})
    mod.run_offset_sensitivity('raw',[(row(0),row(1),row(2))],Candidate(),[0,500],25e6)
    assert len(calls)==2 and calls[0]!=calls[1]


def test_independent_verifier_rejects_ignored_candidate_and_fake_offsets():
    path=ROOT/'scripts/verify_acaf_nf_stage0_static_r13_reconstruction.py'
    spec=importlib.util.spec_from_file_location('verifier',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert not mod.candidate_fingerprints_valid({'expected_unique':2,'fingerprints':{'a':'same','b':'same'}})
    assert mod.candidate_fingerprints_valid({'expected_unique':2,'fingerprints':{'a':'one','b':'two'}})
    calls=[{'invocation':0,'global_offset_samples':o,'start_byte':400+4*o,'end_byte':800+4*o} for o in (-1000,-500,0,500,1000)]
    assert mod.global_offset_calls_valid(calls)
    for call in calls: call['start_byte']=400; call['end_byte']=800
    assert not mod.global_offset_calls_valid(calls)


def test_raw_caf_spies_canonical_generator(monkeypatch):
    path=ROOT/'scripts/run_acaf_nf_stage0_static_r13_reconstruction.py'
    spec=importlib.util.spec_from_file_location('runner_spy',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    seen=[]
    monkeypatch.setattr(mod,'read_iq',lambda *a: np.ones(10,dtype=complex))
    monkeypatch.setattr(mod,'gps_l1ca_code',lambda prn: seen.append(prn) or np.ones(1023))
    monkeypatch.setattr(mod,'caf',lambda *a,**k: {'result_field_hash':'x'})
    monkeypatch.setattr(mod,'candidate_fingerprint',lambda *a,**k: 'fp')
    result=mod.raw_caf('unused',(row(0),row(1),row(2)),Candidate(),start=100,end=110)
    assert seen==[1] and result['n_samples']==10
