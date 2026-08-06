import numpy as np, json
from pathlib import Path
from gnss_doppler_lab.acaf_nf_stage0 import *
def test_raw_caf_complex_peak():
 x=replica(3,1000000,1000);s=caf_complex(x,3,1000000,0,0,[-.125,0,.125],[-50,0,50]);assert np.unravel_index(abs(s).argmax(),s.shape)==(1,1)
def test_gain_invariance():
 x=replica(3,1000000,1000);a,_=normalize_caf(caf_complex(x,3,1000000,0,0));b,_=normalize_caf(caf_complex(augment(x,2,0,0,1),3,1000000,0,0));assert np.allclose(a,b)
def test_phase_invariance():
 x=replica(3,1000000,1000);a,_=normalize_caf(caf_complex(x,3,1000000,0,0));b,_=normalize_caf(caf_complex(augment(x,1,1.2,0,1),3,1000000,0,0));assert np.allclose(a,b,atol=1e-9)
def test_two_source_valid():
 x=two_source_control(3,6,1000000,7);assert x.dtype.kind=='c' and len(x)==1000
def test_k_exact(): assert select_k([16,3,9,5])==[3,5,9,16]
def test_split_chronological(): assert chronological_clean_split(range(8))['fit']==[0,1,2,3]
def test_attack_fit_guard(): assert attack_free_fit({'fit':['cleanStatic']},{'fit':'cleanStatic'})
def test_ds78_fail_closed(): assert ds78_overlap_status(None,None)['status']=='INCONCLUSIVE'
def test_onset_alignment(): assert onset_alignment(110,110)['aligned']
def test_same_epochs(): assert same_epochs([1,2],[1,2]) and not same_epochs([1],[2])
def test_seed_determinism(): assert np.allclose(augment(np.ones(4),1,0,.1,8),augment(np.ones(4),1,0,.1,8))
def test_manifest_strict(): assert json.loads(strict_manifest({'a':1}))['a']==1
def test_h0_and_score():
 h=fit_h0([[0,0],[1,1]]);assert h0_score([0,0],h)>=0
def test_two_source_fit():
 z=np.array([1+0j,0+0j]);d=two_source_fit(z,[np.array([1,0]),np.array([0,1])]);assert 'bic_improvement' in d
