import numpy as np
from gnss_doppler_lab.acaf_nf_stage0 import *
def test_synthetic_caf_peak():
 x=replica(3,1000000,1000);s=caf_surface(x,3,1000000,0,0,[-.125,0,.125],[-50,0,50]);assert np.unravel_index(s.argmax(),s.shape)==(1,1)
def test_two_source_gain_phase_k():
 assert two_source_control(3,6,1000000,7).size==1000;assert np.allclose(augment(np.ones(2),.5,np.pi/2,0,1),.5j);assert select_k([16,3,9,5])==[3,5,9,16]
def test_split_overlap_onset_manifest():
 s=chronological_clean_split(range(12));assert s["fit"]==list(range(6));assert attack_free_fit(s,{"fit":"cleanStatic"});assert ds78_overlap_status(None,None)["status"]=="INCONCLUSIVE";assert onset_alignment(125,125)["aligned"] and same_epochs([1],[1]);assert strict_manifest({"b":1,"a":2})==strict_manifest({"a":2,"b":1})
