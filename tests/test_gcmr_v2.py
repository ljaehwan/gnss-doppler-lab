import json,numpy as np,pytest
from gnss_doppler_lab.gcmr_v2 import *
def test_pair_masked_mse_and_zero_support():
 r=np.array([[2,9,5],[3,4,0],[99,99,99.]]); y=np.array([[0,0,1],[1,0,0],[0,0,0.]])
 assert pair_errors(r,y,[[1,0,1],[1,1,0],[0,0,0]],[1,1,0],observation_scale=[2,2,1]).tolist()==[8.5,2.5]
 with pytest.raises(ValueError,match="zero"): pair_errors(r[:1],y[:1],[[0,0,0]])
def test_triangle_median_and_complete_support():
 p=np.array([[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]); e=np.array([1,9,5,3,7,11])
 prn,a=node_raw(p,e); assert prn.tolist()==[1,2,3,4] and a.tolist()==[5,3,9,7]
 order=[4,1,5,0,3,2]; assert np.array_equal(node_raw(p[order],e[order])[1],a)
 with pytest.raises(InsufficientSupport): node_raw(p[:-1],e[:-1])
 with pytest.raises(InsufficientSupport): node_raw([[1,2],[1,3],[2,3]],[1,2,3])
def test_normalizer_tau_temperature_and_no_deletion():
 n,t,T,m=calibrate([np.array([0,1,2,3])],[np.array([0,1,2,999]),np.array([0,1,2,3])])
 assert n.center==1.5 and n.scale==pytest.approx(1.4826) and T==1.0
 s=score_nodes([8,2,5,1],[999,0,1,2],n,t,m); assert s.prns.tolist()==[1,2,5,8] and s.candidate_prn==8
def test_variable_n_single_bounded_and_multi_classification_strict():
 n=NodeNormalizer(0,1); tau=5
 a=score_nodes([1,2,3,8],[0,0,0,20],n,tau,.2); assert a.single_alarm and a.multi_prn_score<.26 and not a.multi_alarm and a.diffuse_support
 b=score_nodes([1,2,3,4,5,6,7,8],[0,0,0,0,0,0,20,20],n,tau,.2); assert b.multi_alarm and b.classification=="multi"
 boundary=score_nodes([1,2,3,4],[5,0,0,0],n,tau,1); assert not boundary.single_alarm and boundary.classification=="none"
def test_tie_smallest_prn_and_permutation():
 n=NodeNormalizer(0,1); x=score_nodes([9,2,5,1],[7,7,0,0],n,6,.9); assert x.candidate_prn==2
 y=score_nodes([1,5,2,9],[0,0,7,7],n,6,.9); assert np.array_equal(x.prns,y.prns) and np.allclose(x.z,y.z)
def test_calibration_includes_all_events():
 groups=[np.zeros(4) for _ in range(117)]+[np.array([0,0,0,100.]),np.array([0,0,0,100.])]
 _,_,_,threshold=calibrate([np.array([-1,0,1,2])],groups)
 baseline=calibrate([np.array([-1,0,1,2])],groups[:-1])[3]; assert threshold>baseline
def test_checkpoint_roundtrip_tamper(tmp_path):
 h={k:k*64 for k in ["implementation","source","cache","role","config"]}; p=tmp_path/"m.pt"
 save_checkpoint(p,payload={"node_center":1,"T":1.0},hashes=h); assert load_checkpoint(p,expected_hashes=h)["payload"]["T"]==1
 d=json.loads(p.read_text()); d["payload"]["T"]=2;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match="tamper"):load_checkpoint(p)
