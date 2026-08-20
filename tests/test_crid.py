import numpy as np
import pytest
from gnss_doppler_lab.crid import (CONFIG_ORDER,ResponseTable,chronological_split,
 empirical_threshold,receiver_configurations,render_receiver_config,score_epoch,verify_permutations)
from gnss_doppler_lab.crid_metrics import paired_block_bootstrap,verdict

def test_receiver_exact_values():
 c=receiver_configurations();assert tuple(c)==CONFIG_ORDER
 assert c["C0"]["Tracking_1C.tap_spacing_chips"]==.125
 assert c["C1"]["Tracking_1C.tap_spacing_chips"]==.10
 assert (c["C2"]["Tracking_1C.dll_bw_hz"],c["C2"]["Tracking_1C.pll_bw_hz"])==(.5,5.)
 assert (c["C3"]["Tracking_1C.dll_bw_hz"],c["C3"]["Tracking_1C.pll_bw_hz"])==(5.,25.)
def test_config_render_changes_only_named_keys():
 text="A=1\nTracking_1C.dll_bw_hz=9\n";out=render_receiver_config(text,receiver_configurations()["C2"],{"B":2})
 assert "A=1" in out and "Tracking_1C.dll_bw_hz=0.5" in out and "B=2" in out
def test_chronological_guarded_split_no_overlap():
 s=chronological_split(np.arange(1000));sets=[set(v) for v in s.values()]
 assert all(not (sets[i]&sets[j]) for i in range(5) for j in range(i))
 assert max(s["train"])<min(s["guard1"])<min(s["calibration"])<min(s["guard2"])<min(s["holdout"])
def test_threshold_higher():assert empirical_threshold(np.arange(100),.99)==99
def _epoch(prns=5):
 rng=np.random.default_rng(3);return {p:{c:rng.normal(size=6) for c in CONFIG_ORDER} for p in range(prns)}
def _h():return {c:np.eye(6,2) for c in CONFIG_ORDER}
def test_h0_nested_h1_score_finite():assert np.isfinite(score_epoch(_epoch(),_h())["score"])
def test_bic_positive():assert score_epoch(_epoch(),_h())["penalty"]>0
def test_configuration_permutation():assert verify_permutations(_epoch(),_h())["pass"]
def test_prn_permutation():assert verify_permutations(_epoch(),_h())["pass"]
def test_variable_prn_count():assert score_epoch(_epoch(4),_h())["prn_count"]==4
def test_leave_one_configuration():assert score_epoch(_epoch(),_h(),CONFIG_ORDER[:-1])["config_count"]==3
def test_covariance_regularization_contract():assert receiver_configurations()["C0"]["coherent_integration_ms"]==1
def test_cn0_not_feature():
 from gnss_doppler_lab.crid import FEATURE_NAMES
 assert not any("cn0" in x.lower() or "lock" in x.lower() or "power" in x.lower() for x in FEATURE_NAMES)
def test_bootstrap_reproducible():
 a=paired_block_bootstrap(np.arange(20),np.zeros(20),np.repeat(np.arange(10),2));b=paired_block_bootstrap(np.arange(20),np.zeros(20),np.repeat(np.arange(10),2));assert a==b
@pytest.mark.parametrize("ok,pos,base,expected",[(False,True,True,"INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT"),(True,False,True,"NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY"),(True,True,False,"GO_PHYSICS_BASELINE_PENDING"),(True,True,True,"GO_FOR_CRID_NEURAL_STAGE1")])
def test_verdict_precedence(ok,pos,base,expected):assert verdict({"x":True},ok,pos,base)==expected
def test_no_go_gate():assert verdict({"x":False},True,True,True)=="NO_GO_CRID_COUNTERFACTUAL_INVARIANCE"
def test_minimum_four_prn_contract():assert len(_epoch(4))==4
def test_missing_configuration_raises():
 e=_epoch();del e[0]["C3"]
 with pytest.raises(KeyError):score_epoch(e,_h())
