import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from gnss_doppler_lab.nc_topi import (CANONICAL_TAP_COORDS,NONIDENTIFIABILITY_MARKER,RobustConditioner,aggregate_prn_scores,assert_fit_is_clean_only,b0_rmse,build_causal_iq_context,common_epoch_exact_join,equal_w_norm,fit_shrinkage_covariance,higher_quantile,load_config,normalize_tangents,paired_gap_safe_block_bootstrap,phase_masks,second_peak_perturbation,shift_peak,shuffled_control_target,source_support_split,standardized_pauc,strict_alarms,sustained_alarm_delay,validate_config,weighted_project,w_orthogonal_vector)

def W(): return np.diag(np.linspace(1.,2.,9))
def peak(): return np.exp(-4*CANONICAL_TAP_COORDS**2)

def test_config_explicit_coords_and_contract():
 c=load_config(); assert c["schema"]=="gnss-doppler-lab.nc-topi-stage0.v1"
 assert c["taps"]["coordinates_chips"]==[-.5,-.375,-.25,-.125,0,.125,.25,.375,.5]
 assert "GNSS-SDR" in c["taps"]["coordinate_provenance"]
 assert c["b0"]["checkpoint_sha256"]=="f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b"
 assert c["decision"]["go_primary"]=="q99 NC-TOPI median only"; validate_config(c)
 bad=json.loads(json.dumps(c)); bad["taps"]["coordinates_chips"][1]=-.4
 with pytest.raises(ValueError,match="explicit canonical"): validate_config(bad)

def test_nonuniform_derivative_and_metadata():
 x=np.array([-.5,-.31,-.12,0,.09,.21,.34,.43,.5]); z=normalize_tangents(1+x*x,x,include_width=False)
 assert np.allclose(z.raw[:,z.names.index("shift")],2*x,atol=1e-12)
 assert z.metadata["derivative_coordinates"]=="explicit physical chip coordinates"
 assert "prompt-relative" in z.metadata["normalization_caveat"]

def test_projection_idempotence_and_tangent_suppression():
 w=W(); J=normalize_tangents(peak(),CANONICAL_TAP_COORDS,W=w).matrix; r=2.4*J[:,0]-.7*J[:,1]+.2*J[:,2]
 a=weighted_project(r,J,w,lambda_relative=0); b=weighted_project(a.r_perp,J,w,lambda_relative=0)
 assert np.linalg.norm(a.r_perp)<1e-9 and np.allclose(b.r_perp,a.r_perp,atol=1e-10)

def test_orthogonal_preserved_and_rank_deficient_stable():
 w=W(); J=normalize_tangents(peak(),CANONICAL_TAP_COORDS,W=w).matrix
 o=equal_w_norm(w_orthogonal_vector(J,w,seed=17),J[:,1],w); q=weighted_project(o,J,w)
 assert np.linalg.norm(J.T@w@o)<1e-8 and q.perp_energy/q.total_energy>=.999999
 Jr=np.column_stack(([1.,0,0,0],[1.,0,0,0],[0.,1,0,0])); q=weighted_project([1,2,3,4],Jr,np.eye(4))
 assert q.rank==2 and np.isfinite(q.coefficients).all() and np.isfinite([q.total_energy,q.tangent_energy,q.perp_energy]).all()

def test_covariance_spd_floor_and_b0_rmse():
 c=fit_shrinkage_covariance(np.tile(np.arange(9.),(30,1)))
 assert np.linalg.eigvalsh(c.Sigma).min()>0 and np.isfinite(c.W).all()
 assert c.audit["fit_role"]=="clean_train residual_raw only"
 assert c.audit["floor_epsilon"]==pytest.approx(1e-8*np.trace(c.Sigma_unfloored)/9)
 assert np.allclose(b0_rmse([[3,4],[0,2]]),[np.sqrt(12.5),np.sqrt(2)])

def test_common_epoch_identity_join_exact():
 ids,a,b=common_epoch_exact_join([("r","G1",2),("r","G2",2),("r","G1",3)],[1,2,3],[ ("r","G1",3),("r","G1",2)],[30,10])
 assert ids==[("r","G1",2),("r","G1",3)] and a.tolist()==[1,3] and b.tolist()==[10,30]

def test_prn_permutation_variable_count_top25_ceil_and_nan_mask():
 ids=np.array(["G04","G01","G03","G02","G05"]); s=np.array([4.,1,3,2,100]); a=aggregate_prn_scores(ids,s,"median"); ix=[4,2,0,3,1]; b=aggregate_prn_scores(ids[ix],s[ix],"median")
 assert a.score==b.score==3 and a.ids==b.ids==tuple(sorted(ids)) and a.count==5
 t=aggregate_prn_scores(ids,s,"top25_mean"); assert t.selected_count==2 and t.score==52
 assert aggregate_prn_scores(ids[:3],s[:3],"top25_mean").selected_count==1
 with pytest.raises(ValueError,match="finite"): aggregate_prn_scores(["a","b"],[1,np.nan],"median")
 assert aggregate_prn_scores(["a","b"],[1,np.nan],"median",valid_mask=[1,0]).ids==("a",)

def test_low_signal_nonfinite_and_residual_only_rejected():
 with pytest.raises(ValueError,match="low-signal"): normalize_tangents(np.zeros(9),CANONICAL_TAP_COORDS)
 with pytest.raises(ValueError,match="finite"): normalize_tangents(np.r_[np.ones(8),np.nan],CANONICAL_TAP_COORDS)
 assert "non-identifiable" in NONIDENTIFIABILITY_MARKER
 with pytest.raises(ValueError,match="non-identifiable"): normalize_tangents(np.ones(9),CANONICAL_TAP_COORDS,input_kind="legacy_residual_only")

def test_higher_quantile_strict_and_clean_only():
 t=higher_quantile([1,2,3,4],.5,fit_roles=["clean_calibration"]*4); assert t==3
 assert strict_alarms([3,np.nextafter(3.,4.)],t).tolist()==[False,True]
 with pytest.raises(ValueError,match="clean-only"): higher_quantile([1,9],.99,fit_roles=["clean_calibration","DS1"])
 with pytest.raises(ValueError,match="attack"): assert_fit_is_clean_only(["clean_train","DS2"])

def test_source_splits_holdout_exclusion_and_attack_rejection():
 s=np.array([0,300,319,320,399,420,500.]); e=np.array([1,300,320,321,400,421,501.]); m=source_support_split(s,e,scenario="cleanStatic")
 assert m.train.tolist()==[1,1,0,0,0,0,0] and m.calibration.tolist()==[0,0,0,1,1,0,0] and m.holdout.tolist()==[0,0,0,0,0,1,1]
 assert not np.any(m.train&m.holdout)
 with pytest.raises(ValueError,match="attack.*fit"): source_support_split(s,e,scenario="DS1")

def test_phase_masks():
 s=np.array([20,30,89,90,99.5,100,140]); e=np.array([21,80,90,91,100.5,101,141]); m=phase_masks(s,e,100)
 assert m.stable_pre.tolist()==[0,1,0,0,0,0,0] and m.transition.tolist()==[1,0,1,1,1,0,0]
 assert m.post.tolist()==[0,0,0,0,0,1,1] and m.persistent.tolist()==[0,0,0,0,0,0,1]

def test_causal_iq_no_overlap_future_or_cross_run():
 ts=[2,2.5]; be=[.5,1,1.5,2,2.5,.5,1,1.5,2]; f=np.arange(18.).reshape(9,2)
 r=build_causal_iq_context(ts,be,f,history=4,target_groups=["a","b"],block_groups=["a"]*5+["b"]*4)
 assert r.valid.tolist()==[1,1] and r.block_indices[0].tolist()==[0,1,2,3] and r.block_indices[1].tolist()==[5,6,7,8]
 for i,ix in enumerate(r.block_indices): assert np.all(np.asarray(be)[ix]<=ts[i])

def test_robust_conditioner_contract_and_shuffle():
 X=np.arange(40.).reshape(20,2); y=1+.1*X[:,0]; c=RobustConditioner().fit(X,y,roles=["clean_train"]*20,feature_names=["log_power","flatness"])
 with pytest.raises(RuntimeError,match="calibration cap"): c.predict_scale(X[:2])
 c.calibrate_cap(X[10:],roles=["clean_calibration"]*10); p=c.predict_scale(X); assert np.all(p>0) and p.max()<=c.cap_+1e-12
 with pytest.raises(ValueError,match="forbidden"): RobustConditioner().fit(X,y,roles=["clean_train"]*20,feature_names=["log_power","prn"])
 with pytest.raises(ValueError,match="clean-only"): RobustConditioner().fit(X,y,roles=["clean_train"]*19+["DS1"])
 a=shuffled_control_target(np.arange(10.),roles=["clean_train"]*10); b=shuffled_control_target(np.arange(10.),roles=["clean_train"]*10)
 assert np.array_equal(a,b) and sorted(a)==list(np.arange(10.)) and not np.array_equal(a,np.arange(10.))

def test_pauc_mcclish_and_sustained_gap_reset():
 y=np.array([0,0,0,1,1,1]); s=np.array([.1,.4,.2,.8,.3,.9]); assert standardized_pauc(y,s)==pytest.approx(roc_auc_score(y,s,max_fpr=.05))
 r=sustained_alarm_delay([99,99.5,100,100.5,101,102,102.5,103],[1]*8,onset=100,stable_pre_mask=[1,1,0,0,0,0,0,0])
 assert r.already_alarming_stable_pre and r.alarm_time==101 and r.delay==1
 r=sustained_alarm_delay([100,100.5,101.5,102,102.5],[1]*5,onset=100); assert r.alarm_time==102.5 and r.delay==2.5

def test_gap_safe_block_bootstrap_no_iid_fallback():
 t=np.arange(40)*.5; a=np.arange(40.); r=paired_gap_safe_block_bootstrap(t,a,a-1,statistic=np.mean,reps=100)
 assert r.complete_block_count==2 and r.block_epoch_count==20 and r.point_estimate==pytest.approx(1) and r.ci==pytest.approx((1,1))
 with pytest.raises(ValueError,match="no IID fallback"): paired_gap_safe_block_bootstrap(np.arange(19)*.5,np.ones(19),np.zeros(19),reps=10)

def test_synthetic_physical_interpolation_and_stage0_grid():
 x=CANONICAL_TAP_COORDS; p=peak(); q=shift_peak(p,x,.125); assert q[np.argmax(x==.125)]==pytest.approx(p[np.argmax(x==0)])
 z=second_peak_perturbation(p,x,.2,.25); assert np.allclose(z-p,np.sqrt(.2)*shift_peak(p,x,.25))
 with pytest.raises(ValueError,match="physical grid"): second_peak_perturbation(p,x,.3,.25,enforce_stage0_grid=True)

def test_doc_frozen_grammar():
 text=(Path(__file__).parents[1]/"docs"/"NC_TOPI_STAGE0.md").read_text()
 for phrase in ["q99 NC-TOPI median only","NO-GO","INCONCLUSIVE","non-identifiable","McClish","no IID fallback","never retrain","cannot rescue the primary"]: assert phrase in text
