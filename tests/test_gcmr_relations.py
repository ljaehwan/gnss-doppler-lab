import math
from dataclasses import replace
import h5py
import numpy as np
import pytest
from gnss_doppler_lab.gcmr_geometry import GpsEphemeris
from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent, TrackingRow, _robust_z_differences, build_gcmr_pair_relation_events, code_carrier_consistency_hz, load_gnss_sdr_tracking_rows
from gnss_doppler_lab.trajectory import llh_to_ecef

def eph(p):
 return GpsEphemeris(p,.2*p,4.5e-9,.01,5153.7955,.7,.94,-.3,-8e-9,1e-10,1e-6,2e-6,200.,-80.,3e-8,-2e-8,100000.,2300,SV_health=0)
def rows(prns=(1,2,3,4),stop=1.,constant=False):
 out=[]
 for p in prns:
  for k,t in enumerate(np.arange(0.,stop,.02)):
   w=0. if constant else math.sin(.19*k+.31*p)+.03*p*k; d=-900+30*p+8*w
   out.append(TrackingRow(p,float(t),d,0. if constant else .2*w+.01*p*k,0. if constant else .03*math.cos(.17*k+p),1023000+d/1540+.4*w,.1*w,.2*w,38+p+.1*w,.8+.02*p,10+w,2-w,channel=p,segment_index=0))
 return out
def build(x):
 return build_gcmr_pair_relation_events(x,ephemerides={p:eph(p) for p in range(1,9)},receiver_ecef=llh_to_ecef(45.,-73.,100.),gps_tow_at_time_zero_s=100200.,min_common_samples=8)

def test_hdf5_fixture_extraction_and_duplicates(tmp_path):
 raw=tmp_path/'raw'; raw.mkdir(); vals={'PRN':[2,2,0,1,1,1],'PRN_start_sample_count':[40,60,80,0,20,200],'carrier_doppler_hz':[2,3,0,0,1,9]}; n=6
 for name,off in [('carr_error_filt_hz',.1),('code_error_filt_chips',.2),('code_freq_chips',1023000.),('carrier_doppler_rate_hz',.3),('code_freq_rate_chips',.4),('CN0_SNV_dB_Hz',35.),('carrier_lock_test',.5),('Prompt_I',1.),('Prompt_Q',2.)]: vals[name]=np.arange(n)+off
 def write(p):
  with h5py.File(p,'w') as h:
   for name,value in vals.items(): h.create_dataset(name,data=np.asarray(value))
 write(raw/'epl_tracking_ch_0.mat'); got=load_gnss_sdr_tracking_rows(raw,sample_rate_hz=1000,gap_threshold_s=.05)
 assert [(r.prn,r.time_s,r.segment_index) for r in got]==[(1,0.,0),(1,.02,0),(1,.2,1),(2,.04,0),(2,.06,0)]
 assert got[0].carr_error_filt_hz==pytest.approx(3.1); assert got[0].code_error_filt_chips==pytest.approx(3.2)
 write(raw/'epl_tracking_ch_1.mat')
 with pytest.raises(ValueError,match='duplicate exact tracking row'): load_gnss_sdr_tracking_rows(raw,sample_rate_hz=1000)
def test_cc_sign_formula():
 r=rows((1,),.02)[0]; assert code_carrier_consistency_hz(r)==pytest.approx(r.code_freq_chips-1023000-r.carrier_doppler_hz/1540)
def test_endpoint_input_and_padding_invariance():
 x=rows(); a=build(x)[0]; b=build(list(reversed(x)))[0]
 assert np.array_equal(a.pair_prns,b.pair_prns); assert np.all(a.pair_prns[:,0]<a.pair_prns[:,1]); assert np.allclose(a.observations,b.observations); assert np.allclose(a.conditions,b.conditions)
 padded=x+[replace(r,time_s=r.time_s+3) for r in rows((5,),.4)]; c=[e for e in build(padded) if e.window_start_s==0][0]
 assert np.array_equal(a.pair_prns,c.pair_prns); assert np.allclose(a.observations,c.observations); assert np.allclose(a.conditions,c.conditions)
def test_constant_correlations_masked_not_zeroed():
 e=build(rows(constant=True))[0]; corr=[2,4,6,8]
 assert not e.observation_mask[:,corr].any(); assert np.isnan(e.observations[:,corr]).all(); assert e.observation_mask[:,[1,3,5,7,9]].all()
def test_no_future_influence():
 prefix=rows(); future=[replace(r,time_s=r.time_s+1) for r in rows(stop=.5)]; a=build(prefix)[0]; b=[e for e in build(prefix+future) if e.window_start_s==0][0]
 assert np.allclose(a.observations,b.observations,equal_nan=True); assert np.array_equal(a.observation_mask,b.observation_mask); assert np.allclose(a.conditions,b.conditions)
def test_gap_rejected():
 x=[r for r in rows() if not (.30<=r.time_s<.70)]
 assert build(x)==[]
def test_event_contract_dimensions_and_finiteness():
 e=build(rows())[0]; assert isinstance(e,GcmrPairRelationEvent); assert (e.window_start_s,e.window_end_s)==pytest.approx((0.,1.)); assert e.pair_prns.shape==(6,2); assert e.observations.shape==(6,10); assert e.observation_mask.shape==(6,10); assert e.conditions.shape==(6,8); assert np.isfinite(e.observations[e.observation_mask]).all(); assert np.isfinite(e.conditions).all()


def test_robust_z_flat_plus_step_preserves_jump_when_mad_is_zero():
 values=np.asarray([0.,0.,0.,5.,5.,5.]); z=_robust_z_differences(values,np.arange(len(values)))
 assert np.isfinite(z[3]) and z[3] > 1.
 assert np.nanmax(np.abs(z[[1,2,4,5]])) < z[3]

def test_robust_z_truly_constant_is_zero():
 z=_robust_z_differences(np.ones(5),np.arange(5))
 assert np.array_equal(z[1:],np.zeros(4))


def test_unhealthy_ephemeris_prn_is_absent_and_cannot_bias_common_clock_or_pairs():
 healthy_rows=rows((1,2,3,4))
 poisoned=[replace(r,carrier_doppler_hz=r.carrier_doppler_hz+50000.) for r in rows((5,))]
 ephemerides={p:eph(p) for p in range(1,6)}
 ephemerides[5]=replace(ephemerides[5],SV_health=63)
 kwargs=dict(ephemerides=ephemerides,receiver_ecef=llh_to_ecef(45.,-73.,100.),
  gps_tow_at_time_zero_s=100200.,min_common_samples=8,min_prns=4)
 baseline=build_gcmr_pair_relation_events(healthy_rows,**kwargs)[0]
 candidate=build_gcmr_pair_relation_events(healthy_rows+poisoned,**kwargs)[0]
 assert 5 not in candidate.pair_prns
 assert np.array_equal(candidate.pair_prns,baseline.pair_prns)
 assert np.allclose(candidate.observations,baseline.observations,equal_nan=True)
 assert np.allclose(candidate.conditions,baseline.conditions)

def test_relation_build_rejects_fewer_than_min_healthy_tracked_prns():
 ephemerides={p:eph(p) for p in range(1,5)}
 ephemerides[4]=replace(ephemerides[4],SV_health=63)
 with pytest.raises(ValueError,match='healthy tracked PRNs'):
  build_gcmr_pair_relation_events(rows((1,2,3,4)),ephemerides=ephemerides,
   receiver_ecef=llh_to_ecef(45.,-73.,100.),gps_tow_at_time_zero_s=100200.,min_prns=4)


def test_event_contract_rejects_bad_windows_and_noncanonical_pairs():
 for bounds in ((float("nan"),1.),(1.,1.),(2.,1.)):
  with pytest.raises(ValueError,match="window"):
   GcmrPairRelationEvent(*bounds,np.array([[1,2]]),np.zeros((1,10)),np.ones((1,10),bool),np.zeros((1,8)))
 for pairs in (np.array([[2,1]]),np.array([[1,33]]),np.array([[1,2],[1,2]])):
  with pytest.raises(ValueError,match="pair|PRN"):
   GcmrPairRelationEvent(0.,1.,pairs,np.zeros((len(pairs),10)),np.ones((len(pairs),10),bool),np.zeros((len(pairs),8)))

def test_geometry_uses_median_actual_timestamp_in_each_bin(monkeypatch):
 import gnss_doppler_lab.gcmr_relations as relations
 calls=[]; original=relations.satellite_observation
 def capture(receiver,eph,tow): calls.append(tow); return original(receiver,eph,tow)
 monkeypatch.setattr(relations,"satellite_observation",capture)
 x=[replace(r,time_s=r.time_s + (0.003 if r.prn%2 else 0.007)) for r in rows(stop=.2)]
 build_gcmr_pair_relation_events(x,ephemerides={p:eph(p) for p in range(1,5)},receiver_ecef=llh_to_ecef(45.,-73.,100.),gps_tow_at_time_zero_s=100200.,window_s=.2,stride_s=.2,resample_bin_s=.02,min_common_samples=8)
 assert calls and calls[0] == pytest.approx(100200.003)
 assert abs(calls[0] - 100200.) > 1e-4
