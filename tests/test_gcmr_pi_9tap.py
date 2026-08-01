import numpy as np
import pytest
from gnss_doppler_lab.gcmr_pi_9tap_pipeline import EventRecord, GCMRPeakInnovationPipeline, safe_center_tap_normalize

def event(t, bump=0.0):
    prns=('G01','G02','G03')
    base=np.array([1,2,3,4,10,4,3,2,1.],float)
    histories={p:np.stack([base+i*.01 for i in range(3)]) for p in prns}
    epl=np.stack([base+bump for _ in prns])
    cn0=np.array([38.,40.,42.]); elevation=np.array([20.,30.,40.])
    pairs={(a,b):np.array([.2,20.,40.]) for i,a in enumerate(prns) for b in prns[i+1:]}
    return EventRecord(t,prns,epl,histories,cn0,elevation,pairs)

def test_9d_center_normalization_and_rejection():
    value,valid=safe_center_tap_normalize(np.array([[1,2,3,4,10,4,3,2,1.]]))
    assert valid.tolist()==[True] and value[0,4]==1
    with pytest.raises(ValueError): safe_center_tap_normalize(np.ones((1,3)))

def test_9d_gpu_or_cpu_gru_normal_only_and_attack_scoring():
    pipe=GCMRPeakInnovationPipeline(3,hidden_size=4,epochs=2,seed=4,feature_dim=9)
    with pytest.raises(RuntimeError): pipe.score_attack(event(99))
    pipe.fit_normal([event(i) for i in range(5)],[event(10+i,.001*i) for i in range(5)])
    got=pipe.score_attack(event(99,.1))
    assert pipe.network.gru.input_size==9
    assert set(got.scores)=={'A0','A1','A2','A3','A4','Full'}
    assert np.isfinite(list(got.scores.values())).all()

def test_9d_rejects_three_dimension_record():
    e=event(1)
    bad=EventRecord(e.time,e.prns,np.ones((3,3)),e.histories,e.cn0,e.elevation,e.pair_conditions)
    with pytest.raises(ValueError): bad.validate(3)
