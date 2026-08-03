from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path); m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m); return m

def test_canonical_artifact_hash_contract():
 r=load('r','scripts/run_amcf_r1_texbat.py'); assert set(r.CANONICAL)=={'cleanStatic','DS1','DS2','DS3','DS7','DS8'} and r.DS4_STATUS.startswith('NA'); assert all(len(h)==64 and p.suffix=='.npz' for p,h in r.CANONICAL.values())

def test_saved_alarm_recompute_exact_columns():
 r=load('r2','scripts/run_amcf_r1_texbat.py'); out=r.attach_alarm_columns([{'score':2.}],1.,3.,1.5); assert {k for k in out[0] if k.startswith('alarm_')}=={'alarm_primary_q99','alarm_primary_q995','alarm_matched_clean_diagnostic'}; assert r.recompute_alarm_fraction(out,1.,3.,1.5)==1.

def test_artifact_manifest_tamper(tmp_path):
 r=load('r3','scripts/run_amcf_r1_texbat.py'); (tmp_path/'a').write_text('a\n'); r.write_hash_manifest(tmp_path); assert r.verify_hash_manifest(tmp_path)==1.; (tmp_path/'a').write_text('b\n'); assert r.verify_hash_manifest(tmp_path)<1.

def test_summary_regeneration_independent_decisions(tmp_path):
 s=load('s','scripts/summarize_amcf_r1.py'); (tmp_path/'metrics.csv').write_text('criterion,value,pass\nC1,1,true\nC2,0,false\n'); (tmp_path/'model_audit.json').write_text(json.dumps({'best_restored':True,'finite':True})); (tmp_path/'query_policy_metrics.csv').write_text('collapse_fraction\n0.2\n'); result=s.summarize(tmp_path); assert set(result['criteria'])=={f'C{i}' for i in range(1,10)} and set(result['decisions'])=={'Detector operating point','Complex','Active','WCL'}; before=(tmp_path/'decision.json').read_bytes(); s.summarize(tmp_path); assert before==(tmp_path/'decision.json').read_bytes()

def test_required_output_inventory_constant():
 r=load('r4','scripts/run_amcf_r1_texbat.py'); expected={'config.json','provenance.json','input_hashes.json','environment.json','window_qa.json','prompt_rejection_by_phase.csv','training_history.csv','model_audit.json','thresholds.json','metrics.csv','seed_metrics.csv','ablation_metrics.csv','query_policy_metrics.csv','query_path_histogram.csv','bootstrap_confidence_intervals.csv','decision.json','README.md','hashes.json','per_epoch','plots','models'}; assert expected<=set(r.REQUIRED_OUTPUTS)


def test_summary_exact_c1_c9_semantics_no_file_defaults(tmp_path):
 import csv
 s=load('sx','scripts/summarize_amcf_r1.py')
 scenarios=['DS1','DS2','DS3','DS7','DS8']; rows=[]
 rows.append({'scenario':'cleanStatic','model':'primary 3-seed mean complex IG K7','operating_point':'q99','held_out_clean_fpr':.01})
 for sc in scenarios:
  rows += [
   {'scenario':sc,'model':'primary 3-seed mean complex IG K7','operating_point':'q99','stable_pre_fpr':.01,'roc_auc':.9,'post_detection':.8},
   {'scenario':sc,'model':'ensemble::complex all9::policyNone','operating_point':'q99','roc_auc':.9},
   {'scenario':sc,'model':'ensemble::magnitude all9::policyNone','operating_point':'q99','roc_auc':.8},
   {'scenario':sc,'model':'ensemble::complex all9 phase-destroyed::policyNone','operating_point':'q99','roc_auc':.7},
   {'scenario':sc,'model':'ensemble::complex all9 temporal-shuffled::policyNone','operating_point':'q99','roc_auc':.75},
   {'scenario':sc,'model':'primary 3-seed mean complex IG K7','operating_point':'matched_clean_diagnostic','stable_pre_fpr':.01,'roc_auc':.9,'post_detection':.8},
   {'scenario':sc,'model':'B0 Exact','operating_point':'matched_clean_diagnostic','stable_pre_fpr':.01,'roc_auc':.85,'post_detection':.75},
  ]
 def wc(path,rs):
  keys=[]
  for r in rs:
   for k in r:
    if k not in keys:keys.append(k)
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rs)
 wc(tmp_path/'metrics.csv',rows)
 seed=[]
 for ms in (101,202,303):
  for sc in scenarios:
   for k in (5,7):
    seed += [{'scenario':sc,'model':f'seed::complex IG K{k}::model{ms}::policyNone','roc_auc':.9},{'scenario':sc,'model':f'seed::complex fixed K{k}::model{ms}::policyNone','roc_auc':.8},{'scenario':sc,'model':f'seed::complex random K{k} policy11::model{ms}::policy11','roc_auc':.79}]
 wc(tmp_path/'seed_metrics.csv',seed);wc(tmp_path/'query_policy_metrics.csv',[{'model':'complex IG K7','modal_fraction':.5}])
 (tmp_path/'model_audit.json').write_text(json.dumps({f'complex_seed{x}':{'finite':True,'best_restored':True} for x in (101,202,303)}))
 d=s.summarize(tmp_path);assert all(d['criteria'].values());assert d['decisions']=={'Detector operating point':'GO','Complex':'GO','Active':'GO','WCL':'GO'}
 # No criterion may pass from mere file existence/model audit defaults.
 empty=tmp_path/'empty';empty.mkdir();(empty/'metrics.csv').write_text('');(empty/'query_policy_metrics.csv').write_text('');(empty/'seed_metrics.csv').write_text('');(empty/'model_audit.json').write_text('{}')
 e=s.summarize(empty);assert not any(e['criteria'].values())


def test_phase_destroyed_windows_rebuild_destroyed_history_not_primary_history():
 import numpy as np
 r=load('rphase','scripts/run_amcf_r1_texbat.py')
 from gnss_doppler_lab.amcf_r1 import PromptGate,build_causal_windows
 rng=np.random.default_rng(6);t=np.arange(.01,10,.05);z=(2+rng.random((len(t),9)))*np.exp(1j*rng.normal(size=(len(t),9)));z[:,4]=5*np.exp(1j*rng.normal(size=len(t)));iq=np.stack([z.real,z.imag],-1);recs,_=build_causal_windows(iq,t,np.ones(len(t)),recording_id='DS1',gate=PromptGate(.01))
 feat,dh=r.destroyed_feature_matrix(recs,iq,PromptGate(.01),101)
 assert feat.shape==(len(recs),9,7) and dh.shape==(len(recs),12,18)
 primary=r.histories(recs);assert not np.array_equal(dh,primary)
 # Last history slot is reconstructed from destroyed previous-window features.
 i=13;np.testing.assert_allclose(dh[i,-1,:14],np.r_[feat[i-1,3,:7],feat[i-1,5,:7]])


def test_performance_rows_emit_primary_q99_and_diagnostic_q995_independently():
    r=load('rq995','scripts/run_amcf_r1_texbat.py')
    clean=[]
    for i,score in enumerate(range(20)):
        clean.append({'scenario':'cleanStatic','decision_time_s':340+.5*i,'phase':'calibration','score':float(score)})
    for i,score in enumerate([0.,5.,25.]):
        clean.append({'scenario':'cleanStatic','decision_time_s':420+.5*i,'phase':'clean_test','score':score})
    attack=[
        {'scenario':'DS1','decision_time_s':30.,'phase':'stable_pre','score':0.},
        {'scenario':'DS1','decision_time_s':100.,'phase':'ramp','score':30.},
        {'scenario':'DS1','decision_time_s':140.,'phase':'persistent','score':31.},
    ]
    metrics,_=r.performance_rows({'demo':clean+attack},bootstrap_reps=0)
    ops={(x['scenario'],x['operating_point']) for x in metrics}
    assert ('cleanStatic','q99') in ops and ('cleanStatic','q995') in ops
    assert ('DS1','q99') in ops and ('DS1','q995') in ops
    q99=next(x for x in metrics if x['scenario']=='cleanStatic' and x['operating_point']=='q99')
    q995=next(x for x in metrics if x['scenario']=='cleanStatic' and x['operating_point']=='q995')
    assert q99['threshold']==q99['calibration_q99']
    assert q995['threshold']==q995['calibration_q995']


def test_window_diagnostics_persist_invariance_duplicate_and_count_distributions():
    import numpy as np
    r=load('rqa','scripts/run_amcf_r1_texbat.py')
    from gnss_doppler_lab.amcf_r1 import PromptGate,build_causal_windows
    rng=np.random.default_rng(72)
    time=np.tile(np.arange(.05,8.,.05),2)
    prn=np.repeat([3,7],len(time)//2)
    z=(2+rng.random((len(time),9)))*np.exp(1j*rng.normal(size=(len(time),9)))
    z[:,4]=5*np.exp(1j*rng.normal(size=len(time)))
    iq=np.stack([z.real,z.imag],-1)
    gate=PromptGate(.01)
    records,_=build_causal_windows(iq,time,prn,recording_id='DS1',gate=gate)
    qa=r.window_diagnostics(iq,time,prn,gate,records,recording_id='DS1')
    required={'global_phase_invariance_error_max','navigation_bit_sign_invariance_error_max','duplicate_epoch_prn_count','window_valid_sample_count','window_raw_sample_count','tracked_prn_count','prn_input_permutation_invariance_error_max'}
    assert required<=set(qa)
    assert qa['global_phase_invariance_error_max']<1e-10
    assert qa['navigation_bit_sign_invariance_error_max']<1e-10
    assert qa['duplicate_epoch_prn_count']==0
    assert qa['prn_input_permutation_invariance_error_max']<1e-10
    assert qa['window_valid_sample_count']['count']==len(records)


def test_summary_readme_contains_all_eight_required_interpretation_sections(tmp_path):
    s=load('sreadme','scripts/summarize_amcf_r1.py')
    (tmp_path/'metrics.csv').write_text('')
    (tmp_path/'seed_metrics.csv').write_text('')
    (tmp_path/'query_policy_metrics.csv').write_text('')
    (tmp_path/'prompt_rejection_by_phase.csv').write_text('')
    (tmp_path/'model_audit.json').write_text('{}')
    (tmp_path/'window_qa.json').write_text('{"scenarios": []}')
    s.summarize(tmp_path)
    text=(tmp_path/'README.md').read_text()
    for heading in (
        '1. Temporal aggregation and clean FPR',
        '2. Complex versus magnitude',
        '3. Phase destruction',
        '4. Sample-dependent active paths',
        '5. Fixed/random policy comparison',
        '6. B0 comparison',
        '7. Failure attribution',
        '8. WCL claims',
    ):
        assert heading in text
