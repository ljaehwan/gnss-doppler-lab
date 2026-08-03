from __future__ import annotations
import math
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import pytest
import torch
from gnss_doppler_lab.amcf_r1 import (SIDE_INDICES,TAP_NAMES,AMCFModel,HiddenAccessGuard,PromptGate,alarm_columns,all9_score,assign_clean_role,attack_free_thresholds,build_causal_windows,checkpoint_load,checkpoint_save,complex_summary,epoch_random_extras,expected_information_gain,history_indices,normalize_prompt,phase_destroy,phase_masks,score_then_reveal,student_t_entropy,student_t_nll,verify_alarm_columns)

def iq(n=32,seed=2):
 rng=np.random.default_rng(seed); z=rng.normal(size=(n,9))+1j*rng.normal(size=(n,9)); z[:,4]+=5+2j; return np.stack([z.real,z.imag],-1)

def test_global_phase_and_nav_sign_invariant():
 x=iq(); g=PromptGate(.01,1e-12); a,va=normalize_prompt(x,g); z=(x[...,0]+1j*x[...,1])*np.exp(1j*.731); b,vb=normalize_prompt(np.stack([z.real,z.imag],-1),g); c,vc=normalize_prompt(-x,g); np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12); np.testing.assert_allclose(a,c,rtol=0,atol=0); np.testing.assert_array_equal(va,vb); np.testing.assert_array_equal(va,vc)

def test_prompt_reference_context_never_target_score_query():
 assert TAP_NAMES[4]=='P' and 4 not in SIDE_INDICES; m=AMCFModel(7,hidden=8); assert 4 not in m.target_indices
 with pytest.raises(ValueError,match='Prompt'): m.predict_one(torch.zeros(12,18),torch.zeros(9,7),[3,4],4)

def test_low_prompt_only_raw_row_rejected_finite():
 x=iq(8); x[3,4]=0; z,v=normalize_prompt(x,PromptGate(.1,1e-9)); assert v.sum()==7 and not v[3]; assert np.isfinite(z[v]).all() and np.isnan(z[~v]).all()

def test_all_valid_raw_rows_causal_one_second_and_split_safe():
 t=np.array([.01,.20,.51,.99,1.01,1.49,250.01,250.4,250.8]); p=np.ones(len(t),int); recs,q=build_causal_windows(iq(len(t)),t,p,recording_id='cleanStatic',gate=PromptGate(.01,1e-12)); assert set(i for r in recs for i in r.source_indices)==set(q['unique_used_source_indices']); assert q['future_rows']==q['split_boundary_crossings']==0
 for r in recs:
  assert np.all(t[r.source_indices]>r.end_s-1) and np.all(t[r.source_indices]<=r.end_s); assert {assign_clean_role(t[i]) for i in r.source_indices}=={r.role}

def test_history_previous_only_same_role_no_future_split():
 t=np.arange(.1,8,.2); recs,_=build_causal_windows(iq(len(t)),t,np.ones(len(t)),recording_id='r',gate=PromptGate(.01,1e-12))
 for i,r in enumerate(recs):
  ids=history_indices(recs,i,12); assert i not in ids and len(ids)<=12; assert all(recs[j].end_s<r.end_s and recs[j].role==r.role and recs[j].prn==r.prn for j in ids)

def test_prn_permutation_variable_n_input_permutation():
 t=np.r_[np.arange(.1,2,.2),np.arange(.15,1.2,.2)]; p=np.r_[np.ones(10,int),np.full(6,7,int)]; x=iq(len(t)); a,_=build_causal_windows(x,t,p,recording_id='r',gate=PromptGate(.01,1e-12)); o=np.arange(len(t))[::-1]; b,_=build_causal_windows(x[o],t[o],99-p[o],recording_id='r',gate=PromptGate(.01,1e-12)); assert sorted((r.end_s,len(r.source_indices)) for r in a)==sorted((r.end_s,len(r.source_indices)) for r in b)

def test_summary_manual_same_fair_dimension():
 z=np.array([[1+1j,2+0j],[3+1j,4+0j]]); o=complex_summary(z,.5); assert o.shape==(2,7); assert o[0,0]==2 and o[0,1]==1 and o[0,2]==pytest.approx(1.4826) and o[0,5]==pytest.approx(abs(((1+1j)/abs(1+1j)+(3+1j)/abs(3+1j))/2)) and o[0,6]==.5

def batch():
 g=torch.Generator().manual_seed(4); return torch.randn(5,12,18,generator=g),torch.randn(5,9,7,generator=g),torch.tensor([[1,0,0,1,0,1,0,0,0]]*5,dtype=torch.bool),torch.tensor([0,1,2,6,8])

def test_mask_hidden_no_leak_gradients_finite():
 torch.manual_seed(8); m=AMCFModel(7,hidden=8); h,x,mask,target=batch(); x2=x.clone(); x2[~mask]=1e7; a=m(h,x,mask,target); b=m(h,x2,mask,target); torch.testing.assert_close(a[0],b[0]); torch.testing.assert_close(a[1],b[1]); student_t_nll(x[torch.arange(5),target],*a).mean().backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

class ObservedOnly(Mapping):
 def __init__(self,d): self.d=d
 def __getitem__(self,k):
  if k not in self.d: raise AssertionError('oracle hidden access')
  return self.d[k]
 def __iter__(self): return iter(self.d)
 def __contains__(self,k): return k in self.d
 def __len__(self): return len(self.d)

def test_selector_oracle_guard_manual_ig():
 class Fake:
  feature_dim=1; df=4.
  def distribution(self,h,p,o,t): return np.array([0.]),np.array([.5 if 0 in o else 1+t*.1])
 f=Fake(); obs=ObservedOnly({3:np.array([0.]),5:np.array([0.])}); gain=expected_information_gain(f,np.zeros((0,18)),np.zeros(2),obs,[0,1],mc_samples=8,seed=5,return_all=True); manual=sum(student_t_entropy(np.array([1+.1*j]),4).item() for j in [0,1])-student_t_entropy(np.array([.5]),4).item(); assert gain[0]==pytest.approx(manual)
 guard=HiddenAccessGuard(np.zeros((9,1)),{3,5})
 with pytest.raises(RuntimeError,match='hidden'): _=guard[0]

def test_reveal_after_pre_reveal_score():
 class Fake:
  df=4.; feature_dim=1
  def distribution(self,h,p,o,t): return np.array([0.]),np.array([1.])
 o={3:np.array([1.])}; full=np.arange(9.)[:,None]; r=score_then_reveal(Fake(),np.zeros((0,18)),np.zeros(2),o,full,2); assert r['observed_before']==[3] and 2 in o; assert r['nll']==pytest.approx(student_t_nll(np.array([2.]),np.array([0.]),np.array([1.])).mean())

def test_checkpoint_roundtrip_identical_output_state(tmp_path):
 torch.manual_seed(10); m=AMCFModel(7,hidden=8); opt=torch.optim.AdamW(m.parameters()); h,x,mask,target=batch(); out=m(h,x,mask,target); student_t_nll(x[torch.arange(5),target],*out).mean().backward(); opt.step(); p=tmp_path/'m.pt'; checkpoint_save(p,m,opt,{'epoch':1}); n=AMCFModel(7,hidden=8); o2=torch.optim.AdamW(n.parameters()); meta=checkpoint_load(p,n,o2,'cpu');
 for a,b in zip(m(h,x,mask,target),n(h,x,mask,target)): torch.testing.assert_close(a,b,rtol=0,atol=0)
 assert meta['epoch']==1 and opt.state_dict()['state'].keys()==o2.state_dict()['state'].keys()

def test_three_alarm_independent_saved_recompute_pure():
 rows=[{'score':x} for x in [0.,2.,4.]]; got=alarm_columns(rows,1.,3.,2.5); assert [r['alarm_primary_q99'] for r in got]==[False,True,True]; assert [r['alarm_primary_q995'] for r in got]==[False,False,True]; assert [r['alarm_matched_clean_diagnostic'] for r in got]==[False,False,True]; assert verify_alarm_columns(got,1.,3.,2.5)==1.; assert 'alarm_primary_q99' not in rows[0]

def test_all9_order_invariant_leave_one_side_out():
 class Fake:
  df=4.
  def distribution(self,h,p,o,t): return np.zeros(1),np.ones(1)
 v=np.arange(9.)[:,None]; assert all9_score(Fake(),np.zeros((0,18)),np.zeros(2),v,list(SIDE_INDICES))==pytest.approx(all9_score(Fake(),np.zeros((0,18)),np.zeros(2),v,list(reversed(SIDE_INDICES))))

def test_epoch_random_deterministic_hash_value_blind():
 a=epoch_random_extras('DS1',10.5,7,101,4); assert a==epoch_random_extras('DS1',10.5,7,101,4,object()) and len(set(a))==4 and 4 not in a; assert a!=epoch_random_extras('DS1',11.,7,101,4)

def test_attack_calibration_reject_phase_destroy_magnitude():
 with pytest.raises(ValueError,match='cleanStatic'): attack_free_thresholds([1,2],['calibration']*2,['cleanStatic','DS1'])
 z=np.exp(1j*np.arange(18).reshape(2,9))*np.arange(1,19).reshape(2,9); d=phase_destroy(z,2); np.testing.assert_allclose(abs(z),abs(d)); np.testing.assert_array_equal(d,phase_destroy(z,2))


def test_corrected_phase_masks_use_window_containment():
 m=phase_masks([79.5,80.,100.,100.5,120.,120.5,140.,140.5],100.)
 assert m["stable_pre"].tolist()==[True,True,False,False,False,False,False,False]
 assert m["transition"].tolist()==[False,False,True,False,False,False,False,False]
 assert m["ramp"].tolist()==[False,False,False,True,True,False,False,False]
 assert m["takeover"].tolist()==[False,False,False,False,False,True,True,False]
 assert m["persistent"].tolist()==[False,False,False,False,False,False,False,True]


# AMCF-R1 correction gap regression tests (strict TDD additions).
def test_full_history_only_and_deterministic_training_mask_mixture():
 from gnss_doppler_lab.amcf_r1 import full_history_indices, deterministic_training_masks
 t=np.arange(.1,10,.1); recs,_=build_causal_windows(iq(len(t)),t,np.ones(len(t)),recording_id='r',gate=PromptGate(.01,1e-12))
 ids=full_history_indices(recs,12)
 assert len(ids) and all(len(history_indices(recs,i,12))==12 for i in ids)
 a=deterministic_training_masks(24,101,0); b=deterministic_training_masks(24,101,0)
 for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
 mask,target=a
 assert mask.shape==(24,9) and set(target.tolist())<=set(SIDE_INDICES)
 assert np.all(~mask[np.arange(24),target]) and np.all(~mask[:,4])
 # Mixture includes E/L leave-one-out and extra observed side sets.
 assert any(mask[:,j].any() for j in (0,1,2,6,7,8))
 assert set(mask.sum(1).tolist()) >= {1,2}


def test_train_uses_prompt_and_restores_best_optimizer_audit():
 from gnss_doppler_lab.amcf_r1 import train_model
 rng=np.random.default_rng(19); cur=rng.normal(size=(8,9,7)).astype('f4'); hist=rng.normal(size=(8,12,18)).astype('f4'); pc=rng.normal(size=(8,3)).astype('f4')
 model,opt,history,audit=train_model(cur,hist,cur,hist,train_prompt_context=pc,val_prompt_context=pc,seed=101,hidden=8,epochs=2,patience=1)
 assert {'best_epoch','epochs_run','early_stopped','max_epoch_reached','finite','best_restored'}<=set(audit)
 assert audit['max_epoch_reached']==(audit['epochs_run']==2)
 assert audit['early_stopped'] != audit['max_epoch_reached']
 assert audit['optimizer_best_restored'] is True
 assert not audit.get('converged',False) if audit['max_epoch_reached'] else True
 assert opt.state_dict()['state']


def test_phase_destroy_permutes_each_tap_phase_marginal_and_temporal_hash_shuffle():
 from gnss_doppler_lab.amcf_r1 import phase_destroy,temporal_shuffle
 rng=np.random.default_rng(7); z=(1+rng.random((31,9)))*np.exp(1j*rng.normal(size=(31,9)))
 d=phase_destroy(z,41)
 np.testing.assert_allclose(abs(d),abs(z),rtol=0,atol=1e-14)
 for j in range(9): np.testing.assert_allclose(np.sort(np.angle(d[:,j])),np.sort(np.angle(z[:,j])),rtol=0,atol=1e-14)
 assert np.array_equal(d,phase_destroy(z,41))
 h=np.arange(12*3).reshape(12,3); a=temporal_shuffle(h,101,recording_id='DS1',time_s=10.5,prn=7); b=temporal_shuffle(h,101,recording_id='DS1',time_s=10.5,prn=7)
 assert np.array_equal(a,b) and not np.array_equal(a,h) and not np.array_equal(a,h[::-1])
 np.testing.assert_array_equal(np.sort(a,axis=0),np.sort(h,axis=0))


def test_batched_ig_cuda_api_matches_scalar_reference_and_value_blind():
 from gnss_doppler_lab.amcf_r1 import expected_information_gain_batch
 torch.manual_seed(3); dev='cuda' if torch.cuda.is_available() else 'cpu'; m=AMCFModel(2,hidden=8).to(dev).eval()
 h=np.random.default_rng(2).normal(size=(3,12,18)).astype('f4'); p=np.zeros((3,3),'f4'); values=np.random.default_rng(5).normal(size=(3,9,2)).astype('f4')
 masks=np.zeros((3,9),bool); masks[:,3]=masks[:,5]=True; cand=[0,1,2]
 gains=expected_information_gain_batch(m,h,p,values,masks,cand,mc_samples=3,seeds=[7,11,13],chunk_size=10000)
 assert gains.shape==(3,3) and np.isfinite(gains).all()
 # Candidate true values are ignored: changing all hidden values cannot affect gains.
 changed=values.copy(); changed[:,cand]=1e6
 gains2=expected_information_gain_batch(m,h,p,changed,masks,cand,mc_samples=3,seeds=[7,11,13],chunk_size=10000)
 np.testing.assert_allclose(gains,gains2,rtol=2e-5,atol=2e-5)
 # Batch row agrees with independently evaluated scalar row under the same seed.
 one=expected_information_gain_batch(m,h[:1],p[:1],values[:1],masks[:1],cand,mc_samples=3,seeds=[7],chunk_size=10000)
 np.testing.assert_allclose(gains[0],one[0],rtol=2e-5,atol=2e-5)


def test_pure_metric_evaluator_complete_and_nonmutating():
 from gnss_doppler_lab.amcf_r1 import evaluate_detector
 rows=[]
 for i,t in enumerate(np.arange(30,151,.5)):
  ph='stable_pre' if t<80 else ('post' if t<140 else 'persistent')
  rows.append({'scenario':'DS1','decision_time_s':float(t),'phase':ph,'score':float((t>=100)+i/10000)})
 before=[dict(x) for x in rows]
 m,boot=evaluate_detector(rows,.8,scenario='DS1',bootstrap_reps=8,seed=9)
 assert rows==before
 required={'stable_pre_fpr','roc_auc','pr_auc','post_detection','persistent_detection','first_sustained_3_delay_s','q99','q995','comparison','exact_binomial_ci_low','exact_binomial_ci_high'}
 assert required<=set(m) and m['comparison']=='strict_greater' and boot


def test_window_qa_json_required_and_b0_scores_only_contract(tmp_path):
 import csv
 r=__import__('importlib.util').util.spec_from_file_location('rr',Path(__file__).resolve().parents[1]/'scripts/run_amcf_r1_texbat.py'); mod=__import__('importlib.util').util.module_from_spec(r); r.loader.exec_module(mod)
 assert 'window_qa.json' in mod.REQUIRED_OUTPUTS and 'window_qa.csv' not in mod.REQUIRED_OUTPUTS
 p=tmp_path/'b0.csv'; p.write_text('decision_time_s,score_B0_Exact,alarm\n350,1.5,true\n420,3.5,false\n')
 rows=mod.load_b0_exact(p,'cleanStatic')
 assert rows==[{'scenario':'cleanStatic','decision_time_s':350.0,'score':1.5},{'scenario':'cleanStatic','decision_time_s':420.0,'score':3.5}]


def test_policy_seed_contract_and_eval_limit_not_defaulted():
 import importlib.util
 p=Path(__file__).resolve().parents[1]/'scripts/run_amcf_r1_texbat.py'; spec=importlib.util.spec_from_file_location('rp',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 assert m.POLICY_SEEDS==(11,23,37)
 args=m.parser().parse_args(['--out','/tmp/x','--max-train-samples','4'])
 m.normalize_args(args)
 assert args.max_eval_samples is None


def test_batched_ig_matches_manual_scalar_mc_formula():
 import hashlib,json
 from gnss_doppler_lab.amcf_r1 import expected_information_gain_batch,student_t_entropy
 torch.manual_seed(23);m=AMCFModel(1,hidden=8).eval();rng=np.random.default_rng(4);h=rng.normal(size=(1,12,18)).astype('f4');p=rng.normal(size=(1,3)).astype('f4');v=rng.normal(size=(1,9,1)).astype('f4');mask=np.zeros((1,9),bool);mask[:,3]=mask[:,5]=True;cand=[0,1,2];seed=17;mc=3
 got=expected_information_gain_batch(m,h,p,v,mask,cand,mc_samples=mc,seeds=[seed])[0]
 obs={3:v[0,3],5:v[0,5]};base={j:m.distribution(h[0],p[0],obs,j) for j in cand};total=sum(float(np.sum(student_t_entropy(x[1],m.df))) for x in base.values());manual=[]
 for j in cand:
  mu,sc=base[j];dig=hashlib.sha256(json.dumps([seed,j,mc],separators=(',',':')).encode()).digest();rr=np.random.default_rng(int.from_bytes(dig[:8],'little'));samples=mu+sc*rr.standard_t(m.df,size=(mc,len(mu)));remaining=[]
  for sample in samples:
   augmented=dict(obs);augmented[j]=sample;remaining.append(sum(float(np.sum(student_t_entropy(m.distribution(h[0],p[0],augmented,k)[1],m.df))) for k in cand if k!=j))
  manual.append(total-float(np.mean(remaining)))
 np.testing.assert_allclose(got,manual,rtol=2e-5,atol=2e-5)


def test_evaluator_reports_exact_fpr_intervals_separately():
 from gnss_doppler_lab.amcf_r1 import evaluate_detector
 rows=[{'scenario':'DS1','decision_time_s':i*.5,'phase':'stable_pre' if i<20 else 'post','score':float(i%3)} for i in range(40)]
 m,_=evaluate_detector(rows,1.,scenario='DS1',bootstrap_reps=2,onset_s=10.)
 assert {'stable_pre_exact_binomial_ci_low','stable_pre_exact_binomial_ci_high','post_exact_binomial_ci_low','persistent_exact_binomial_ci_low'}<=set(m)
