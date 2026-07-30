from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/"scripts"/"train_peak_floor_dynamic_innovation_copula.py"
def load():
 s=importlib.util.spec_from_file_location("pf_dic_train",P); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); return m

def test_masked_epoch_summary_ignores_padding_and_prn_order():
 m=load(); x=torch.randn(2,4,m.MAX_PRNS,5); mask=torch.zeros(2,4,m.MAX_PRNS,dtype=torch.bool); mask[:,:,:3]=True
 a=m.masked_peak_summary(x,mask); y=x.clone(); y[:,:,3:]=1e8
 assert torch.allclose(a,m.masked_peak_summary(y,mask))
 perm=torch.randperm(m.MAX_PRNS)
 assert torch.allclose(a,m.masked_peak_summary(x[:,:,perm],mask[:,:,perm]),atol=1e-6)

def test_nll_finite_and_singleton_model_output_separate_branches():
 m=load(); y=torch.randn(1,7); mu=torch.zeros_like(y); log_s=torch.full_like(y,-100.)
 assert torch.isfinite(m.gaussian_nll(y,mu,log_s)) and torch.isfinite(m.student_t_nll(y,mu,log_s,nu=5.))
 cfg=m.ModelConfig(peak_input_dim=10,floor_input_dim=7,hidden_dim=12,layers=1)
 model=m.DynamicInnovationModel(cfg); out=model(torch.randn(1,3,10),torch.randn(1,3,7))
 assert out["peak_mean"].shape==(1,10) and out["floor_mean"].shape==(1,7)

def test_correlated_aligned_relation_has_lower_nll_than_permuted():
    m=load(); rng=np.random.default_rng(4); x=rng.normal(size=(500,8)); y=.75*x+rng.normal(scale=.45,size=(500,8))
    rel=m.fit_relation(x,y,rank=4,shrinkage=.05)
    aligned=m.relation_nll(x,y,rel); shifted=m.relation_nll(x,np.roll(y,137,axis=0),rel)
    assert aligned.mean()<shifted.mean() and np.linalg.eigvalsh(rel["R"]).min()>0

def test_relation_deviation_is_two_sided_and_permutation_shifts_have_guard_band():
    m=load(); signed=np.array([-3.,-1.,0.,1.,3.])
    assert np.array_equal(m.relation_deviation_scores(signed,0.),np.array([3.,1.,0.,1.,3.]))
    shifts=m.valid_circular_shifts(n=108,guard=13,max_shifts=99)
    assert len(shifts)>19 and min(shifts)>=13 and max(shifts)<=95
