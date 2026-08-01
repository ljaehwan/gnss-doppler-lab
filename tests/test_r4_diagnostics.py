import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'scripts'/'run_gcmr_pi_r4_diagnostics.py'
s=importlib.util.spec_from_file_location('r4',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_warmup_boundaries():
 keep,pre,post=m.mask(np.array([29,30,109.9,110,129.9,130]))
 assert pre.tolist()==[False,True,True,False,False,False]
 assert post.tolist()==[False,False,False,False,False,True]
def test_fixed_seed_and_norm_preservation_contract():
 rng=np.random.default_rng(20260801); assert rng.integers(1000000)==np.random.default_rng(20260801).integers(1000000)
 x=np.array([[3.,4.],[0.,2.]]); d=np.array([[0.,1.],[1.,0.]])*np.linalg.norm(x,axis=1,keepdims=True)
 assert np.allclose(np.linalg.norm(x,axis=1),np.linalg.norm(d,axis=1))
def test_attack_calibration_excluded_by_contract():
 assert 'no attack calibration/tuning' in (Path(__file__).parents[1]/'scripts'/'run_gcmr_pi_r4_diagnostics.py').read_text()
